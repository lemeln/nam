import gc
from joblib import Parallel, delayed
import os
from typing import Callable, Mapping
from typing import Sequence

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from nam.trainer.metrics import Metric
from nam.models.saver import Checkpointer
from nam.trainer.metrics import *
from nam.utils.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split


class Trainer:

    def __init__(self, 
        models: Sequence[nn.Module], 
        dataset: torch.utils.data.Dataset,
        criterion: Callable,
        metric: str,
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        test_split: float = None,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        save_model_frequency: int = 0,
        patience: int = 40,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        regression: bool = True,
        num_learners: int = 1,
        n_jobs: int = None,
        random_state: int = 0
    ) -> None:
        self.models = models
        self.dataset = dataset
        self.criterion = criterion
        self.metric_name = metric.upper() if metric else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.regression = regression
        self.num_learners = num_learners
        self.n_jobs = n_jobs
        self.random_state = random_state
        # Disable tqdm if concurrency > 1
        self.disable_tqdm = n_jobs not in (None, 1)

        self.log_dir = log_dir
        if not self.log_dir:
            self.log_dir = 'output'

        self.val_split = val_split
        self.test_split = test_split

        self._best_checkpoint_suffix = 'best'

        self.setup_dataloaders()
        
    def setup_dataloaders(self):
        test_size = int(self.test_split * len(self.dataset)) if self.test_split else 0
        val_size = int(self.val_split * (len(self.dataset) - test_size))
        train_size = len(self.dataset) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self.dataset, [train_size, val_size, test_size])

        # TODO: Possibly find way not to store data longterm -- maybe use close function
        self.train_dl = DataLoader(train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)

        self.val_dl = DataLoader(val_subset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)

        self.test_dl = None
        if test_size > 0:
            self.test_dl = DataLoader(test_subset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers)

    def train_step(self, batch: torch.Tensor, model: nn.Module, 
                   optimizer: optim.Optimizer, metric: Metric) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""
        features, targets, weights = [t.to(self.device) for t in batch]

        # Resets optimizer's gradients.
        optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = model(features)

        loss = self.criterion(predictions, targets, weights, fnn_out, model)
        self.update_metric(metric, predictions, targets, weights)

        # Backward pass.
        loss.backward()

        # Performs a gradient descent step.
        optimizer.step()

        return loss

    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader, metric: Metric) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on
        `dataloader`."""
        model.train()
        loss = 0.0
        with tqdm(dataloader, leave=False, disable=self.disable_tqdm) as pbar:
            for batch in pbar:
                # Performs a gradient-descent step.
                step_loss = self.train_step(batch, model, optimizer, metric)
                loss += step_loss

        metric_train = None
        if metric:
            metric_train = metric.compute()
            metric.reset()

        return loss / len(dataloader), metric_train

    def evaluate_step(self, model: nn.Module, batch: Mapping[str, torch.Tensor],
                      metric: Metric) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        features, targets, weights = [t.to(self.device) for t in batch]

        # Forward pass from the model.
        predictions, fnn_out = model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, targets, weights, fnn_out, model)
        self.update_metric(metric, predictions, targets, weights)

        return loss

    def evaluate_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                       metric: Metric) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader`."""
        model.eval()
        loss = 0.0
        with tqdm(dataloader, leave=False, disable=self.disable_tqdm) as pbar:
            for batch in pbar:
                # Accumulates loss in dataset.
                with torch.no_grad():
                    step_loss = self.evaluate_step(model, batch, metric)
                    loss += step_loss

        metric_val = None
        if metric:
            metric_val = metric.compute()
            metric.reset()

        return loss / len(dataloader), metric_val

    def train_ensemble(self):
        if self.regression:
            ss = ShuffleSplit(n_splits=self.num_learners, 
                test_size=self.val_split, random_state=self.random_state)
        else:
            ss = StratifiedShuffleSplit(n_splits=self.num_learners, 
                test_size=self.val_split, random_state=self.random_state)

        self.models[:] = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_learner)(i, train_ind, val_ind)
            for i, (train_ind, val_ind) in enumerate(ss.split(self.dataset.X, self.dataset.y)))
        
        return

    def train_learner(self, model_index, train_indices, val_indices):
        # Set random seed for each process to guarantee reproducibility
        torch.manual_seed(self.random_state + model_index)

        model = self.models[model_index]
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        
        train_dl = DataLoader(train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)

        val_dl = DataLoader(val_subset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)

        log_subdir = os.path.join(self.log_dir, str(model_index))
        writer = TensorBoardLogger(log_dir=log_subdir)
        checkpointer = Checkpointer(log_dir=log_subdir)

        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=self.lr,
                                        weight_decay=0)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            gamma=0.995,
                                            step_size=1)

        metric = self.create_metric()

        model = self.train(model, train_dl, val_dl, optimizer, scheduler, writer, checkpointer, metric)
        return model

    def train(self, model, train_dl, val_dl, optimizer, scheduler, writer, checkpointer, metric):
        """Train the model for a specified number of epochs."""
        num_epochs = self.num_epochs
        best_loss_or_metric = float('inf')
        epochs_since_best = 0

        with tqdm(range(num_epochs), disable=self.disable_tqdm) as pbar_epoch:
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train, metric_train = self.train_epoch(model, optimizer, train_dl, metric)
                writer.write({"loss_train_epoch": loss_train.detach().cpu().numpy().item()})
                if metric:
                    writer.write({f"{self.metric_name}_train_epoch": metric_train})

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metric_val = self.evaluate_epoch(model, val_dl, metric)
                writer.write({"loss_val_epoch": loss_val.detach().cpu().numpy().item()})
                if metric:
                    writer.write({f"{self.metric_name}_val_epoch": metric_val})

                scheduler.step()

                # Updates progress bar description.
                desc = f"""Epoch({epoch}):
                    Training Loss: {loss_train.detach().cpu().numpy().item():.3f} |
                    Validation Loss: {loss_val.detach().cpu().numpy().item():.3f}"""
                if metric:    
                    desc += f' | {self.metric_name}: {metric_train:.3f}'
                pbar_epoch.set_description(desc)

                # Checkpoints model weights.
                if self.save_model_frequency > 0 and epoch % self.save_model_frequency == 0:
                    checkpointer.save(model, epoch)

                # Save best checkpoint for early stopping
                loss_or_metric = loss_val if self.monitor_loss else metric_val
                if self.early_stop_mode == 'max':
                    loss_or_metric = -1 * loss_or_metric

                if self.patience > 0 and loss_or_metric < best_loss_or_metric:
                    best_loss_or_metric = loss_or_metric
                    epochs_since_best = 0
                    checkpointer.save(model, self._best_checkpoint_suffix)

                # Stop training if early stopping patience exceeded
                epochs_since_best += 1
                if self.patience > 0 and epochs_since_best > self.patience:
                    return checkpointer.load(self._best_checkpoint_suffix)

            # If early stopping is not used, save last checkpoint as best
            checkpointer.save(model, self._best_checkpoint_suffix)
            return model

    def close(self):
        del self.dataset
        gc.collect()
        return
    
    def create_metric(self):
        if not self.metric_name:
            return None
        if self.metric_name.lower() == 'auroc':
            return AUROC()
        if self.metric_name.lower() == 'accuracy':
            return Accuracy(input_type='logits')
        if self.metric_name.lower() == 'avgprecision':
            return AveragePrecision()
        if self.metric_name.lower() == 'mse':
            return MeanSquaredError()
        if self.metric_name.lower() == 'rmse':
            return RootMeanSquaredError()
        if self.metric_name.lower() == 'mae':
            return MeanAbsoluteError()

    def update_metric(self, metric, predictions, targets, weights):
        if metric:
            predictions, targets = predictions.view(-1), targets.view(-1)
            indices = weights.view(-1) > 0
            predictions, targets = predictions[indices], targets[indices]
            metric.update(predictions, targets)
