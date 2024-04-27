from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_loss(loss_func: Callable, logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    loss = loss_func(logits, targets, reduction='none')
    loss *= weights
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.sum(weights, dim=0)
    return loss


def reg_penalty(fnn_out: torch.Tensor, model: nn.Module,
    output_regularization: float, l2_regularization: float
) -> torch.Tensor:
    """Computes penalized loss with L2 regularization and output penalty.

    Args:
      config: Global config.
      model: Neural network model.
      inputs: Input values to be fed into the model for computing predictions.
      targets: Target values containing either real values or binary labels.

    Returns:
      The penalized loss.
    """

    def features_loss(per_feature_outputs):
        b, f = per_feature_outputs.shape[0], per_feature_outputs.shape[2]
        out = (per_feature_outputs ** 2).sum(dim=0).sum(dim=1) / (b * f)
        return output_regularization * out

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = len(model.feature_nns)
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    reg_loss = 0.0
    if output_regularization > 0:
        reg_loss += features_loss(fnn_out)

    if l2_regularization > 0:
        reg_loss += l2_regularization * weight_decay(model)

    return reg_loss


def make_penalized_loss_func(loss_func, regression, output_regularization, l2_regularization):
    def penalized_loss_func(logits, targets, weights, fnn_out, model):
        loss = weighted_loss(loss_func, logits, targets, weights)
        loss += reg_penalty(fnn_out, model, output_regularization, l2_regularization)
        loss = torch.sum(loss) # Sum over tasks
        return loss

    if not loss_func:
        loss_func = F.mse_loss if regression else F.binary_cross_entropy_with_logits
    return penalized_loss_func


