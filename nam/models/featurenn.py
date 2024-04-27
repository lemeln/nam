import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import ExU
from .activation import LinReLU


class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          input_shape: Dimensionality of input data.
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          hidden_sizes: List of hidden dimensions for each layer.
          activation: Activation function of first layer (relu or exu).
        """
        super(FeatureNN, self).__init__()
        self.input_shape = input_shape
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        all_hidden_sizes = [self.num_units] + self.hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)

        ## First layer is ExU
        if self.activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs


class MultiFeatureNN(torch.nn.Module):
    def __init__(
        self,
        input_shape: int,
        num_units: int,
        num_subnets: int,
        num_tasks: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.
        Args:
            input_shape: Dimensionality of input data.
            num_units: Number of hidden units in first hidden layer.
            num_tasks: Number of tasks.
            num_subnets: Number of subnets.
            dropout: Coefficient for dropout regularization.
            hidden_sizes: List of hidden dimensions for each layer.
            activation: Activation function of first layer (relu or exu).
        """
        super(MultiFeatureNN, self).__init__()
        subnets = [
            FeatureNN(
                input_shape=input_shape,
                num_units=num_units,
                dropout=dropout,
                hidden_sizes=hidden_sizes,
                activation=activation
            )
            for i in range(num_subnets)
        ]
        self.feature_nns = nn.ModuleList(subnets)
        self.linear = torch.nn.Linear(num_subnets, num_tasks)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training mode."""
        individual_outputs = []
        for fnn in self.feature_nns:
            individual_outputs.append(fnn(inputs)) 

        # (batch_size, num_subnets)
        stacked = torch.stack(individual_outputs, dim=-1)
        # (batch_size, num_tasks)
        weighted = self.linear(stacked)
        return weighted