from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn

from nam.models.featurenn import FeatureNN, MultiFeatureNN


class NAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str = 'relu'
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.activation = activation

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[i], 
                dropout=self.dropout, 
                hidden_sizes=self.hidden_sizes,
                activation=self.activation
            )
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out).unsqueeze(1)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out


class MultiTaskNAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: list,
        num_units: int,
        num_subnets: int,
        num_tasks: int,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str = 'relu',
    ) -> None:
        super(MultiTaskNAM, self).__init__()

        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_subnets = num_subnets
        self.num_tasks = num_tasks
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.activation = activation

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            MultiFeatureNN(
                    input_shape=1,
                    num_units=self.num_units[i],
                    num_subnets=self.num_subnets,
                    num_tasks=self.num_tasks,
                    dropout=self.dropout,
                    hidden_sizes=self.hidden_sizes,
                    activation=self.activation
                )
            for i in range(self.num_inputs)
        ])
        
        self._bias = torch.nn.Parameter(data=torch.zeros(1, self.num_tasks))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tuple: (batch, num_tasks) x num_inputs
        individual_outputs = self.calc_outputs(inputs)
        # (batch, num_tasks, num_inputs)
        stacked_out = torch.stack(individual_outputs, dim=-1).squeeze(dim=1)
        dropout_out = self.dropout_layer(stacked_out)

        # (batch, num_tasks)
        summed_out = torch.sum(dropout_out, dim=2) + self._bias
        return summed_out, dropout_out

    def feature_output(self, feature_index, inputs):
        return self.feature_nns[feature_index](inputs)
