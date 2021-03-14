import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

  def __init__(
      self,
      in_features: int,
      out_features: int,
  ) -> None:
    super(ExU, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    # self.weights = Parameter(torch.Tensor(out_features, in_features),
    #                          requires_grad=True)
    self.weights = Parameter(torch.Tensor(in_features, out_features),
                             requires_grad=True)

    ## bias should match in_features? or as regular output features?
    self.bias = Parameter(torch.Tensor(in_features), requires_grad=True)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    ## Page(4): initializing the weights using a normal distribution
    ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
    torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
    torch.nn.init.trunc_normal_(self.bias, std=0.5)

  def forward(
      self,
      inputs: torch.Tensor,
      n: int = 1,
  ) -> torch.Tensor:
    ## TODO(amr): check this with nick?!
    ## input shape and bias differ?
    ## e.g. inputs(13, 128) but bias is (1, 128)
    # print(f'Inputs: {inputs.shape}, weights: {self.weights.shape}')
    # print(
    #     f'weights: {torch.exp(self.weights).shape}, Inputs: {(inputs - self.bias).shape}'
    # )
    output = (inputs - self.bias).matmul(torch.exp(self.weights))

    # ReLU activations capped at n (ReLU-n)
    output = torch.clamp(F.relu(output), 0, n)

    return output

  def extra_repr(self):
    return f'in_features={self.in_features}, out_features={self.out_features}'
