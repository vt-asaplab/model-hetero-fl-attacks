import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
import numpy as np

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(constants.INPUT_DIM, constants.HIDDEN_LAYERS)
    self._init_weights_2(self.linear1)
    self.linear2 = nn.Linear(constants.HIDDEN_LAYERS, constants.OUTPUT_DIM)
    self._init_weights_2(self.linear2)

  def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(1)

  def _init_weights_2(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, a=0.0, b=0.0005)
            if module.bias is not None:
                torch.nn.init.uniform_(module.bias, a=0.0, b=0.0005)
                # module.bias.data()

  # def _init_weights_trap(self, module):
    # TODO: Implement trap weights. 
    # num_elements = module.bias.size()[0]
    # print("TRAP INIT: num_elements = %s"%(num_elements))
    # down_scale_factor = 0.95
    # vector = np.random.normal(0, 0.5, num_elements)
    # abs_vector = abs(vector) * (-1)
    # num_pos = np.floor(num_elements / 2).astype(int)

    # # random positive indices
    # pos_indices = np.random.choice(num_elements, num_pos, replace=False)
    # negative_elements = np.delete(abs_vector, pos_indices)
    # abs_vector[pos_indices] = -down_scale_factor * negative_elements  # set the negative values and turn them positive


  def forward(self, x):
    x = F.relu(self.linear1(x), inplace=True)
    x = self.linear2(x)
    output = F.log_softmax(x, dim=1)
    return output

class Network_Scaled(nn.Module):
  def __init__(self, beta_n):
    # beta_n defines the percentage of nodes to keep for a given layer. 
    super(Network_Scaled, self).__init__()
    hidden_nodes = int(constants.HIDDEN_LAYERS * beta_n)
    self.linear1 = nn.Linear(constants.INPUT_DIM, hidden_nodes)
    # self._init_weights(self.linear1)
    self.linear2 = nn.Linear(hidden_nodes, constants.OUTPUT_DIM)
    # self._init_weights(self.linear2)
  
  def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(1)
  
  def forward(self, x):
    x = F.relu(self.linear1(x), inplace=True)
    x = self.linear2(x)
    output = F.log_softmax(x, dim=1)
    return output