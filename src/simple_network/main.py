# Step 1: Create the model. 
# Model Architecture: 4 neurons in input, 6 neurons in middle layer, 3 neurons in the output layer. 

import model
from client import Client
from server import Server
from data_backup import Data
import constants
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.datasets import make_classification

# X_train, y_train = make_classification(
#   n_samples=300, n_features=4, n_redundant=0,
#   n_informative=3,  n_clusters_per_class=2, n_classes=3, random_state=42
# )

# Set the seeds. 
np.random.seed(42)
torch.manual_seed(42)

# Wrap the data inside of a Data() object. 
# traindata = Data(X_train, y_train)

# trainloader = DataLoader(traindata, batch_size=constants.BATCH_SIZE, 
#                          shuffle=True, num_workers=2)

clf = model.Network() # clf will be W_0. 
# print(clf.parameters)

# Now, initialize the clients and the server. 
client_capacities = [1]
client_list = []
seed = 41

for i in range(constants.NUM_CLIENTS):
  seed = seed + 1
  X_train, y_train = make_classification(
    n_samples=3, n_features=4, n_redundant=0,
    n_informative=3,  n_clusters_per_class=2, n_classes=3, random_state=seed
  )

  client_traindata = Data(X_train, y_train)

  # client_traindata = Data()
  # client_traindata = Data(traindata[(constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * i : (constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * (i+1)], )
  # print("client_traindata = ", client_traindata)
  # print("client_traindata = ", client_traindata)
  c = Client(client_traindata)
  client_list.append(c)

  # Now train for the client. 
  local_model = copy.deepcopy(clf)
  updated_state_dict = c.train(constants.NUM_EPOCHS, local_model)
  print("updated_state_dict = %s and orig state_dict = %s"%(updated_state_dict, clf.state_dict()))
  weight_grad = torch.subtract(clf.state_dict()['linear1.weight'], updated_state_dict['linear1.weight'])
  bias_grad = torch.subtract(clf.state_dict()['linear1.bias'], updated_state_dict['linear1.bias'])

  for i in range(weight_grad.size()[0]):
    # Print out the recovered inputs. 
    if (bias_grad[i] != 0.0):
      partial_recon = torch.divide(weight_grad[i], bias_grad[i])
      print("Partial_recon for i = %s = %s"%(i, partial_recon))


# Now, initialize the server. 
# aggregator = Server(clf, client_list, client_capacities)
# aggregator.fed_training() # Perform federated training. 

'''
for j in range(constants.NUM_ROUNDS):
    for i in range(constants.NUM_CLIENTS):
      # Client i's data will be traindata[33i : 33(i+1)]
      client_traindata = traindata[(constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * i : (constants.NUM_DATA_SAMPLES // constants.NUM_CLIENTS) * (i+1)]
      client_model = model.Network_Scaled(client_capacities[i])

      # print("Client ", i, "state dict = ", client_model.state_dict())

      # Now, copy over the appropriate entries in the global state_dict to each individual client's state dict. 
      # print("Global state_dict = ", clf.state_dict())
      global_state_dict = clf.state_dict()
      local_state_dict = copy.deepcopy(clf.state_dict())

      j_hat = j % (constants.HIDDEN_LAYERS)
      
        # local_state_dict['linear1.weight'] = 
      # local_state_dict['linear1.weight'] = local_state_dict['linear1.weight'][j:]

      for key in local_state_dict.keys():
        if (key == 'linear1.weight'):
          if ((j_hat + client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
            local_state_dict[key] = global_state_dict[key][j_hat:j_hat+int(client_capacities[i]*constants.HIDDEN_LAYERS), :]
            # print("IF CONDITION lin1: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
          else:
            first_part = global_state_dict[key][:j_hat + int(client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
            second_part = global_state_dict[key][j_hat:constants.HIDDEN_LAYERS]
            local_state_dict[key] = torch.cat((first_part, second_part))
            # print("ELSE CONDITION lin1: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
        elif (key == 'linear2.weight'):
          if ((j_hat + client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
            local_state_dict[key] = global_state_dict[key][:,j_hat:j_hat+int(client_capacities[i]*constants.HIDDEN_LAYERS) ]
            # print("IF CONDITION lin2: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
          else:
            first_part = global_state_dict[key][:, :j_hat + int(client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
            second_part = global_state_dict[key][:, j_hat:constants.HIDDEN_LAYERS]
            local_state_dict[key] = torch.cat((first_part, second_part), dim=1)
            # print("ELSE CONDITION lin2: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
        elif (key == 'linear1.bias'):
          if ((j_hat + client_capacities[i] * constants.HIDDEN_LAYERS) <= constants.HIDDEN_LAYERS):
            local_state_dict[key] = global_state_dict[key][j_hat:j_hat+int(client_capacities[i]*constants.HIDDEN_LAYERS)]
            # print("IF CONDITION lin1 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
          else:
            first_part = global_state_dict[key][:j_hat + int(client_capacities[i] * constants.HIDDEN_LAYERS) - constants.HIDDEN_LAYERS]
            second_part = global_state_dict[key][j_hat:constants.HIDDEN_LAYERS]
            # print("ELSE CONDITION lin1 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
        else:
          print(key)
          # print("lin2 bias: local_state_dict for client %s after slicing = %s"%(i, local_state_dict[key]))
      
      # Now, copy over state dicts for each client (use load_state_dict). 
      client_model.load_state_dict(local_state_dict)
      print("Client %s state dict = %s"%(i, client_model.state_dict()))

      # c = Client(client_traindata, client_model)
      # client_list.append(c)
'''


# Now, split up the training data amongst the 3 different clients. 
# print("train_data[:1] = ", traindata[:1])
# client_1_data = traindata[:33]
# client_2_data = traindata[33:66]
# client_3_data = traindata[66:99]

# print("clf type = ", type(clf))
# print("clf state dict before = ", clf.state_dict())

# model = MyModel()
# clf.linear1.register_forward_hook(get_activation('linear1'))
# clf.linear1.register_forward_hook(get_activation('linear2'))
# output = clf(X_train)

# for name, module in clf.named_modules():
#   print("name = ", name)
#   print("module = ", module)

# epochs = 2
# for epoch in range(epochs):
#   running_loss = 0.0
#   for i, data in enumerate(trainloader, 0):
#     inputs, labels = data
#     # set optimizer to zero grad to remove previous epoch gradients
#     optimizer.zero_grad()
#     # forward propagation
#     outputs = clf(inputs)
#     loss = criterion(outputs, labels)
#     # backward propagation
#     loss.backward()
#     # optimize
#     optimizer.step()
#     running_loss += loss.item()
#   # display statistics
#   print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
  # print("first 2 training outputs = ", outputs[:2])

# print("clf state dict after = ", clf.state_dict())

# print("Printing activation for linear1... ")
# print(activation['linear1'])

# print("Printing activation for linear2... ")
# print(activation['linear2'])