from data_backup import Data
import constants
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.datasets import make_classification

# X_train, y_train = make_classification(
#   n_samples=99, n_features=4, n_redundant=0,
#   n_informative=3,  n_clusters_per_class=2, n_classes=3, random_state=42
# )

# Base class for clients: 
class Client:
    # Don't need global params, because the server is responsible for initializing client local models with global parameters. 
    def __init__(self, train_data: Data):
        self.traindata = train_data
        self.trainloader = DataLoader(self.traindata, batch_size=constants.BATCH_SIZE, 
                         shuffle=False, num_workers=2)
        torch.manual_seed(42)

    # Train locally for num_epochs. 
    # Input: num_epochs
    # Output: Tensor of trained parameter weights and biases. 
    def train(self, num_epochs, local_model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)

        # print("TRAIN: self.traindata[:2] = ", self.traindata[:2])

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                print("data in trainloader = ", data)
                inputs, labels = data
                # set optimizer to zero grad to remove previous epoch gradients
                optimizer.zero_grad()
                # forward propagation
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                # backward propagation
                loss.backward()
                # optimize
                optimizer.step()

                running_loss += loss.item()
            # display statistics
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            # print("first 2 training outputs = ", outputs[:2])
        
        # print("local_model state_dict after train = ", local_model.state_dict())
        return local_model.state_dict()
        # pass
