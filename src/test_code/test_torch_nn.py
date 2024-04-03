import seaborn as sns
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
# import torchview
# from torchview import draw_graph

NUM_FEATURES = 4
NUM_CLASSES = 3

# print(sns.get_dataset_names())
penguins = sns.load_dataset('penguins')

print(type(penguins))
print("penguins length before = ", len(penguins))

sns.set(rc={'figure.figsize':(10,7)})

np.random.seed(42)
torch.manual_seed(42)

penguins = penguins.dropna()
print("penguins length after = ", len(penguins))
print(penguins.head())

print(penguins.species.value_counts())

(train_set, test_set) = train_test_split(penguins, test_size=0.2)

print("train_set_shape before = ", train_set.shape)
print("test_set_shape before = ", test_set.shape)

train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

print("train_set_shape after = ", train_set.shape)
print("test_set_shape after = ", test_set.shape)

# print("train_set_head after = ", train_set.head())
# print("test_set_head after = ", test_set.head())

SPECIES_MAP = {
    'Adelie': 0,
    'Chinstrap': 1, 
    'Gentoo': 2
}

def create_dataset(data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        data[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].to_numpy(),
        dtype=torch.float
    )
    labels=torch.tensor(data.species.map(SPECIES_MAP), dtype=torch.long)
    return (features, labels)

(X_train, y_train) = create_dataset(train_set)
(X_test, y_test) = create_dataset(test_set)
print(X_train[:2])
print(y_train[:2])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

class PenguinClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear_layer_1 = nn.Linear(n_features, 8)
        # self._init_weights(self.linear_layer_1)
        self.linear_layer_2 = nn.Linear(8, n_classes)
        # self._init_weights(self.linear_layer_2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, features):
        x = torch.relu(self.linear_layer_1(features))
        return self.linear_layer_2(x)

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

model = PenguinClassifier(n_features=NUM_FEATURES, n_classes=NUM_CLASSES)
# model.linear_layer_1.register_forward_hook(get_activation('linear_layer_1'))
# model.linear_layer_2.register_forward_hook(get_activation('linear_layer_2'))
# x = torch.randn(1, 4)
# output = model(x)
X_1 = X_train[:1]
# output = model(X_1)

training_res = model(X_train[:2])
print(training_res[:2])

# testing_res = model(X_test)
# print(testing_res[:2])

# print("activation ll1 = ", activation['linear_layer_1'])
# print("activation ll2 = ", activation['linear_layer_2'])

