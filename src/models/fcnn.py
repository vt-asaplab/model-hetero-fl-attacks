import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param
from modules import Scaler
import torchsummary
from torchsummary import summary

class Fcnn(nn.Module):
    def __init__(self, data_shape, classes_size, rate=1, track=False):
        # Do something. 
        super().__init__()

        self.flatten = nn.Flatten() # equivalent of keras.layers.Flatten
        self.layers = nn.Sequential(
            nn.Linear(int(data_shape[1]) * int(data_shape[2]), int(5000 * rate)), # input shape, output shape         
            nn.ReLU(), # you have to add the activation function after
            # nn.Linear(512, 512), 
            # nn.ReLU(), 
            nn.Linear(int(5000 * rate), classes_size)
        )

        # print("data_shape[1] = %s and data_shape[2] = %s"%(data_shape[1], data_shape[2]))
        # self.linear1 = nn.Linear(int(data_shape[1]) * int(data_shape[2]), int(1000 * rate))
        # # self.relu1 = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(int(1000 * rate), classes_size)
        # pass

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        x = self.flatten(x)
        out = self.layers(x)
        if 'label_split' in input and cfg['mask']:
            label_mask = torch.zeros(cfg['classes_size'], device=out.device)
            label_mask[input['label_split']] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output['score'] = out
        output['loss'] = F.cross_entropy(out, input['label'], reduction='mean')
        return output
        # pass
        # Do something. 

def fcnn(model_rate=1, track=False):
    data_shape = cfg['data_shape']
    # hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['conv']['hidden_size']]
    classes_size = cfg['classes_size']
    scaler_rate = model_rate / cfg['global_model_rate']
    model = Fcnn(data_shape, classes_size, scaler_rate, track)
    # print("FCNN: Model = ", model)
    model.apply(init_param)
    return model