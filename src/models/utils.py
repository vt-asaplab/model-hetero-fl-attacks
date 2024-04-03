import torch.nn as nn


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.uniform_(to=0.0005)
        m.weight.data.uniform_(to=0.0005)
    # elif isinstance(m, nn.Conv2d):
    #    m.bias.data.zero_()
    # elif isinstance(m, nn.MaxPool2d):
        # m.bias.data.zero_()
    #    m.weight.data.fill_(1)
    return m