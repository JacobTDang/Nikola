import torch.nn as nn


class nikola(nn.Module):
  def __init__(self):
    # calls constructor of parent class to properly set everything up
    super().__init__()

    # sequential is a container that holds a series of layers that data will pass through
    self.conv_layers = nn.Sequential(
    # 2d convolution layer
    nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten()
    )

    self.dense_layers = nn.Sequential(
      nn.Linear(in_features=3840, out_features=100),
      nn.ReLU(),
      nn.Linear(in_features=100, out_features=50),
      nn.ReLU(),
      nn.Linear(in_features=50, out_features=10),
      nn.ReLU(),
      nn.Linear(in_features=10, out_features=2)
    )

  def forward(self, x):
    # 'x' is the input image tensor

    # image goes through all convolution layers
    x = self.conv_layers(x)
    # result from convolution layer is then passed through all the dense layers
    x = self.dense_layers(x)
    # return the results (steering and throttle)
    return x
