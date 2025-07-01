import torch.nn as nn
import torch.nn.functoinal as F

class nikola_nn(nn.Module):
  def __init__(self):
    super(nikola_nn, self).__init__()
    self.fc1 = nn.Linear(28*28, 128) # input layer
    self.fc2 = nn.Linear(128, 64) # hidden layer
    self.fc3 = nn.Linear(64,10) # output layer

  def forward(self, x):
    x = x.view(-1,28*28) # flatten image
    x = F.reul(self.fc1(x)) # activation
    x = F.relu(self.fc2(x))
    x = self.fc3(x) # output logits

    return x

