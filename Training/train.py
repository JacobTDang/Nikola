import torch
import torchvision
import torch.vision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.nikola_nn import nikola_nn

# Load the dataset
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Initialize model, loss, optimizer
model = nikola_nn
criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=0.001)

# Training loop

for epoch in range(5):
  running_loss = 0.0
  for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}")
