
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from sklearn.model_selection import train_test_split

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read csv files
train = pd.read_csv('train.csv')
test = pd.resd_csv('test.csv')

# set Hyperparameters
batch_size = 64
num_classes = 10
learning_rate = 1e-4
num_epochs = 10

# data augmentation
transforms = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.RandomRotation(degrees=10),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5,), std=(0.5,))          
])

# create the dataset
class Custom_Dateset(Dataset):
  def __init__(self, data, transform=None):

    self.data = data.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)
    self.labels = torch.from_numpy(data.label.values)
    self.transform = transforms 

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.transform(self.data[idx]), self.labels[idx]


# splitting the data with sklearn
train_df, test_df = train_test_split(train, test_size=0.2, random_state=42)

# create the datasets
train_dataset = Custom_Dateset(train_df, transform=transforms)
test_dateset = Custom_Dateset(test_df)

# Data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=False)


# create the CNN network
class CNN_Model(nn.Module):
  def __init__(self, in_channels=1, num_classes=num_classes):
    super(CNN_Model, self).__init__()
    self.conv_block = nn.Sequential(
              nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(64), 
              nn.MaxPool2d((2,2), stride=2),

              nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(128), 
              nn.MaxPool2d((2,2), stride=2),

              nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias =False),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(256), 
              nn.MaxPool2d((2,2), stride=2),

              # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
              # nn.ReLU(inplace=True),
              # nn.BatchNorm2d(512), 
              # nn.MaxPool2d((2,2), stride=2),
    )
    
    self.Linear_block = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(256*7*7, 128),
        nn.Linear(128, num_classes)
    )
  def forward(self, x):
    x = self.conv_block(x)
    x = x.reshape(x.shape[0], -1)
    x = self.Linear_block(x)
    return x


model = CNN_Model().to(device)

# the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
  losses = []
  for batch_idx, (data, target) in enumerate(train_loader):
    # get data into gpu
    data = data.to(device=device)
    targets = target.to(device=device)

    # forward
    scores = model(data)
    loss = criterion(scores, targets)
    losses.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# check the accuracy on training and test to see how good the model
def check_accuracy(loader, model):
  if loader == train_loader:
    print('Accuracyof training data')
  else:
    print('Accuracy of test data')

  num_correct = 0
  num_samples = 0

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device = device)
      y = y.to(device = device)

      scores = model(x)
      _, predictions = scores.max(1)  
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

    print(f'We got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples) * 100: .2f}')
  model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)