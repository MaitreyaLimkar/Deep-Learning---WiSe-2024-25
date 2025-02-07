import torch as t
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Load the data from the csv file and perform a train-test split
csv_file = "data.csv"  # Adjust path if necessary
data = pd.read_csv(csv_file, sep=';')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Set up data loading for the training and validation set using DataLoader and ChallengeDataset objects
train_ds = ChallengeDataset(train_data, mode='train')
val_ds = ChallengeDataset(val_data, mode='val')
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

# Create an instance of our ResNet model
resnet_model = model.ResNet()

# Set up a suitable loss criterion for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# Set up the optimizer
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

# Create an object of type Trainer and set its early stopping criterion
trainer = Trainer(resnet_model, crit=criterion, optim=optimizer, train_dl=train_dl, val_test_dl=val_dl, cuda=True, early_stopping_patience=5)

# Start training
res = trainer.fit(epochs=50)

# Plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
