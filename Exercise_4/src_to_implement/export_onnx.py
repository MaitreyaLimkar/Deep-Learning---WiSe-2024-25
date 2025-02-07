import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import ResNet

epoch = int(sys.argv[1])
# Model initiated
model = ResNet()

# Define the loss function
crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
