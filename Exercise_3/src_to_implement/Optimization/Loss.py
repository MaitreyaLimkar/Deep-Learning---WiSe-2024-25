""" Implementation of Cross Entropy Loss """

import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Using the formula for Cross Entropy loss
        self.prediction_tensor = prediction_tensor + np.finfo(float).eps
        loss_tensor = np.sum(label_tensor * -np.log(self.prediction_tensor))
        return loss_tensor

    def backward(self, label_tensor):
        return -label_tensor / self.prediction_tensor