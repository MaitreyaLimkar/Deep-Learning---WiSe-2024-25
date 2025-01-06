""" SoftMax Activation Function """

import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.resultant_tensor = None

    def forward(self, input_tensor):

        # We will use the SoftMax Function for the Forward pass
        # For numerical stability, we shift xk
        self.resultant_tensor = np.exp(input_tensor - np.max(input_tensor))
        self.resultant_tensor = self.resultant_tensor / np.sum(self.resultant_tensor, axis=1, keepdims=True)
        return self.resultant_tensor

    def backward(self, error_tensor):

        sum_tensor = np.sum(error_tensor * self.resultant_tensor, axis=1, keepdims=True)
        return self.resultant_tensor * (error_tensor - sum_tensor)