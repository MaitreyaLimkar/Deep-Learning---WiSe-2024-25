""" TanH Activation function """

from .Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1 - np.square(self.activation))