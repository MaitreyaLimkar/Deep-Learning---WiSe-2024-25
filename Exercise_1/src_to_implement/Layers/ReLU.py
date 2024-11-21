""" Rectified Linear Unit class """
from Exercise_1.src_to_implement.Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        # ReLU is a non-trainable layer, so we don't modify the `trainable` attribute.
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # We use the ReLU activation for the output f(x) = max(0, x)
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):

        # Using the following method, the gradient is 1 where input_tensor > 0, or 0 otherwise
        relu_grad = np.where(self.input_tensor > 0, 1, 0)
        return error_tensor @ relu_grad