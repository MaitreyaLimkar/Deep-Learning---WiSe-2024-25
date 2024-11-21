""" SoftMax Activation Function """

import numpy as np
from Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.resultant_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # We will use the SoftMax Function for the Forward pass
        # For numerical stability, we shift xk
        exp_xk = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        resultant_tensor = exp_xk / np.sum(exp_xk, axis=1, keepdims=True)
        return resultant_tensor

    def backward(self, error_tensor):

        product_tensor = error_tensor * self.resultant_tensor
        sum_tensor = np.sum(product_tensor, axis=1, keepdims=True)
        return self.resultant_tensor * (error_tensor - np.sum(sum_tensor))