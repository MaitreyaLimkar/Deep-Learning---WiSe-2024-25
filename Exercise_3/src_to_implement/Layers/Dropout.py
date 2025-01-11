""" Dropout Regularizer """

import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability  # Probability of keeping a unit active (not dropping)
        self.mask = None  # Will store the dropout mask for backpropagation
        self.testing_phase = False  # Inherited from BaseLayer

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            # Generate binary mask (1 for keep, 0 for drop)
            binary_mask = np.random.binomial(1, self.probability, size=input_tensor.shape)
            self.mask = binary_mask / self.probability # Include scaling factor 1/p in the mask
            return input_tensor * self.mask

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor # During testing, just pass through - no operations needed
        else:
            return error_tensor * self.mask # Apply the same scaled mask from forward pass