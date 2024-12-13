""" Flatten Layer """
# This layer reshapes the multidimensional input to a one dimensional feature vector

#from Exercise_2.src_to_implement.Layers.Base import BaseLayer
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return input_tensor.reshape(self.shape[0], -1)
        
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)