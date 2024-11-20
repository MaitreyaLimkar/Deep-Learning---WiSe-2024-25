""""  Fully Connected (FC) layer """
#  Inherits the base layer

from Exercise_1.src_to_implement.Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):

        # Calling the super-constructor and initializing the inputs uniformly
        super().__init__()
        self.grad_value = None
        self._optimizer = None
        self.weights = np.random.uniform(0, 1, (input_size, output_size))

        self.trainable = True
        self.input_tensor= None
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input_tensor):

        # input_tensor will be used in backward pass
        self.input_tensor = input_tensor

        # Getting the dimensions
        batch_size, input_size = np.shape(input_tensor)

        # Appending additional column
        self.input_tensor = np.column_stack((input_tensor, np.ones(batch_size)))

        # Dot product and output returned
        return input_tensor @ self.weights

    @property
    def optimizer(self):
        """ This is the getter function for optimizer """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_value):
        """ This is the setter function for optimizer """
        self._optimizer = new_value

    def backward(self, error_tensor):
        """ Returns a tensor that serves as the error tensor for the previous layer. """

        # Here the gradient is calculated using the input_tensor and error_tensor
        self.grad_value = self.input_tensor.T @ error_tensor

        # Won't perform an update if the optimizer is not set
        if self._optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # We exclude the bias term and then return the contribution of the layer's weight
        return error_tensor @ self.weights[:-1].T
    
    @property
    def gradient_weights(self):
        """ Returns the gradient with respect to the weights. """
        return self.grad_value