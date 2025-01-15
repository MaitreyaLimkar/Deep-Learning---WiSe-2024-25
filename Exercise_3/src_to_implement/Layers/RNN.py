""" Recurrent Neural Network Layer """

import numpy as np
import copy
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Using parameters to control training and state memory
        self.trainable = True
        self.memorize_value = False
        self.optimizer_value = None

        # Defining core dimensions
        self.input_size = input_size  # Accepting vectors of this size
        self.hidden_size = hidden_size  # Maintaining states of this size
        self.output_size = output_size  # Producing outputs of this size

        # Initializing hidden state with zeros
        self.hidden_state = [np.zeros(self.hidden_size)]

        # Creating component layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh_layer = TanH()
        self.sig_layer = Sigmoid()
        self.fc_out = FullyConnected(hidden_size, output_size)

    def initialize(self, weights_initializer, bias_initializer):

        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_out.initialize(weights_initializer, bias_initializer)
        self.weights = self.fc_hidden.weights
        self.weights_hy = self.fc_out.weights

    def forward(self, input_tensor):
        # Resetting hidden state if not memorizing
        if self.memorize == False:
            self.hidden_state = [np.zeros((1, self.hidden_size))]

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.output = np.zeros((self.batch_size, self.output_size))

        # Processing each time step in sequence
        for time, temp_tensor in enumerate(input_tensor):
            # Reshaping input to handle batching
            x_t = temp_tensor.reshape(1, -1)
            h_t1 = self.hidden_state[-1].reshape(1, -1)

            # Combining input with previous hidden state
            x_tilde = np.hstack([x_t, h_t1])

            # Processing through layers
            u_t = self.fc_hidden.forward(x_tilde)
            self.hidden_state.append(self.tanh_layer.forward(u_t))
            o = self.fc_out.forward(self.hidden_state[-1])
            self.output[time] = self.sig_layer.forward(o)
        return self.output

    def backward(self, error_tensor):
        # Initializing gradients
        self.gradient_weights_value = np.zeros_like(self.fc_hidden.weights)
        self.gradient_weights_hy_value = np.zeros_like(self.fc_out.weights)
        output_error = np.zeros((self.batch_size, self.input_size))
        error_h = np.zeros((1, self.hidden_size))

        # Performing backpropagation through time
        for reverse_time in reversed(range(error_tensor.shape[0])):
            # Reconstructing forward pass
            x_t = self.input_tensor[reverse_time, :].reshape(1, -1)
            h_t1 = self.hidden_state[reverse_time].reshape(1, -1)
            x_tilde = np.hstack([x_t, h_t1])

            # Setting activations
            self.sig_layer.forward(self.fc_out.forward(
                self.tanh_layer.forward(self.fc_hidden.forward(x_tilde))))

            # Propagating gradients backward
            grad = self.sig_layer.backward(error_tensor[reverse_time, :])
            grad = self.fc_out.backward(grad) + error_h
            self.gradient_weights_hy_value += self.fc_out.gradient_weights
            grad = self.tanh_layer.backward(grad)
            grad = self.fc_hidden.backward(grad)
            self.gradient_weights_value += self.fc_hidden.gradient_weights

            # Splitting gradients
            output_error[reverse_time, :] = grad[:, :self.input_size]
            error_h = grad[:, self.input_size:]

        # Updating weights if optimizer exists
        if self.optimizer:
            self.fc_hidden.weights = self.optimizer.calculate_update(
                self.fc_hidden.weights, self.gradient_weights_value)
            self.fc_out.weights = self.optimizer.calculate_update(
                self.fc_out.weights, self.gradient_weights_hy_value)

        self.weights = self.fc_hidden.weights
        self.weights_hy = self.fc_out.weights

        return output_error

    @property
    def memorize(self):
        return self.memorize_value

    @memorize.setter
    def memorize(self, memorize):
        self.memorize_value = memorize

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_value

    @gradient_weights.setter
    def gradient_weights(self, new_weights):
        self.fc_hidden._gradient_weights = new_weights

    @property
    def optimizer(self):
        return self.optimizer_value

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_value = copy.deepcopy(optimizer)