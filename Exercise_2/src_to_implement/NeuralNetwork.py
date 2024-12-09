""" The Neural Network de nes the whole architecture by containing all its layers from the input
 to the loss layer. This Network manages the testing and the training, that means it calls all
 forward methods passing the data from the beginning to the end, as well as the optimization
 by calling all backward passes afterward. """

import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer                      # Optimizer object passed during initialization.
        self.loss = []                                  # List to store loss values for each training iteration.
        self.layers = []                                # List to hold the architecture i.e. the layers of the network.
        self.data_layer = None                          # Reference to the data layer providing input data and labels.
        self.loss_layer = None                          # Reference to the loss layer, our case - Cross Entropy Loss
        self.label_tensor = None

    def forward(self):

        # We will fetch input_tensor and label_tensor from data_layer
        input_tensor, self.label_tensor = self.data_layer.next()

        # We pass the input through each layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # We shall pass through the loss layer for Cross Entropy loss
        output = self.loss_layer.forward(input_tensor, self.label_tensor)
        return output

    def backward(self):

        # We now start the backprop from the loss
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):

        # Here a deep copy is created for the Optimizer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, num_epochs):

        for n in range(num_epochs):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor