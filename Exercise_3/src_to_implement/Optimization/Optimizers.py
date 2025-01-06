"""" Optimizers """

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

# The following is a very basic optimization algorithm widely used in DL
class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Apply regularizer if present
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate(weight_tensor)
        weight_tensor -= self.learning_rate * gradient_tensor
        return weight_tensor


"""" SGD w/ Momentum and Adam Optimizer """
import numpy as np

#  Advanced optimization schemes are implemented to increase the speed of convergence.
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.regularizer: # Apply regularizer if present
            gradient_tensor += self.regularizer.calculate(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        weight_tensor += self.velocity # Update the weights
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu                        # Beta 1
        self.rho = rho                      # Beta 2
        self.gk = 0                         # Initial gradient
        self.vk = 0                         # biased first moment
        self.rk = 0                         # biased second moment
        self.time_step = 1                  # Time step

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Apply regularizer if present
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate(weight_tensor)

        self.gk = gradient_tensor
        self.vk = self.mu * self.vk + (1 - self.mu) * self.gk
        self.rk = self.rho * self.rk + (1 - self.rho) * np.multiply(self.gk, self.gk)
        vk_hat = self.vk / (1 - self.mu ** self.time_step)
        rk_hat = self.rk / (1 - self.rho ** self.time_step)
        self.time_step += 1

        return weight_tensor - self.learning_rate * (vk_hat / (np.sqrt(rk_hat) + np.finfo(float).eps))