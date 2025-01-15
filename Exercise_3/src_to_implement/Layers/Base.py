""" Implementation of Base Layer """

class BaseLayer:
    def __init__(self):
        self.trainable = False # This member will be used to distinguish trainable from non-trainable layers.
        # self.weights = [] - Commented out due to error in RNN test
        self.testing_phase = False