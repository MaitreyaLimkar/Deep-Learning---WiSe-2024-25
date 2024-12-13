""" Convolution Layer """

""" Using convolutional layers, problems with Fully connected layer can be circumvented by restricting the layers 
parameters to local receptive fields. """

#from Exercise_2.src_to_implement.Layers.Base import BaseLayer
from .Base import BaseLayer
import numpy as np
from scipy.signal import convolve, correlate
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        # Initialize placeholders for key layer attributes
        self.channels = None                    # Number of input channels
        self.output_tensor = None               # Stores the output of forward pass
        self.batch_size = None                  # Number of samples in a batch
        self.input_tensor = None                # Stores the input during forward pass

        # Layer is trainable (weights can be updated)
        self.trainable = True

        # Convolution parameters stored
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape  # 1D: [c,m], 2D: [c,m,n]
        self.num_kernels = num_kernels

        # Detecting whether this is a 2D convolution based on stride shape
        self.stride_2D = bool(len(self.stride_shape) == 2)

        # Initializing kernels and biases randomly
        # Kernels: random values between 0 and 1, shaped as [num_kernels, channel, spatial_dimensions]
        self.kernels = np.random.rand(self.num_kernels, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)

        # Placeholders for gradient and optimizer values
        self.gradient_weights_val = None            # Stores gradients for kernels
        self.gradient_bias_val = None               # Stores gradients for biases
        self.weight_optimizer_val = None            # Optimizer for kernel weights
        self.bias_optimizer_val = None              # Optimizer for biases

    def forward(self, input_tensor):
        # Storing the input tensor for the backward pass
        self.input_tensor = input_tensor  # 1D: [batch, channels, y], 2D: [batch, channels, y, x]

        self.batch_size, self.channels, *spatial_dimensions = self.input_tensor.shape # Unpack input tensor dimensions

        # This will store convolution results for each batch and kernel
        self.output_tensor = np.zeros((self.batch_size, self.num_kernels, *spatial_dimensions))

        # Iterating through each sample in batch
        for batch_num, image in enumerate(self.input_tensor):
            # Applying each kernel to the image
            for kernel_num, kernel in enumerate(self.kernels):
                # Using cross-correlation to apply kernel
                # mode='same' used to ensure output size matches the input
                # [self.channels // 2] selects the middle channel slice
                self.output_tensor[batch_num, kernel_num] = correlate(image, kernel, mode='same')[self.channels // 2]

                # Adding bias to each kernel's output
                self.output_tensor[batch_num, kernel_num] += self.bias[kernel_num]

        # Applying strides - downsample output based on stride shape
        if self.stride_2D:
            # Down sampling both spatial dimensions
            return self.output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        else:
            # Down sampling only one spatial dimension
            return self.output_tensor[:, :, ::self.stride_shape[0]]

    # noinspection SpellCheckingInspection
    def backward(self, error_tensor):
        # Upsample error tensor to match original output dimensions
        upsampling_error_tensor = np.zeros_like(self.output_tensor)
        if self.stride_2D:
            # For 2D, upsample both spatial dimensions
            upsampling_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        else:
            # For 1D, upsample one spatial dimension
            upsampling_error_tensor[:, :, ::self.stride_shape[0]] = error_tensor

        # Prepare kernels for backpropagation
        # Swap axes and flip left-right to correctly compute gradients
        gradient_kernel = np.swapaxes(self.kernels, 1, 0)
        gradient_kernel = np.fliplr(gradient_kernel)

        # Initializing error tensor for previous layer
        error_tensor_previous = np.zeros_like(self.input_tensor)

        # Computing error for previous layer and distributing error across input channels using convolution
        for arg_num, arg in enumerate(upsampling_error_tensor):
            for kernel_num, kernel in enumerate(gradient_kernel):
                # Convolving the error with flipped kernels to distribute error
                error_tensor_previous[arg_num, kernel_num] = convolve(arg, kernel, mode="same")[self.num_kernels // 2]

        # Padding the input tensor to handle border effects during gradient computation
        if self.stride_2D:
            # 2D padding calculations
            pad_left = (self.convolution_shape[1] - 1) // 2
            pad_right = self.convolution_shape[1] // 2
            pad_top = (self.convolution_shape[2] - 1) // 2
            pad_bottom = self.convolution_shape[2] // 2
            self.input_tensor = np.pad(self.input_tensor,((0, 0), (0, 0),
                                                          (pad_left, pad_right), (pad_top, pad_bottom)))
        else:
            # 1D padding calculations
            pad_left = (self.convolution_shape[1] - 1) // 2
            pad_right = (self.convolution_shape[1]) // 2
            self.input_tensor = (np.pad(self.input_tensor, ((0, 0), (0, 0),
                                                            (pad_left, pad_right))))

        self.gradient_weights_val = np.zeros_like(self.kernels)
        self.gradient_bias_val = np.zeros_like(self.bias)

        # Computing gradients for weights and biases
        for arg_num, error_arg in enumerate(upsampling_error_tensor):
            for error_channel_num, error_channel in enumerate(error_arg):
                for input_channel_ctr in range(self.convolution_shape[0]):
                    # Compute weight gradients by correlating input with error
                    self.gradient_weights_val[error_channel_num, input_channel_ctr] += \
                        correlate(self.input_tensor[arg_num, input_channel_ctr], error_channel, mode='valid')

                self.gradient_bias_val[error_channel_num] += np.sum(error_channel) # Computing bias gradients by summing error

        # Updating weights and biases if optimizers are available
        if self.weight_optimizer_val is not None:
            self.kernels = self.weight_optimizer_val.calculate_update(self.kernels, self.gradient_weights_val)
        if self.bias_optimizer_val is not None:
            self.bias = self.bias_optimizer_val.calculate_update(self.bias, self.gradient_bias_val)

        return error_tensor_previous

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.kernels = weights_initializer.initialize(self.kernels.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self.gradient_weights_val

    @gradient_weights.setter
    def gradient_weights(self, grad_weight_new_val):
        self.gradient_weights_val = grad_weight_new_val

    @property
    def gradient_bias(self):
        return self.gradient_bias_val

    @gradient_bias.setter
    def gradient_bias(self, grad_bias_new_value):
        self.gradient_bias_val = grad_bias_new_value

    @property
    def optimizer(self):
        return self.weight_optimizer_val

    @optimizer.setter
    def optimizer(self, optimizer_val):
        self.weight_optimizer_val = copy.deepcopy(optimizer_val)
        self.bias_optimizer_val = copy.deepcopy(optimizer_val)

    @property
    def weights(self):
        return self.kernels

    @weights.setter
    def weights(self, new_val):
        self.kernels = new_val