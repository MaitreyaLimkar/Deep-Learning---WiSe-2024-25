""" Pooling Layer """
# Used to reduce the dimensionality of the input and therefore also decrease memory consumption.

#from Exercise_2.src_to_implement.Layers.Base import BaseLayer
from .Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.max_pos = None
        self.input_tensor_shape = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):

        _, _, rows, cols = self.input_tensor_shape = input_tensor.shape # Storing the input tensor shape for backpropagation
        self.max_pos = [] # Resetting max positions list
        pooling_rows, pooling_columns = self.pooling_shape
        stride_rows, stride_columns = self.stride_shape

        # Calculating output dimensions
        outer_rows = ((rows - pooling_rows) // stride_rows) + 1
        outer_columns = ((cols - pooling_columns) // stride_columns) + 1

        output_tensor = np.zeros((*self.input_tensor_shape[:2], outer_rows, outer_columns))

        # Iterating through batches and channels
        for batch_num, cyx_dat in enumerate(output_tensor):
            for channel_num, out_yx_dat in enumerate(cyx_dat):
                in_row = 0
                for out_row in range(outer_rows):
                    in_col = 0
                    for out_col in range(outer_columns):
                        # Extracting the current pooling window
                        in_dat = input_tensor[batch_num, channel_num, in_row: in_row + pooling_rows,
                                 in_col: in_col + pooling_columns]
                        out_yx_dat[out_row, out_col] = np.max(in_dat) # Finding and storing the maximum value in the window

                        # Tracking the position of the maximum value
                        max_loc_row, max_loc_col = np.unravel_index(np.argmax(in_dat), in_dat.shape)
                        self.max_pos.append([batch_num, channel_num, in_row + max_loc_row, in_col + max_loc_col])

                        in_col += stride_columns # Moving the window horizontally
                    in_row += stride_rows # Moving the window vertically

        return output_tensor

    def backward(self, error_tensor):
        # Initializing error output tensor with zeros (same shape as input)
        error_output = np.zeros(self.input_tensor_shape)

        # Distributing the error to the max value positions tracked during forward pass
        for i, max_pos in enumerate(self.max_pos):
            error_output[*max_pos] += error_tensor[np.unravel_index(i, error_tensor.shape)]

        return error_output