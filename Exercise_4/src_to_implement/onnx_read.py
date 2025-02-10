import onnxruntime as ort
import numpy as np


# Load model
session = ort.InferenceSession("checkpoint_032.onnx")

# Get input shape and replace dynamic batch size with 1
input_shape = session.get_inputs()[0].shape
input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]

# Generate dummy input
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# Run inference
output = session.run(None, {session.get_inputs()[0].name: dummy_input})

print("Inference Output:", output)