import torch
import onnx
from onnx import shape_inference

# Load the ONNX model
model = onnx.load("yolov4.onnx")

# Infer the shapes of the model's inputs and outputs
model = shape_inference.infer_shapes(model)

# Get the input shape
input_shape = model.graph.input[0].type.tensor_type.shape
input_shape.dim[0].dim_value = -1

onnx.save(model, 'yolov4_dynamic.onnx')