# import onnx
# from onnxoptimizer import optimize

# print("Loading model...")
# model = onnx.load('yolov4.onnx')
# print("Loaded model.")

# print("Optimizing model...")
# optimized_model = optimize(model, ['fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'eliminate_identity', 'eliminate_nop_transpose'])
# print("Optimized model.")

# print("Saving optimized model...")
# onnx.save(optimized_model, 'yolov4_optimized.onnx')
# print("Saved optimized model.")

import onnx
import onnx_graphsurgeon as gs

onnx_model = onnx.load('yolov4_optimized.onnx')

graph = gs.import_onnx(onnx_model)
graph.fold_constants()
graph.cleanup()

onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, 'yolov4_optimized_folded.onnx')
