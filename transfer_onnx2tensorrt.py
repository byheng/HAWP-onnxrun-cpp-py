# Description: This script is used to convert the onnx model to fit for tensorRT.

import onnx

model = onnx.load('hawp_512x512_float32.onnx')
inferred_model = onnx.shape_inference.infer_shapes(model)
onnx.save(inferred_model, 'hawp_512x512_float32_inferred.onnx')