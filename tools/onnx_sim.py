import onnx
from onnxsim import simplify

# 简化ONNX模型，解决冗余op问题
output_path = "models/ONNX/fasterrcnn_resnet50_fpn.onnx"
onnx_model = onnx.load(output_path)  # load onnx model
model_sim, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
output_path = "models/ONNX/fasterrcnn_resnet50_fpn_sim.onnx"
onnx.save(model_sim, output_path)
print('finished exporting onnx')
