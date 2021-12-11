import torch
import torchvision
import numpy as np
import cv2
import onnx
import onnxruntime
import argparse
import warnings
import _init_paths
import models
from config import cfg
from config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Model transformation')
    parser.add_argument('--cfg', type=str, default='demo/mpii_demo_config.yaml')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


args = parse_args()
update_config(cfg, args)


""" 将模型转换为ONNX格式 """
warnings.filterwarnings("ignore")
box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # 人体检测模型
box_model.eval()
box_input = []
image_rgb = cv2.cvtColor(cv2.imread('tools/box_model_input.jpg'), cv2.COLOR_BGR2RGB)  # 输入任意一张真实图像
image_tensor = torch.from_numpy(image_rgb / 255.).permute(2, 0, 1).float()
box_input.append(image_tensor)
# print(box_input[0].shape)
box_out = box_model(box_input)
torch.onnx.export(box_model, box_input, "models/ONNX/fasterrcnn_resnet50_fpn.onnx",
                  input_names=['input'], output_names=['boxes', 'labels', 'scores'], opset_version=11)

pose_model = eval('models.pose_hrnet.get_pose_net')(cfg, is_train=False)  # 姿态估计模型
pose_model.load_state_dict(torch.load('models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth'), strict=False)
pose_model.eval()
pose_input = torch.randn(1, 3, 256, 256, requires_grad=True)
pose_out = pose_model(pose_input)
torch.onnx.export(pose_model, pose_input, "models/ONNX/pose_hrnet_256.onnx",
                  input_names=['box_detection'], output_names=['pose_estimation'], opset_version=11)


""" 验证ONNX图的有效性"""
onnx_box_model = onnx.load("models/ONNX/fasterrcnn_resnet50_fpn.onnx")
onnx.checker.check_model(onnx_box_model)
onnx_pose_model = onnx.load("models/ONNX/pose_hrnet_256.onnx")
onnx.checker.check_model(onnx_pose_model)


""" 验证ONNX Runtime和PyTorch正在为网络计算相同的值 """
ort_box_session = onnxruntime.InferenceSession("models/ONNX/fasterrcnn_resnet50_fpn.onnx")
ort_pose_session = onnxruntime.InferenceSession("models/ONNX/pose_hrnet_256.onnx")
# compute ONNX Runtime output prediction
ort_box_inputs = {ort_box_session.get_inputs()[0].name: box_input[0].detach().cpu().numpy()}
ort_box_outs = ort_box_session.run(None, ort_box_inputs)
ort_pose_inputs = {ort_pose_session.get_inputs()[0].name: pose_input.detach().cpu().numpy()}
ort_pose_outs = ort_pose_session.run(None, ort_pose_inputs)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(box_out[0]['boxes'].detach().cpu().numpy(), ort_box_outs[0], rtol=1e-03, atol=1e-05)
print("Exported box_model has been tested with ONNXRuntime, and the result looks good!")
np.testing.assert_allclose(pose_out.detach().cpu().numpy(), ort_pose_outs[0], rtol=1e-03, atol=1e-05)
print("Exported pose_model has been tested with ONNXRuntime, and the result looks good!")
