from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import csv
import os
import numpy as np
import time

import lib.models
from lib.config import cfg
from lib.config import update_config
from lib.core.function import get_final_preds
from lib.utils.transforms import get_affine_transform

COCO_KEYPOINT_INDEXES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder',
    6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

SKELETON = [[1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8],
            [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

Colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def draw_pose(keypoints, img):
    """ 绘制关键点和骨架 """
    assert keypoints.shape == (NUM_KPTS, 2)  # the shape should be equal to [17,2]
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Colors[i], 4)
        cv2.circle(img, (int(x_a), int(y_a)), 6, Colors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, Colors[i], -1)


def draw_bbox(box, img):
    """ 绘制检测包围框 """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)  # green


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = list(pred[0]['labels'].cpu().numpy())
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        # 1 -> person (instance_category_names)
        if pred_classes[idx] == 1:
            person_boxes.append(box)
    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(image, trans, (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
                                 flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray([center]), np.asarray([scale]))
        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """ 将框转换为姿态变换所需的中心、尺度信息 """
    center = np.zeros(2, dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array([box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    """ 解析参数 """
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', type=str, default='coco_demo_config.yaml')
    parser.add_argument('--webcam', type=bool, default=False)
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--image', type=bool, default=True)

    parser.add_argument('--writeBoxFrames', type=bool, default=False)
    parser.add_argument('--saveFile', type=bool, default=True)
    parser.add_argument('--showFps', type=bool, default=True)
    parser.add_argument('--outputDir', type=str, default='../output/coco_demo')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # 检测模型
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('lib.models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)  # 姿态估计模型

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # 加载摄像头或视频或图像
    if args.webcam:
        vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        estimate_webcam_or_video(args, box_model, pose_model, vidcap)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
        estimate_webcam_or_video(args, box_model, pose_model, vidcap)
    elif args.image:
        csv_output_rows = []
        for filename in os.listdir("./image/"):  # 注意图像存储路径
            if filename[-4:] == '.jpg':
                new_csv_row = ['image_{}'.format(filename[:-4])]
                new_csv_row.extend(estimate_image(args, box_model, pose_model, filename))
                csv_output_rows.append(new_csv_row)
        # 将关键点坐标写入csv表格
        csv_output_filename = os.path.join(args.outputDir, 'coco_pose_coord_data.csv')
        csv_headers = ['image_name']
        for i, keypoint in COCO_KEYPOINT_INDEXES.items():
            csv_headers.extend([str(i) + ': ' + keypoint + '_x', str(i) + ': ' + keypoint + '_y'])
        with open(csv_output_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_headers)
            csvwriter.writerows(csv_output_rows)
    else:
        print('please use --video or --webcam or --image to define the input.')
        return


def estimate_webcam_or_video(args, box_model, pose_model, vidcap):
    """ estimate on the webcam or video """
    if args.saveFile:
        if args.webcam:
            time_str = time.strftime('%m-%d-%H-%M')
            save_path = '{}/coco_webcam_{}.avi'.format(args.outputDir, time_str)
        else:
            save_path = '{}/coco_video_{}.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.video))[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, 24.0, (int(vidcap.get(3)), int(vidcap.get(4))))
    while True:
        ret, image_bgr = vidcap.read()
        if ret:
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
            input.append(img_tensor)

            # object detection box
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
            if args.writeBoxFrames:
                for box in pred_boxes:
                    draw_bbox(box, image_bgr)

            # pose estimation
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                    if len(pose_preds) >= 1:
                        for kpt in pose_preds:
                            draw_pose(kpt, image_bgr)  # draw the poses

            if args.showFps:
                fps = 1 / (time.time() - last_time)
                image_bgr = cv2.putText(image_bgr, 'FPS: ' + "%.2f" % fps, (25, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            if args.saveFile:
                out.write(image_bgr)

            cv2.imshow('coco_demo', image_bgr)
            if cv2.waitKey(1) & 0XFF == ord('q'):  # 按 "q" 退出
                break
        else:
            print('cannot load the video.')
            break

    cv2.destroyAllWindows()
    vidcap.release()
    if args.saveFile:
        print('video has been saved as {}'.format(save_path))
        out.release()


def estimate_image(args, box_model, pose_model, filename):
    """ estimate on the image """
    image_bgr = cv2.imread('./image/' + filename)
    image = image_bgr[:, :, [2, 1, 0]]

    input = []
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
    input.append(img_tensor)

    # object detection box
    pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
    if args.writeBoxFrames:
        for box in pred_boxes:
            draw_bbox(box, image_bgr)

    # pose estimation
    new_csv_row = []
    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            if len(pose_preds) >= 1:
                for kpt in pose_preds:
                    draw_pose(kpt, image_bgr)  # draw the poses
                    for coord in kpt:
                        new_csv_row.extend([coord[0], coord[1]])

    if args.saveFile:
        save_path = '{}/coco_image_{}.jpg'.format(args.outputDir, os.path.splitext(os.path.basename(filename))[0])
        cv2.imwrite(save_path, image_bgr)
        print('The result image has been saved as {}'.format(save_path))

    # 新建窗口显示
    # cv2.imshow('coco_demo', image_bgr)
    # cv2.waitKey(1000)  # 等待 1s
    # cv2.destroyAllWindows()
    return new_csv_row


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
