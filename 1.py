import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from openpose import pyopenpose as op

# 初始化OpenPose
params = {"model_folder": "models/", "hand": False, "face": False}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# 定义人体宽度的估计值（假设为1米）
average_person_width = 0.5  # 米
distance_threshold = 2 * average_person_width  # 欧式距离阈值

def detect_people(frame):
    """
    使用YOLOv5检测视频帧中的人
    """
    results = model(frame)
    detections = results.xyxy[0].numpy()  # 获取检测结果
    people = []
    for det in detections:
        if det[5] == 0:  # 假设类别0为人
            x_min, y_min, x_max, y_max = det[:4]
            people.append([x_min, y_min, x_max, y_max])
    return people

def calculate_euclidean_distance(center1, center2):
    """
    计算两个点之间的欧式距离
    """
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def estimate_pose(frame, people):
    """
    使用OpenPose估计姿态
    """
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints

def analyze_obstruction(central_pose, neighbor_pose):
    """
    分析两个姿态是否存在相互阻碍
    """
    head1 = central_pose[0]
    head2 = neighbor_pose[0]
    head_distance = calculate_euclidean_distance(head1, head2)
    if head_distance < 0.5:  # 假设头部距离小于0.5米时存在阻碍
        return True
    return False

def find_central_person(people):
    """
    找到与周围目标人的欧氏距离之和最小的目标框作为中心点
    """
    if not people:
        return None
    min_total_distance = float('inf')
    central_person = None

    for i, person1 in enumerate(people):
        person1_center = [(person1[0] + person1[2]) / 2, (person1[1] + person1[3]) / 2]
        total_distance = 0
        for j, person2 in enumerate(people):
            if i == j:
                continue
            person2_center = [(person2[0] + person2[2]) / 2, (person2[1] + person2[3]) / 2]