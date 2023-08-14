import atexit
import json
import os

import cv2
import torch
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from numpy import random
import time
import paho.mqtt.client as mqtt
import uuid

import threading
from detect_img_streams import recognize_head
from models.experimental import attempt_load
from utils.cut_images.cut_polygon_imgs import image_to_base64
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.recongnized_color import identify_color
from utils.torch_utils import select_device, TracedModel

_device = 'cpu'
_weights = {
    "vedio": "weights/video.pt",
    "safetyCap": "weights/safety_best8.pt"
}
_models = {

}
_status = False
_thread_detect = None


# 加载模型
def init_model():
    # Initialize
    device = select_device(_device)
    for item in _weights:
        # Load model
        _models[item] = attempt_load(_weights[item], map_location=device)  # load FP32 model


def detect_img(img_path, imgSize=640, labelName=[], _device='cpu', _models={},
               _trace=True, _conf_thres=0.45, _iou_thres=0.45, _agnostic_nms=False):
    try:
        # Initialize
        img = cv2.imread(img_path)
        im0 = img
        device = select_device(_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        result = {}
        # Load model
        t1 = time.time()
        img_temp = None
        for item in _models:
            model = _models[item]
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgSize, s=stride)  # check img_size
            if _trace:
                model = TracedModel(model, device, imgsz)

            if half:
                model.half()  # to FP16
            """
                    数据源
            """
            # padded resize
            img = letterbox(im0, imgsz, stride)[0]
            # convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1
            """
                    原本循环图片列表|视频流 修改为读取图片流
                """
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, _conf_thres, _iou_thres, classes=None, agnostic=_agnostic_nms)
            # Process detections
            flag = False
            if item != 'video':
                flag = True
            for i, det in enumerate(pred):  # detections per image
                s = ''
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        if names[int(c)] not in ['safety_cap']:
                            result[names[int(c)]] = int(n)
                        if item == 'vedio' and (names[int(c)] == 'person' or names[int(c)] == 'person_head'):
                            flag = True

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 坐标
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if label_name == 'person':
                            # 修改图片
                            img_temp = recognize_head(im0,
                                                      labelName="person_head", model=_models['vedio'])

                        elif label_name == 'safety_cap':
                            txt = identify_color.get_color(im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
                            result['cap'] = {
                                'type': label_name,
                            }
                            result['cap'].update(txt)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                     line_thickness=int(im0.shape[1] / 850))
                else:
                    flag = False
            if item == 'vedio':
                # result['img'] = image_to_base64(im0)
                if not (img_temp is None):
                    im0 = img_temp
            if not flag:
                break
            print(f'耗时：{time.time() - t1}')
            t1 = time.time()
        del_images(img_path)
        send_message(result)
    except Exception as e:
        print(e)


# 连接MQTT服务端
client = mqtt.Client()


# 连接MQTT
def con_mqtt():
    try:
        global _config_data
        with open("detect-core.config", "r") as file:
            config_data = json.load(file)
            _config_data = config_data
            config_data = config_data['mqtt']

        client.keep_alive = 60

        client.username_pw_set(config_data["mqtt_user"], config_data["mqtt_password"])
        client.connect(config_data["mqtt_server"], config_data["mqtt_port"], 60)
        client.on_connect = on_connect
        client.on_message = on_message
        client.loop_start()  # 保持连接
    except Exception as e:
        print(e)
    while True:
        time.sleep(60)


# 连接回调
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("/petrochemical/Service/Command")


def del_images(path):
    if os.path.exists(path):
        os.remove(path)


# 获取消息回调
def on_message(client, userdata, msg):
    if msg.topic == '/petrochemical/Service/Command':
        img_name = f"./{str(uuid.uuid4())}.jpg"
        # 处理消息是否为文件流数据
        with open(img_name, "ab") as f:
            f.write(msg.payload)
        detect_img(img_name, _models=_models)


# 发送消息对象data
def send_message(data):
    if client.is_connected():
        msg = json.dumps(data)
        client.publish("/petrochemical/Service/Data", msg, qos=0, retain=False)


if __name__ == '__main__':
    init_model()
    thread = threading.Thread(target=con_mqtt)
    thread.start()

    # 防止主线程退出
    while True:
        time.sleep(1)
