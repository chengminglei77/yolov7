import atexit
import json
import os
from flask import Flask, request
import cv2
import torch
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from numpy import random
import time
import paho.mqtt.client as mqtt
import uuid

import threading

from Report import HttpCode, Error, Result
from detect_img_streams import recognize_head
from models.experimental import attempt_load
from utils.cut_images.cut_polygon_imgs import image_to_base64
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.recongnized_color import identify_color
from utils.torch_utils import select_device, TracedModel

app = Flask(__name__)
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_AS_ASCII'] = False

# 模型名称
_video_model = 'video'
_cap_model = 'safetyCap'
_uniform_model = 'uniform'
# 模型路径
_weights = {
    _video_model: "weights/video.pt",
    _cap_model: "weights/cap10.pt",
    _uniform_model: "weights/uniform1.pt"
}
# 初始化模型
_models = {

}
# 默认设备
_device = 'cpu'
# 标签名映射
change_txt = {
    "person": "person",
    "person_head": "personHead",
    "alarm_fumes": "alarmFumes",
    "alarm_fire": "alarmFire",
    "behavior_smoking": "behaviorSmoking",
    "behavior_call": "behaviorCall",
    "behavior_look_phone": "behaviorLookPhone",
    "safety_cap": "cap",
    "upper_body": "upperBody",
    "lower_body": "lowerBody"
}


# 加载模型
def init_model():
    # Initialize
    device = select_device(_device)
    for item in _weights:
        # Load model
        _models[item] = attempt_load(_weights[item], map_location=device)  # load FP32 model
        # 优化模型
        _models[item] = TracedModel(_models[item], device, 640)  # load FP32 model


def detect_img(img_path, imgSize=640, labelName=[], _device='cpu', _models={},
               _trace=True, _conf_thres=0.25, _iou_thres=0.45, _agnostic_nms=False):
    try:
        # Initialize
        img = cv2.imread(img_path)
        im0 = img.copy()
        device = select_device(_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        result = {}
        images = {}
        # Load model
        t1 = time.time()
        for item in _models:
            model = _models[item]
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgSize, s=stride)  # check img_size

            if half:
                model.half()  # to FP16
            """
                    数据源
            """
            # padded resize
            if item in images:
                imgOs = images[item]
            else:
                imgOs = im0
            img = letterbox(imgOs.copy(), imgsz, stride)[0]
            # cv2.imwrite(f'{item}.jpg', imgOs)

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
            if item != _video_model:
                flag = True
            for i, det in enumerate(pred):  # detections per image
                s = ''
                if len(det):
                    # Rescale boxes from img_size to im0 sizes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgOs.shape).round()
                    result['success'] = True
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # 过滤结果
                        if names[int(c)] not in ['safety_cap', 'no_cap', 'upper_body', 'lower_body']:
                            result[change_txt[names[int(c)]]] = int(n)
                        if item == _video_model and (names[int(c)] == 'person' or names[int(c)] == 'person_head'):
                            flag = True
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 坐标
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度
                        label = f'{names[int(cls)]} {conf:.2f}'
                        print(xyxy)
                        if item == _video_model and label_name == 'person':
                            # 修改图片
                            images[_cap_model] = recognize_head(imgOs,
                                                                labelName="person_head", model=_models[_video_model])
                            images[_uniform_model] = imgOs[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                        elif label_name in ['upper_body', 'lower_body', 'safety_cap']:
                            txt = identify_color.get_color(imgOs[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
                            if label_name == 'safety_cap':
                                result[change_txt[label_name]] = {
                                    'isHelmet': True,
                                }
                            else:
                                result[change_txt[label_name]] = {
                                    'type': 'other'
                                }
                            result[change_txt[label_name]].update(txt)

                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],line_thickness=int(im0.shape[1] / 850))
                else:
                    if item == _cap_model:
                        result['cap'] = {
                            'isHelmet': False,
                        }
                    elif item == _video_model:
                        result['success'] = False
                        flag = False
            # if item == 'video':
            #    result['img'] = image_to_base64(im0)

            if result == {}:
                flag = False
                result = {
                    'success': False
                }

            if item == _cap_model and 'cap' not in result:
                result['cap'] = {
                    'isHelmet': False,
                }
            if not flag:
                break
            print(f'耗时：{time.time() - t1}')
            t1 = time.time()
        del_images(img_path)
        return result
    except Exception as e:
        print(e)
        del_images(img_path)
        return result


@app.route('/WB_AI/petrochemical/report', methods=['POST'])
def petrochemical_predict():
    start = time.time()
    if request.method == 'POST':
        # 获取上传的文件,若果没有文件默认为None
        files = request.files.getlist('images', None)
        value = request.values.get('alarmType')
        results = []
        for file in files:
            if file is None:
                return Error(HttpCode.servererror, 'no files for upload')
            img_name = f"./{str(uuid.uuid4())}.jpg"
            file.save(img_name)
            result = detect_img(img_name, _models=_models)
            results.append(result)
        end_time = time.time()
        return Result(HttpCode.ok, "预测成功", cost=round(float(end_time - start), 3), data=results)
    else:
        return Result(HttpCode.servererror, '请求方式错误,请使用post方式上传')


#
# # 连接MQTT服务端
# client = mqtt.Client()
#
#
# # 连接MQTT
# def con_mqtt():
#     try:
#         global _config_data
#         with open("detect-core.config", "r") as file:
#             config_data = json.load(file)
#             _config_data = config_data
#             config_data = config_data['mqtt']
#
#         client.keep_alive = 60
#
#         client.username_pw_set(config_data["mqtt_user"], config_data["mqtt_password"])
#         client.connect(config_data["mqtt_server"], config_data["mqtt_port"], 60)
#         client.on_connect = on_connect
#         client.on_message = on_message
#         client.loop_start()  # 保持连接
#     except Exception as e:
#         print(e)
#     while True:
#         time.sleep(60)
#
#
# # 连接回调
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code " + str(rc))
#     client.subscribe("/petrochemical/Service/Command")
#
#
def del_images(path):
    if os.path.exists(path):
        os.remove(path)


# # 获取消息回调
# def on_message(client, userdata, msg):
#     if msg.topic == '/petrochemical/Service/Command':
#         img_name = f"./{str(uuid.uuid4())}.jpg"
#         # 处理消息是否为文件流数据
#         with open(img_name, "ab") as f:
#             f.write(msg.payload)
#         detect_img(img_name, _models=_models)
#
#
# # 发送消息对象data
# def send_message(data):
#     if client.is_connected():
#         msg = json.dumps(data)
#         client.publish("/petrochemical/Service/Data", msg, qos=0, retain=False)

#
# def mqtt():
#     init_model()
#     thread = threading.Thread(target=con_mqtt)
#     thread.start()
#
#     # 防止主线程退出
#     while True:
#         time.sleep(1)

if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=9797)
