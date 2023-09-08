import json
from concurrent.futures import ThreadPoolExecutor

import os
from flask import Flask, request
import cv2
import torch
import numpy as np
from numpy import random
import time
import uuid

from Report import HttpCode, Error, Result
from detect_img_streams import recognize_head
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.recongnized_color import identify_color
from utils.torch_utils import select_device, TracedModel

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_AS_ASCII'] = False

# 配置路径
_config_path = "detect_img.config"
# 模型名称
_video_model = 'video'
_cap_model = 'cap'
_uniform_model = 'uniform'
_reflective_model = 'reflective'
# 模型路径
_weights = {
    _video_model: "weights/video.pt",
    _cap_model: "weights/cap.pt",
    _uniform_model: "weights/uniform1.pt",
    _reflective_model: "weights/reflective1.pt"

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
    "lower_body": "lowerBody",
    "e_pillar": "ePillar",
    "e_lights": "eLights",
    "e_gas_tank": "eGasTank"
}
# 模型和标签配置
_config = {}
_debug_img_path = './output/debug/'
# 设置线程池
pool = ThreadPoolExecutor(max_workers=2)


# 加载配置文件
def load_config():
    global _config
    try:
        with open(_config_path, "r") as file:
            _config = json.load(file)
    except Exception as e:
        print(e)
        _config['modelConfig'] = []
        _config['labelConfig'] = []


# 根据模型名称获取获取模型参数（device,imageSize）
def parse_model_config(name=_video_model):
    global _config
    modelConfig = _config['modelConfig']

    if modelConfig is None:
        return select_device(_device), 480

    model = [element for element in modelConfig if isinstance(element, dict) and element['name'] == name]
    if len(model) < 1:
        return select_device(_device), 480
    return select_device(model[0]['device']), model[0]['imageSize']


# 根据标签名称解析标签置信度
def parse_label_conf_value(labelName='person'):
    labelConfig = _config['labelConfig']
    if labelConfig is None:
        return 0.25
    labels = [element for element in labelConfig if isinstance(element, dict) and element['name'] == labelName]
    if len(labels) < 1:
        return 0.25
    return labels[0]['confValue']


# 根据模型获取基础置信度
def parase_model_conf_value(modelName='video'):
    global _config
    modelConfig = _config['modelConfig']

    if modelConfig is None:
        return 0.25

    model = [element for element in modelConfig if isinstance(element, dict) and element['name'] == modelName]
    if len(model) < 1:
        return 0.25
    return model[0]['confValue']


# 加载模型
def init_model():
    # 加载模型参数
    load_config()
    # Initialize
    for item in _weights:
        d1, size = parse_model_config(item)
        # Load model
        _models[item] = attempt_load(_weights[item], map_location=d1)  # load FP32 model
        # 优化模型
        _models[item] = TracedModel(_models[item], d1, size)  # load FP32 model


# 预处理图片
def pre_parse_img(model, img, stride, imgsz, device, half):
    img = letterbox(img, imgsz, stride)[0]
    # convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    if device.type != _device:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    if device.type != _device and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=False)[0]
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    return pred, img


def detect_cloth_type(img, _conf_thres=0.25, _iou_thres=0.45, _agnostic_nms=False, is_debug=False):
    global _models
    try:
        # Initialize
        device, imgSize = parse_model_config(_reflective_model)
        half = device.type != _device  # half precision only supported on CUDA
        # Load model
        model = _models[_reflective_model]

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        t1 = time.time()
        img0s = img.copy()
        # 图片添加白框
        img0s = cv2.copyMakeBorder(img0s, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        if is_debug:
            cv2.imwrite(f"{_debug_img_path}uniform-{uuid.uuid4()}.jpg", img0s)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        pred, img = pre_parse_img(model=model, img=img0s, stride=stride, imgsz=imgsz, device=device,
                                  half=half)

        # Apply NMS
        pred = non_max_suppression(pred, _conf_thres, _iou_thres, classes=None, agnostic=_agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                return 'reflective'
        return 'other'

    except Exception as e:
        print(e)
        return 'other'


# 识别模型uniform
def detect_uniform(images, _conf_thres=0.25, _iou_thres=0.45,
                   _agnostic_nms=False, is_debug=False):
    global _models
    try:
        # Initialize
        device, imgSize = parse_model_config(_uniform_model)
        half = device.type != _device  # half precision only supported on CUDA
        # Load model
        model = _models[_uniform_model]

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        t1 = time.time()
        uniform_image = {
            'upper_body': [],
            'lower_body': []
        }
        for item in images:
            # 解析图片
            xyxy = item['points']
            img0s = item['image'][int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

            # 给图片填充白框
            img0s = cv2.copyMakeBorder(img0s, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            if is_debug:
                cv2.imwrite(f"{_debug_img_path}uniform-{uuid.uuid4()}.jpg", img0s)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            pred, img = pre_parse_img(model=model, img=img0s.copy(), stride=stride, imgsz=imgsz, device=device,
                                      half=half)

            # Apply NMS
            pred = non_max_suppression(pred, _conf_thres, _iou_thres, classes=None, agnostic=_agnostic_nms)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 sizes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度
                        if conf_val >= parse_label_conf_value(labelName=label_name):
                            uniform_image[label_name].append({
                                "image": img0s[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])],
                                'value': conf_val * (int(xyxy[3]) - int(xyxy[1])) * (int(xyxy[2]) - int(xyxy[0]))
                            })
        result = {}
        for item in uniform_image:
            if len(uniform_image[item]) > 0:
                sorted(uniform_image[item], key=lambda k: (k.get('value', 0)))
                result[change_txt[item]] = {
                    'type': 'other',
                }
                txt = identify_color.get_color(uniform_image[item][-1]['image'])
                result[change_txt[item]].update(txt)
                if item == 'upper_body':
                    type = detect_cloth_type(uniform_image[item][-1]['image'])
                    result[change_txt[item]]['type'] = type
        return result
    except Exception as e:
        print(e)
        return {}


# 识别模型cap
def detect_cap(images, _conf_thres=0.25, _iou_thres=0.45, _agnostic_nms=False, is_debug=False):
    global _models
    try:
        # Initialize
        device, imgSize = parse_model_config(_cap_model)
        half = device.type != _device  # half precision only supported on CUDA
        # Load model
        model = _models[_cap_model]

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        t1 = time.time()
        header_imgs = []
        heads_temp = []
        for item in images:
            # 解析图片
            xyxy = item['points']
            img0 = item['image'][int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            # 获取人头图片
            flag, imgs = recognize_head(img0, labelName="person_head",
                                        _conf_thres=parse_label_conf_value(labelName='person_head'),
                                        model=_models[_video_model])
            if not flag:
                return {'isHelmet': False}
            heads_temp.extend(imgs)

        for item in heads_temp:
            # 给图片填充白框
            img0s = cv2.copyMakeBorder(item, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            if is_debug:
                cv2.imwrite(f"{_debug_img_path}cap-{uuid.uuid4()}.jpg", img0s)
            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            pred, img = pre_parse_img(model=model, img=img0s.copy(), stride=stride, imgsz=imgsz, device=device,
                                      half=half)
            # Apply NMS
            pred = non_max_suppression(pred, parase_model_conf_value(_cap_model), _iou_thres, classes=None,
                                       agnostic=_agnostic_nms)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 sizes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度
                        if label_name == 'safety_cap' and conf_val >= parse_label_conf_value(labelName=label_name):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # plot_one_box(xyxy, img0s, label=label, color=colors[int(cls)], line_thickness=1)
                            header_imgs.append({
                                "image": img0s[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])],
                                'value': conf_val * (int(xyxy[3]) - int(xyxy[1])) * (int(xyxy[2]) - int(xyxy[0]))
                            })

        if len(header_imgs) < 1 or len(header_imgs) != len(heads_temp):
            return {'isHelmet': False}
        # 对所有头盔的图片进行排序，取最大的一个返回颜色
        sorted(header_imgs, key=lambda k: (k.get('value', 0)))
        result = {
            'isHelmet': True,
        }
        img = header_imgs[-1]['image']
        result.update(identify_color.get_color(img))
        return result
    except Exception as e:
        print(e)
        return {'isHelmet': False}


# 识别模型video
def detect_img(img_path, _conf_thres=0.25, is_padding=False, _iou_thres=0.45,
               _agnostic_nms=False, is_cut=False, is_debug=False):
    global _models
    try:
        # Initialize
        img = cv2.imread(img_path)
        # 是否截取图片
        if is_cut:
            img = identify_color.cut_rect_image(img_path)
        # 是否对图片添加白框
        if is_padding:
            # top buttom left right
            img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if is_debug:
            cv2.imwrite(f'{_debug_img_path}video-{uuid.uuid4()}.jpg', img)
        im0 = img.copy()

        device, imgSize = parse_model_config(_video_model)
        half = device.type != _device  # half precision only supported on CUDA
        result = {}
        # Load model
        t1 = time.time()
        model = _models[_video_model]
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        """
                数据源
        """
        imgOs = im0

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        pred, img = pre_parse_img(model=model, img=imgOs.copy(), stride=stride, imgsz=imgsz, device=device, half=half)

        # Apply NMS
        pred = non_max_suppression(pred, _conf_thres, _iou_thres, classes=None, agnostic=_agnostic_nms)
        # Process detections
        persons = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 sizes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgOs.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label_name = names[int(cls)]  # 类别
                    conf_val = float(conf)  # 置信度
                    if conf_val < parse_label_conf_value(labelName=label_name):
                        continue
                    result[change_txt[label_name]] = result.get(change_txt[label_name], 0) + 1
                    if 'success' not in result:
                        result['success'] = True

                    if label_name in ['person']:
                        persons.append({
                            "image": imgOs.copy(),
                            "conf_val": conf_val,
                            "area": (int(xyxy[3]) - int(xyxy[1])) * (int(xyxy[2]) - int(xyxy[0])),
                            "points": xyxy,
                            "value": conf_val * (int(xyxy[3]) - int(xyxy[1])) * (int(xyxy[2]) - int(xyxy[0]))
                        })
                if len(persons) >= 1:
                    params = {
                        "images": persons,
                        "is_debug": is_debug
                    }
                    future1 = pool.submit(lambda x: detect_cap(**x), params)
                    future2 = pool.submit(lambda x: detect_uniform(**x), params)
                    # 获取安全帽信息
                    result['cap'] = future1.result()
                    # 获取工服信息
                    result.update(future2.result())

        if 'success' not in result:
            result['success'] = False
        print(f'耗时：{time.time() - t1}')
        del_images(img_path)
        return result

    except Exception as e:
        print(e)
        del_images(img_path)
        result['success'] = False
        return result


# 删除文件
def del_images(path):
    if os.path.exists(path):
        os.remove(path)


@app.route('/WB_AI/petrochemical/report', methods=['POST'])
def petrochemical_predict():
    start = time.time()
    if request.method == 'POST':
        # 获取上传的文件,若果没有文件默认为None
        files = request.files.getlist('images', None)
        # is_padding = request.values.get('isPadding', False)
        alarm_type = request.values.get('alarmType', "no_safetycap")
        image_type = request.values.get('imageType', -1)
        is_debug = request.values.get('isDebug', False)
        if image_type == -1 and alarm_type == 'person_off_duty_querying':
            is_cut = False
        else:
            is_cut = (int(image_type) == 1)
        results = []
        for file in files:
            if file is None:
                return Error(HttpCode.servererror, 'no files for upload')
            img_name = f"./{str(uuid.uuid4())}.jpg"
            file.save(img_name)
            result = detect_img(img_name, is_padding=False, is_cut=is_cut, is_debug=is_debug)
            results.append(result)
        end_time = time.time()
        return Result(HttpCode.ok, "预测成功", cost=round(float(end_time - start), 3), data=results)
    else:
        return Error(HttpCode.servererror, '请求方式错误,请使用post方式上传')


@app.route("/WB_AI/petrochemical/modelConfig/view", methods=['GET'])
def view_model_config():
    global _config
    start = time.time()
    load_config()
    end_time = time.time()
    return Result(HttpCode.ok, "查看成功", cost=round(float(end_time - start), 3), data=_config)


@app.route("/WB_AI/petrochemical/modelConfig/update", methods=['POST'])
def update_model_config():
    start = time.time()
    if request.method == 'POST':
        modelConfig = request.json.get('modelConfig')
        labelConfig = request.json.get('labelConfig')
        data = {
            'modelConfig': modelConfig,
            'labelConfig': labelConfig
        }
        with open(_config_path, 'w') as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
        # 重新载入数据
        load_config()
        end_time = time.time()
        return Result(HttpCode.ok, "更新成功", cost=round(float(end_time - start), 3))
    else:
        return Error(HttpCode.servererror, '请求方式错误,请使用post方式上传')


if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=9797)
