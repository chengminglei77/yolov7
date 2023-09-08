import json

import cv2
import torch
import numpy as np
from numpy import random

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.recongnized_color import identify_color
from utils.torch_utils import select_device, TracedModel


# 视频报警_识别图片流
def detect_img_stream(imgs, imgSize=640, model_name='safety_cap', labelName=[], subModel={}, _device='cpu', _models={},
                      _trace=True, _conf_thres=0.45, _iou_thres=0.45, _agnostic_nms=False):
    try:
        # Initialize
        device = select_device(_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        if model_name not in _models:
            raise NameError(f"{model_name} not exsit,pleas concact system Administrator")
        model = _models[model_name]

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size
        # if _trace:
        #     model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16
        """
            数据源
        """
        result = []
        for img in imgs:
            person_id = img
            temp = {'id': person_id}
            img0 = imgs[img]
            if model_name == 'safetyCap':
                flag, img0 = recognize_head(img0, labelName="person_head", model=_models['vedio'])
            # padded resize
            img = letterbox(img0, imgsz, stride)[0]
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
            image_info = {}
            for i, det in enumerate(pred):  # detections per image
                s = ''
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 坐标
                        label_name = names[int(cls)]  # 类别
                        if label_name not in labelName:
                            break
                        conf_val = float(conf)  # 置信度
                        info = {
                            "conf_val": conf_val,
                            "area": (int(xyxy[3]) - int(xyxy[1])) * (int(xyxy[2]) - int(xyxy[0])),
                            "points": xyxy
                        }
                        if label_name not in image_info:
                            image_info[label_name] = []
                        image_info[label_name].append(info)
                # # 未识别出对象
                # else:
                #     for item in labelName:
                #         image_info[item] = []

                # 循环遍历获取最明显的对象信息
            for item in labelName:
                # 对应label有数据
                if item in image_info and image_info[item]:
                    # 按照覆盖面积排序
                    image_info[item].sort(key=lambda k: (k.get('area', 0)))
                    # 获取面积最大的一个
                    info = image_info[item][-1]
                    if item == 'safety_cap':
                        temp['cap'] = {'type': item}
                    else:
                        temp[item] = {}

                    if model_name in subModel and subModel[model_name]["color"]:
                        # 截取图片
                        xyxy = info['points']
                        img_roi = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        # 检测颜色
                        txt = identify_color.get_color(img_roi)
                        temp.update(txt)
                # 对应label无数据
                else:
                    if item == 'safety_cap':
                        temp['cap'] = {'type': "no_cap"}
                    else:
                        temp[item] = 'unknown'
                    result.append(temp)

        return result
    except Exception as e:
        print(e)


# 视频识别—获取person_head位置
def recognize_head(img, imgSize=320, model=None, labelName='person_head', _device='cpu',
                   _trace=True, _conf_thres=0.25, _iou_thres=0.45, _agnostic_nms=False):
    try:
        # Initialize
        device = select_device(_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if model is None:
            raise NameError(f"model not exist,please concact system administrator")
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgSize, s=stride)  # check img_size
        # if _trace:
        #     model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16
        img0 = img
        # img0 = cv2.copyMakeBorder(img0, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = letterbox(img0, imgsz, stride)[0]
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
        header_imgs = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label_name = names[int(cls)]  # 类别
                    conf_val = float(conf)  # 置信度
                    if label_name == labelName and conf_val >= _conf_thres:
                        header_imgs.append({
                            'conf': conf_val,
                            'image': img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        })

        if len(header_imgs) >= 1:
            # 排序
            sorted(header_imgs, key=lambda k: (k.get('conf', 0)))
            images = [item['image'] for item in header_imgs]
            return True, images

        return False, [img0]
    except Exception as e:
        print(e)
