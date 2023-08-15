import time
import paho.mqtt.client as mqtt
import threading
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import json
import socket
import atexit

from detect_img import detect_img
from detect_img_streams import detect_img_stream, recognize_head
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox, LoadStreamsWithWork
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.recongnized_color import identify_color
from utils.torch_utils import select_device, TracedModel
from urllib.parse import urlparse
from apscheduler.schedulers.background import BackgroundScheduler

_weights = {
    "vedio": "weights/video.pt",
    "safetyCap": "weights/safety_best8.pt",
    "clothes": "weights/union_best1.pt"
}
_models = {

}
subModel = {
    "safetyCap": {
        "demandLabels": "person_head",
        "labels": ["safety_cap"],
        "colors": "true"
    },
    "clothes": {
        "demandLabels": "person",
        "labels": ["upper_body", "lower_body"],
        "colors": "true"
    }
}
_polygon_points = [(200, 300), (1400, 300), (1400, 800), (200, 800)]
_videoDatas = []
_device = 'cpu'
_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_img_size = 640
_trace = True
_webcam = True
_view_img = True
_isShowPreview = False
_conf_thres = 0.45
_iou_thres = 0.45
_agnostic_nms = False
_labels = {
    "0": "person",
    "1": "person_head",
    "2": "alarm_fumes",
    "3": "alarm_fire",
    "4": "behavior_smoking",
    "5": "behavior_call",
    "6": "behavior_look_phone"
}
_status = False
_thread_detect = None
_frames = 0
_config_data = None


def parse_rtsp_url(url):
    """
    解析RTSP地址，获取其中的协议、主机、端口和路径等信息
    :param url: RTSP地址
    :return: 返回一个包含协议、主机、端口和路径等信息的元组
    """
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    host = parsed_url.hostname
    port = parsed_url.port or 554
    path = parsed_url.path or "/"
    return scheme, host, port, path


def check_rtsp_stream(url):
    """
    判断一个RTSP地址是否可以连接成功
    :param url: RTSP地址
    :return: True表示连接成功，False表示连接失败
    """
    scheme, host, port, path = parse_rtsp_url(url)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect((host, port))
            sock.close()
            return True
    except Exception as e:
        return False


# 加载模型
def init_model():
    # Initialize
    device = select_device(_device)
    for item in _weights:
        # Load model
        _models[item] = attempt_load(_weights[item], map_location=device)  # load FP32 model


# 解析模型参数,构造子模型任务
def parse_labels(msg: dict):
    """
    params:
      msg = {
            "type":"start_detect|restart_detect",
            "videoDatas":[
                {
                    "id":"0001",
                    "rtsp":"rtsp://root:root@192.168.1.183:554/axis-media/media.amp"
                },
                {
                    "id":"0002",
                    "rtsp":"rtsp://admin:webuild1234@192.168.1.64:554/h264/ch1/main/av_stream"
                }
            ],

            "imgSize":800,
            "labels":
                {
                   "person":{"enabled":true},
                   "cap":{"enabled":true,"color":true}
                   "person_head":{"enabled":true},
                   "alarm_fumes":{"enabled":true},
                   "alarm_fire":{"enabled":true},
                   "behavior_smoking":{"enabled":true},
                   "behavior_call":{"enabled":true},
                   "behavior_look_phone":{"enabled":true},
              }
            }
    result:
         subModel ={
            "safetyCap":{
                "demandLabels":"person",
                "labels":["safety_cap"],
                "colors":"true"
            },
            "clothes":{
                "demandLabels":"person",
                "labels":["upper_body","lower_body"],
                "colors":"true"
            }
        }
    """
    labels = msg['labels']
    try:
        with open("detect-core.config", "r") as file:
            config_data = json.load(file)
    except Exception as e:
        print(e)
    config_data = config_data['model']
    label = dict(filter(lambda x: x[1]['enabled'], labels.items()))
    subModel = {}
    if 'cap' in label:
        subModel["safetyCap"] = {
            "demandLabels": "person_head",
            "labels": ["safety_cap"],
            "color": labels['cap']['color']
        }
    if 'upper_body' in label or "lower_body" in label:
        subModel['clothes'] = {
            "demandLabels": "person",
            "labels": ["upper_body", "lower_body"],
            "color": labels['upper_body']['color'] or labels['upper_body']['color']
        }
    return subModel


# 模型检测
def detect(videoDatas, labels, imgSize=640, **subModel):
    global _videoDatas, _isShowPreview
    global _status
    global _img_size
    global _frames
    global _labels
    global _classes
    global _models
    global _polygon_points
    _videoDatas = videoDatas
    _labels = labels
    # 把_labels的key值转成list，内容为int
    # _classes = list(map(int, _labels.keys()))

    if len(_videoDatas) == 0:
        print('没有可用的视频流')
        return

    video_data = _videoDatas
    # for i in range(len(_videoDatas)):
    #     if check_rtsp_stream(_videoDatas[i]['rtsp']):
    #         video_data.append(_videoDatas[i])
    # if len(video_data) == 0:
    #     print('没有可用的视频流')
    #     return
    try:
        if imgSize is not None:
            _img_size = imgSize

        _source = [video_data[i]['rtsp'] for i in range(len(video_data))]

        # Initialize
        device = select_device(_device)

        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = _models['vedio'] if _models['vedio'] is not None else attempt_load(_weights,
                                                                                   map_location=device)  # load FP32 model

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(_img_size, s=stride)  # check img_size

        if _trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16

        # Set Dataloader
        if _webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreamsWithWork(_source, img_size=imgsz, stride=stride, videoDatas=_videoDatas)
        else:
            dataset = LoadImages(_source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        _status = True

        # 上报数据集合
        data_all = []
        loopCount = 0
        for path, img, im0s, vid_cap, alramId in dataset:
            if _status == False:
                break

            loopCount += 1

            # torch加载图片
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

            # t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
            # t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, _conf_thres, _iou_thres, classes=None, agnostic=_agnostic_nms)
            # t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if _webcam:  # batch_size >= 1
                    p, s, im0, frame, alarmid = path[i], '%g: ' % i, im0s[i].copy(), dataset.count, alramId[i]
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # 根据rtsp字符串值获取_videoDatas中对应的数据
                video_info = None
                for vd in video_data:
                    if vd['rtsp'] == p:
                        video_info = vd
                        break

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    video_OneFrame_Data = {}  # 单个摄像头单帧数据
                    detail_info = []
                    for c in det[:, -1].unique():
                        n = int((det[:, -1] == c).sum())  # detections per class
                        video_OneFrame_Data[names[int(c)]] = n
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # print(s)
                    # video_OneFrame_Data = []      #单个摄像头单帧数据

                    img_rois = {}
                    # Write results
                    person_id = 0
                    for *xyxy, conf, cls in reversed(det):

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 坐标
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度
                        # 提取label_name为人的图片流
                        if label_name == subModel['safetyCap']['demandLabels']:
                            person_id += 1
                            if 'person' not in img_rois:
                                img_rois['person'] = {}
                            img_rois['person'][f"person_{person_id}"] = im0s[i][int(xyxy[1]):int(xyxy[3]),
                                                                        int(xyxy[0]):int(xyxy[2])]

                        if _view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                         line_thickness=int(im0.shape[1] / 850))

                    for item in img_rois:
                        for sm in subModel:
                            detail_info.extend(detect_img_stream(img_rois[item], imgSize, model_name=sm,
                                                                 labelName=subModel[sm]['labels'], _models=_models,
                                                                 _device=_device, _trace=_trace,
                                                                 _agnostic_nms=_agnostic_nms,
                                                                 _conf_thres=_conf_thres, _iou_thres=_iou_thres,
                                                                 subModel=subModel))
                    person_ids = set([i.get('id') for i in detail_info])

                    detail_list_group = []
                    for x in person_ids:
                        temp = []
                        for y in detail_info:
                            if y.get('id') == x:
                                temp.append(y)
                        if temp:
                            detail_list_group.append(temp)
                    details = []
                    for item in detail_list_group:
                        p1 = {}
                        for person in item:
                            p1.update(person)
                        details.append(p1)
                    data_all.append({"id": video_info["id"], "alarmConfigId": alarmid, "data": video_OneFrame_Data,
                                     "detail": details})

                # Stream results
                if _view_img:
                    if not _isShowPreview:
                        # 创建一个窗口并设置大小
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)  # 使用cv2.WINDOW_NORMAL参数可以调整窗口大小
                        cv2.resizeWindow(str(p), 800, 600)  # 设置窗口的宽度和高度
                    _isShowPreview = True
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

            # t0和当前时间是否大于1分钟
            if time.time() - t0 >= 1:
                try:
                    build_body(data_all, loopCount, im0s)
                except Exception as e:
                    print(e)
                t0 = time.time()
                data_all = []
                loopCount = 0

        print(f'Done. ({time.time() - t0:.3f}s)')
    except Exception as e:
        _status = False
        print(e)


def build_body(data_all, loopCount, im0s):
    global _frames
    if len(data_all) > 0:
        video_list = {}
        for item in data_all:
            id = item['id']
            alarmId = item['alarmConfigId']
            id = f"{id}#{alarmId}"
            if id not in video_list:
                video_list[id] = {}
            for k, v in item['data'].items():
                if k not in video_list[id]:
                    video_list[id][k] = 0
                video_list[id][k] += v

        report_data_all = []
        for key in video_list:
            for inner_key in video_list[key]:
                video_list[key][inner_key] = int(round(video_list[key][inner_key] / loopCount))
            ids = key.split("#")
            id = ids[0]
            alarmId = ids[-1]

            detail = []
            # 匹配detail数据
            for d in data_all:
                if id == d['id'] and alarmId == d['alarmConfigId'] \
                        and video_list[key]['person'] == len(d['detail']):
                    detail = d['detail']
                    break
            if not detail:
                max_len = -1
                index = -1
                for i in range(0, len(data_all)):
                    if id == data_all[i]['id'] and alarmId == data_all[i]['alarmConfigId']:
                        if len(data_all[i]['detail']) > max_len:
                            max_len = len(data_all[i]['detail'])
                            index = i
                if index != -1:
                    detail = data_all[index]['detail'][:video_list[key]['person']]
                else:
                    detail = []
            report_data_all.append(
                {"id": id,
                 "alarmConfigId": alarmId,
                 "time": int(time.time()),
                 # "img": image_to_base64(im0s[alramId.index(alarmId)]),
                 "data": video_list[key],
                 "detail": detail})
        # 上报数据
        print(str(loopCount) + "-" + json.dumps(report_data_all))
        _frames = loopCount
        send_message(report_data_all)


def start_detect(videoDatas, labels, imgSize, subModel={}):
    stop_detect()
    global _thread_detect
    _thread_detect = threading.Thread(target=detect, args=(videoDatas, labels, imgSize), kwargs=subModel)
    _thread_detect.start()


def stop_detect():
    global _status
    _status = False
    global _thread_detect
    if _thread_detect != None:
        _thread_detect.join()
    _thread_detect = None
    cv2.destroyAllWindows()


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
    client.subscribe("/AI_Core_Service_Video/Service/Command")
    send_message_status()
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_message_status, 'interval', seconds=60)
    scheduler.start()
    atexit.register(lambda: (scheduler.remove_all_jobs(), scheduler.shutdown()))


# 获取消息回调
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))
    msg_obj = json.loads(msg.payload)
    if msg_obj["type"] == "start_detect":
        for item in msg_obj['videoDatas']:
            for config in item['config']:
                points = []
                for p in config['polygonPoints']:
                    points.append((p[0], p[-1]))
                config['polygonPoints'] = points
        # 解析模型
        subModel = parse_labels(msg_obj)
        start_detect(msg_obj["videoDatas"], msg_obj["labels"], msg_obj.get("imgSize", 640), subModel=subModel)
    elif msg_obj["type"] == "stop_detect":
        stop_detect()


# 发送当前状态
def send_message_status():
    # 判断client是否连接状态
    if client.is_connected():
        msg = json.dumps({"videoDatas": _videoDatas, "labels": _labels, "status": _status, "frames": _frames})
        """
            消息体携带盒子number config_data['number']
        """
        client.publish("/AI_Core_Service_Video/Service/Status", msg, qos=0, retain=False)


# 发送消息对象data
def send_message(data):
    if client.is_connected():
        msg = json.dumps(data)
        """
            消息体携带盒子number config_data['number']
        """
        client.publish("/AI_Core_Service_Video/Service/Data", msg, qos=0, retain=False)


def test_cap():
    subModel = {
        "safetyCap": {
            "demandLabels": "person_head",
            "labels": ["safety_cap"],
            "color": True
        }
    }
    img_rois = {
        "person":
            {"person_01": cv2.imread('./datasets/helmon/test/img.jpg'),
             "person_02": cv2.imread('./datasets/helmon/test/img2.jpg')}

    }
    print(detect_img_stream(img_rois['person'], 640, model_name='safetyCap',
                            labelName='safety_cap', subModel=subModel))


def test_LoadStream():
    data = {
        "type": "start_detect|restart_detect",
        "videoDatas": [
            {
                "id": "0001",
                "rtsp": "rtsp://admin:webuild1234@192.168.1.64:554/h264/ch1/main/av_stream",
                "config": [
                    {
                        "alarmConfigId": "1",
                        "videoId": "1",
                        "polygonPoints": [(200, 300), (1400, 300), (1400, 800), (200, 800)]
                    },
                    {
                        "alarmConfigId": "2",
                        "videoId": "1",
                        "polygonPoints": [(200, 300), (2400, 300), (2400, 800), (200, 800)]
                    }
                ]
            }
        ]
    }
    source = [item['rtsp'] for item in data['videoDatas']]
    dataset = LoadStreamsWithWork(source, img_size=640, stride=32, videoDatas=data['videoDatas'])
    for path, img, im0s, vid_cap, alarmConfig in dataset:
        print(alarmConfig)


def test_tuple():
    data = [[200, 300], [1400, 300], [1400, 800], [200, 800]]
    result = []
    for item in data:
        result.append((item[0], item[-1]))
    print(result)


def test_pic():
    init_model()
    img = cv2.imread('./datasets/helmon/test.jpg')
    result = detect_img(img, _models=_models)
    # cv2.imwrite('./output.jpg', base64_to_image(result['img']))
    del result['img']
    print(result)


def test_recognize_head():
    init_model()
    img = cv2.imread('./datasets/helmon/test.jpg')
    model = _models['vedio']
    recognize_head(img, model=model)


if __name__ == '__main__':
    # test_tuple()
    # test_LoadStream()
    # test_cap()
    # test_recognize_head()
    # result = test_pic()

    init_model()
    thread = threading.Thread(target=con_mqtt)
    thread.start()

    # 防止主线程退出
    while True:
        time.sleep(1)
