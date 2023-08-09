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

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
from urllib.parse import urlparse
from apscheduler.schedulers.background import BackgroundScheduler

_weights = "weights/video.pt"
# _videoDatas = [{ "id":"0001","rtsp":"rtsp://root:root@192.168.1.183:554/axis-media/media.amp"},{ "id":"0002","rtsp":"rtsp://admin:webuild1234@192.168.1.64:554/h264/ch1/main/av_stream"}]
_videoDatas = []
_device = 'cpu'
_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_img_size = 640
_trace = True
_webcam = True
_view_img = True
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


#

def detect(videoDatas, labels, imgSize=640):
    global _videoDatas
    global _status
    global _img_size
    global _frames
    global _labels
    global _classes
    _videoDatas = videoDatas
    _labels = labels
    # 把_labels的key值转成list，内容为int
    _classes = list(map(int, _labels.keys()))

    if len(_videoDatas) == 0:
        print('没有可用的视频流')
        return

    video_data = []
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
        model = attempt_load(_weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(_img_size, s=stride)  # check img_size

        if _trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16

        # Set Dataloader
        if _webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(_source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(_source, img_size=imgsz, stride=stride)

        # rtsp = 'rtsp://192.168.1.86/live/sz?secret=035c73f7-bb6b-4889-a715-d9eb2d1925cc'
        # size = (int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # sizeStr = str(size[0]) + 'x' + str(size[1])

        # command = ['ffmpeg.exe',
        #     # '-hwaccel', 'cuda',
        #     '-y', '-an',
        #     '-f', 'rawvideo',
        #     '-vcodec','rawvideo',
        #     '-pix_fmt', 'bgr24',
        #     '-s', sizeStr,
        #     '-r', '16.5',
        #     '-i', '-',
        #     # '-b_ref_mode', '0',
        #     '-c:v', 'libx264',#'libx265','libx264','hevc_nvenc','h264_nvenc'
        #     # '-b:v', '900k',
        #     # '-bufsize', '900k',
        #     # '-maxrate', '1400k',
        #     '-pix_fmt', 'yuvj420p',
        #     '-preset', 'ultrafast',#ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow、placebo
        #     '-f', 'rtsp',
        #     '-rtsp_transport', 'tcp',
        #     rtsp]
        # pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        # start_time = time.time()
        # fps_counter = 0
        # out_fps = 0
        # out_fps_list=[]
        # avg_fps = 0
        t0 = time.time()

        _status = True

        # 上报数据集合
        data_all = []
        loopCount = 0
        for path, img, im0s, vid_cap in dataset:
            if _status == False:
                break

            loopCount += 1
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
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

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
                    for c in det[:, -1].unique():
                        n = int((det[:, -1] == c).sum())  # detections per class
                        l = int(c)
                        name = _labels.get(str(l), "unknown")
                        video_OneFrame_Data[name] = n

                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # print(s)
                    # video_OneFrame_Data = []      #单个摄像头单帧数据

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 坐标
                        label_name = names[int(cls)]  # 类别
                        conf_val = float(conf)  # 置信度


                        if _view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                         line_thickness=int(im0.shape[1] / 850))

                        # video_OneFrame_Data.append({"id":video_info["id"],"label":int(cls),"name":label_name, "position":{"x1":c1[0], "y1":c1[1], "x2":c2[0], "y2":c2[1]}, "conf":conf_val})

                    data_all.append({"id": video_info["id"], "data": video_OneFrame_Data})

                # fps_counter += 1  # 计算帧数
                # if (time.time() - start_time) != 0:  # 实时显示帧数
                #     out_fps = fps_counter / (time.time() - start_time)
                #     # out_fps_list.append(out_fps)
                #     # avg_fps = np.mean(out_fps_list)
                #     fps_counter = 0
                #     start_time = time.time()

                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, FPS:{out_fps:.1f},AVG_FPS:{avg_fps:.2f} ')

                # pipe.stdin.write(im0.tostring())

                # Stream results
                if _view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

            # t0和当前时间是否大于1分钟
            if time.time() - t0 >= 1:
                try:
                    if len(data_all) > 0:
                        video_list = {}
                        for item in data_all:
                            id = item['id']
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
                            report_data_all.append({"id": key, "data": video_list[key]})
                        # 上报数据
                        print(str(loopCount) + "-" + json.dumps(report_data_all))
                        _frames = loopCount
                        send_message(report_data_all)
                except Exception as e:
                    print(e)
                t0 = time.time()
                data_all = []
                loopCount = 0

        print(f'Done. ({time.time() - t0:.3f}s)')
    except Exception as e:
        _status = False
        print(e)


def start_detect(videoDatas, labels, imgSize):
    stop_detect()
    global _thread_detect
    _thread_detect = threading.Thread(target=detect, args=(videoDatas, labels, imgSize))
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
        with open("detect-core.config", "r") as file:
            config_data = json.load(file)

        client.keep_alive = 60
        config_data = config_data['mqtt']
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
        start_detect(msg_obj["videoDatas"], msg_obj["labels"], msg_obj.get("imgSize", 640))
    elif msg_obj["type"] == "stop_detect":
        stop_detect()


# 发送当前状态
def send_message_status():
    # 判断client是否连接状态
    if client.is_connected():
        msg = json.dumps({"videoDatas": _videoDatas, "labels": _labels, "status": _status, "frames": _frames})
        client.publish("/AI_Core_Service_Video/Service/Status", msg, qos=0, retain=False)


# 发送消息对象data
def send_message(data):
    if client.is_connected():
        msg = json.dumps(data)
        client.publish("/AI_Core_Service_Video/Service/Data", msg, qos=0, retain=False)


if __name__ == '__main__':
    thread = threading.Thread(target=con_mqtt)
    thread.start()

    # 防止主线程退出
    while True:
        time.sleep(1)

    # detect()
