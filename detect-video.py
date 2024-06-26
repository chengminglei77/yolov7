import argparse
import time
import subprocess
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np

LABLES = ['person', 'person_head', 'clothes_upper', 'clothes_lower']  # 需要判断的目标信息
# LABLES = ['car']  # 需要判断的目标信息
RULE_TYPE = 'personoffDuty'  # 报警规则
RULE_TYPE = 'carDuty'  # 报警规则
# ROI = [[600, 620], [1400, 1440]]  # 报警区域
# ROI = [[1440, 80], [2200, 550]]  # 主码流报警区域
# ROI = [[130, 130], [390, 330]]  # 子码流，报警区域
# ROI = [[640, 320], [1000, 700]]  # 前台，报警区域
# ROI = [[380, 320], [780, 700]]  # 中韩现场画面，报警区域
# ROI = [[450, 600], [780, 710]]  # 监控室画面，报警区域
ROI = [[550, 200], [800, 350]]  # 车辆违停，报警区域
COLORS = [

    (0, 255, 0),  # 绿色
    (0, 0, 255)  # 红色
]


def draw_box_string(img, box, msg):
    """
    img: read by cv;
    box:[xmin, ymin, xmax, ymax];
    string: what you want to draw in img;
    return: img
    """
    x, y, x1, y1 = box

    cv2.rectangle(img, (x, y), (x1, y1), (48, 48, 255), 2)
    y -= 24
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simhei.ttf", 24, encoding="utf-8")
    p1_x, p1_y, p2_x, p2_y = font.getbbox(msg)
    chars_w = p2_x - p1_x
    chars_h = p2_y - p1_y
    coords = [(x, y), (x, y + chars_h),
              (x + chars_w, y + chars_h), (x + chars_w, y)]
    draw.polygon(coords, fill=(255, 24, 24))
    draw.text((x, y), msg, fill="white", font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def plot_one_box1(imgInfos):
    """
    第一类
    人员离岗 （区域，没人时显示红色，有人时显示绿色）
    车辆违停 （区域，没车时显示绿色，有车时显示红色）
    人员入侵 （区域，没人时显示绿色，有人时显示红色）
    超员/少员 (全画面，统计总人数，正常时不显示任何信息，不在区域范围时,显示)
    """
    status = False
    color = 0
    ps = []  # 三维数组
    message = ''

    def is_rect_cross(x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) < min(x2, x4) and max(y1, y3) < min(y2, y4):
            return True
        else:
            return False

    if RULE_TYPE == 'personoffDuty':
        ps = ROI
        color = 1
        message = '人员离岗'
        if len(imgInfos) > 0:
            # 获取区域坐标
            x1 = ROI[0][0]
            y1 = ROI[0][1]
            x2 = ROI[1][0]
            y2 = ROI[1][1]
            flag = False
            for item in imgInfos:
                xyxy = item.get('xyxy')
                x3 = int(xyxy[0])
                y3 = int(xyxy[1])
                x4 = int(xyxy[2])
                y4 = int(xyxy[3])
                # 区域内存在目标
                if is_rect_cross(x1, y1, x2, y2, x3, y3, x4, y4):
                    # 修改颜色为红色
                    flag = True
                    break
            if flag:
                color = 0
                message = ''
            else:
                status = True
                color = 1
        return status, message, color, ps
    elif RULE_TYPE == 'carDuty':
        ps = ROI
        color = 0
        status = True
        if len(imgInfos) > 0:
            message = '车辆违停'
            # 获取区域坐标
            x1 = ROI[0][0]
            y1 = ROI[0][1]
            x2 = ROI[1][0]
            y2 = ROI[1][1]
            flag = False
            for item in imgInfos:
                xyxy = item.get('xyxy')
                x3 = int(xyxy[0])
                y3 = int(xyxy[1])
                x4 = int(xyxy[2])
                y4 = int(xyxy[3])
                # 区域内存在目标
                if is_rect_cross(x1, y1, x2, y2, x3, y3, x4, y4):
                    # 修改颜色为红色
                    flag = True
                    break
            if flag:
                status = True
                color = 1
            else:
                color = 0
                message = ''
        return status, message, color, ps
    else:
        return status, message, color, ps


def detect():
    global LABLES

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print(device.type)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        view_img = check_imshow()
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    cap = cv2.VideoCapture(source)
    rtsp = 'rtsp://192.168.1.86/live/sz?secret=035c73f7-bb6b-4889-a715-d9eb2d1925cc'
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])

    command = ['ffmpeg',
               # '-hwaccel', 'cuda',
               '-y', '-an',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', sizeStr,
               '-r', '12',
               '-i', '-',
               # '-b_ref_mode', '0',
               '-c:v', 'libx264',  # 'libx265','libx264','hevc_nvenc','h264_nvenc'
               # '-b:v', '900k',
               # '-bufsize', '900k',
               # '-maxrate', '1400k',
               '-pix_fmt', 'yuvj420p',
               '-preset', 'ultrafast',  # ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow、placebo
               '-f', 'rtsp',
               '-rtsp_transport', 'tcp',
               rtsp]

    pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    start_time = time.time()
    fps_counter = 0
    out_fps = 0
    out_fps_list = []
    avg_fps = 0

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            save_path = str(save_dir / p.name)  # img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                imgInfos = []
                for *xyxy, conf, cls in reversed(det):

                    if names[int(cls)] in LABLES and conf > 0.25:
                        imgInfos.append({
                            'label': names[int(cls)],  # 目标标签
                            'conf': conf,  # 置信度
                            'xyxy': xyxy,  # 目标坐标
                        })
                        if names[int(cls)] != 'person':
                            label = f'blue_{names[int(cls)]} {conf:.2f}'
                        else:
                            label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                # 进行逻辑判断，是否触发报警,返回颜色和坐标信息
                # status, message, index, ps = plot_one_box1(imgInfos)
                # if message != '':
                #     box = []
                #     box.extend(ps[0])
                #     box.extend(ps[1])
                #     print(box)
                #     im0 = draw_box_string(im0, box, message)
                # try:
                #     cv2.rectangle(im0, ps[0], ps[1], COLORS[index], 3)
                # except:
                #     print(ps)

                # print(status, message, index, ps)
            # else:
            #     # 补丁1-没有识别到目标时，如果配置了人员离岗，按照人员离岗显示
            #     status, message, index, ps = plot_one_box1([])
            #     if message != '':
            #         box = []
            #         box.extend(ps[0])
            #         box.extend(ps[1])
            #         print(box)
            #         im0 = draw_box_string(im0, box, message)
            #     try:
            #         cv2.rectangle(im0, ps[0], ps[1], COLORS[index], 3)
            #     except:
            #         print(ps)
            #
            #     print(status, message, index, ps)
            fps_counter += 1  # 计算帧数
            if (time.time() - start_time) != 0:  # 实时显示帧数
                out_fps = fps_counter / (time.time() - start_time)
                out_fps_list.append(out_fps)
                avg_fps = np.mean(out_fps_list)
                fps_counter = 0
                start_time = time.time()

            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, FPS:{out_fps:.1f},AVG_FPS:{avg_fps:.2f} ')

            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 24, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                # 定义编解码器
                # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
            # vid_writer.write(im0)
            pipe.stdin.write(im0.tostring())
            cv2.imwrite("test.jpg", im0)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/video.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        # default="rtsp://admin:webuild1234@192.168.1.64:554/h264/ch1/sub/av_stream"
                        # default="rtsp://192.168.1.85:10054/live/TzChAdfSR"
                        default="datasets/video5.mp4"
                        , help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
