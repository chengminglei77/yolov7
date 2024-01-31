import os
import yaml

import models
import test
import torch
import collections
from pathlib import Path
from utils.activations import SiLU
from EfficientNMS import End2End
from models.yolo import Model
from pytorch_quantization import calib
from absl import logging as quant_logging
from utils.datasets import create_dataloader
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
import torch.nn as nn

from utils.torch_utils import select_device


def load_yolov7_model(weight, device='cpu'):
    ckpt = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=20).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model


# 预处理验证集
def prepare_val_dataset(cocodir, batch_size=4):
    dataloader = create_dataloader(
        f"{cocodir}/val.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader


# 预处理训练集集
def prepare_train_dataset(cocodir, batch_size=4):
    with open("./data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    dataloader = create_dataloader(
        f"{cocodir}/train.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader


# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)


# 自动插入DQD值
def prepare_model(weight, device):
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()  # conv bn 进行层的合并, 加速
    return model


# 手动插入DQD值
def transfer_torch_to_quantization(nn_instance, quant_module):
    quant_instances = quant_module.__new__(quant_module)

    # 属性赋值
    for k, val in vars(nn_instance).items():
        setattr(quant_instances, k, val)

    # 初始化
    def __init__(self):
        # 返回两个 QuantDescriptor 的实例 self.__class__ 是 quant_instance 的类, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            # 加快量化速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instances)
    return quant_instances


import re


def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False


def tranfer_torch_to_quantization(nn_instance, quant_module):
    quant_instances = quant_module.__new__(quant_module)

    # 属性赋值
    for k, val in vars(nn_instance).items():
        setattr(quant_instances, k, val)

    # 初始化
    def __init__(self):
        # 返回两个 QuantDescriptor 的实例 self.__class__ 是 quant_instance 的类, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            # 加快量化速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instances)
    return quant_instances


def torch_module_find_quant_module(model, module_list, ignore_layer, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, ignore_layer, prefix=path)  # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization : {path} has ignored.")
                continue
            # 转换
            model._modules[name] = tranfer_torch_to_quantization(submodule, module_list[submodule_id])


def replace_to_quantization_model(model, ignore_layer=None):
    module_list = {}

    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod

    torch_module_find_quant_module(model, module_list, ignore_layer)


def evaluate_coco(model, loader, save_dir='', conf_thres=0.001, iou_thres=0.45):
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        "./dataset/person/data.yaml",
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]


# 收集模型在给定数据集上的激活统计信息
def collect_stats(model, data_loader, device, num_batch=4):
    # 设置模型为 eval 模型，确保不启用如 dropout 这样的训练特有的行为
    model.eval()

    # 开启校准器
    #  遍历模型的所有模块，对于每一个 TensorQuantizer 实例
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果有校准器存在，则禁用量化（不对输入进行量化）并启动校准模式（收集统计信息）
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                # 如果没有校准器，则完全禁用该量化器（不执行任何操作）
                module.disable()

    # test
    with torch.no_grad():
        # 使用 data_loader 来提供数据，并通过模型执行前向传播
        for i, datas in enumerate(data_loader):
            # 将数据转移到 device 上，并进行适当的归一化
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            # 对每个批次数据，模型进行推理，但不进行梯度计算，收集激活统计信息直到处理指定数量的批次
            model(imgs)
            if i >= num_batch:
                break

    # 关闭校准器
    # 遍历模型的所有模块，对于每一个 TensorQuantizer 实例
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # 如果有校准器存在，则启用量化并禁用校准模式
                module.enable_quant()
                module.disable_calib()
            else:
                # 如果没有校准器，则重新启用该量化器
                module.enable()


# 一旦收集了激活的统计信息，该函数就会计算量化的尺度因子 amax（动态范围的最大值
def compute_amax(model, **kwargs):
    # 遍历模型的所有模块，对于每一个 TensorQuantizer 实例

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果有校准器存在，则根据收集的统计信息计算 amax 值，这个值代表了激活的最大幅值，用于确定量化的尺度
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                # 将 amax 值转移到 device 上，以便在后续中使用
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device):
    # 收集前向信息
    collect_stats(model, dataloader, device)

    # 获取动态范围，计算 amax 值，scale 值
    compute_amax(model, method='mse')


def export_ptq(model, save_file, device, dynamic_batch=True):
    input_dummy = torch.randn(1, 3, 640, 640, device=device)

    # 打开 fake 算子
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        # torch.onnx.export(model, input_dummy, save_file, opset_version=13,
        #                   input_names=['images'], output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'],
        #                   dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None)
        torch.onnx.export(model, input_dummy, save_file, opset_version=13,
                          input_names=['images'], output_names=['output'],
                          dynamic_axes=None)


def export(model, save_file, device):
    # 打开 fake 算子
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    img = torch.randn(1, 3, 640, 640, device=device)
    model.eval()
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) or isinstance(m,
                                                           models.common.RepConv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    model.model[-1].export = False  # set Detect() layer grid export
    model = End2End(model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=False,
                    device=device)

    y = model(img)  # dry run
    # ONNX export
    try:
        torch.onnx.export(model, img, save_file, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'],
                          dynamic_axes=None)
    except Exception as e:
        print('ONNX export failure: %s' % e)


# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

    return False


class disable_quantization:

    # 初始化
    def __init__(self, model):
        self.model = model

    # 应用 关闭量化
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(disabled=True)

    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)


# 重启量化
class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)


import json


class SummaryTools:

    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def sensitive_analysis(model, loader):
    save_file = "senstive_analysis.json"

    summary = SummaryTools(save_file)

    # for 循环每一个层
    print(f"Sensitive analysis by each layer...")
    # for 循环 model 的每一个 quantizer 层
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断 layer 是否是量化层
        if have_quantizer(layer):  # 如果是量化层
            # 使该层的量化失效，不进行 int8 的量化，使用 fp16 精度运算
            disable_quantization(layer).apply()

            # 计算 map 值
            # 验证模型的精度, evaluate_coco(), 并保存精度值
            ap = evaluate_coco(model, loader)

            # 保存精度值，json 文件
            summary.append([ap, f"model.{i}"])
            print(f"layer {i} ap: {ap}")

            # 重启层的量化，还原
            enable_quantization(layer).apply()

        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束，打印前 10 个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary")
    # 排序，得到前 10 个对精度影响比较大的层，将这些层进行打印输出
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


if __name__ == "__main__":
    weight = "./weights/video.pt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pth_model = load_yolov7_model(weight, device)
    cocodir = "./dataset/person"

    val_dataloader = prepare_val_dataset(cocodir)
    train_dataloader = prepare_train_dataset(cocodir)

    # pth 模型验证
    print("Evalute Origin...")
    ap = evaluate_coco(pth_model, val_dataloader)

    # 获取伪量化模型(手动 initial(), 手动插入 QDQ)
    model = prepare_model(weight, device)
    replace_to_quantization_model(model)

    # 模型标定
    calibrate_model(model, train_dataloader, device)

    # 敏感层分析
    sensitive_analysis(model, val_dataloader)

    # PTQ 模型验证
    print("Evaluate PTQ...")
    ptq_ap = evaluate_coco(model, val_dataloader)

    # PTQ 模型导出
    print("Export PTQ...")

    export(model, "weights/video.onnx", device)
