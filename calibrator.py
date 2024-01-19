import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
import numpy as np
import cv2

LOGGER = trt.Logger(trt.Logger.VERBOSE)


class Calibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, quantification=1, batch_size=1, height=640, width=640, calibration_images="", cache_file=""):
        trt.IInt8EntropyCalibrator.__init__(self)
        self.index = 0
        self.length = quantification
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.height = height
        self.width = width
        self.img_list = [calibration_images + '/' + l for l in os.listdir(calibration_images)]
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = self.preprocess(img)
                self.calibration_data[i] = img
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, name):
        batch = self.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def preprocess(self, img):
        h, w, c = img.shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_w = self.width / w
        r_h = self.height / h
        if r_h > r_w:
            tw = self.width
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.height - th) / 2)
            ty2 = self.height - th - ty1
        else:
            tw = int(r_h * w)
            th = self.height
            tx1 = int((self.width - tw) / 2)
            tx2 = self.width - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        return image


def buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                img_height, img_wdith, calibration_images, calibration_cache):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if FP16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    elif INT8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(quantification, batch_size, img_height, img_wdith, calibration_images,
                                            calibration_cache)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")
        # exit(0)

    with open(engine_file, "wb") as f:
        f.write(engine)


def main():
    quantification = 1  # 量化次数
    batch_size = 1
    img_height = 640
    img_wdith = 640
    calibration_images = "./images"
    onnx_file = "./weights/yolov7.onnx"
    engine_file = "./weights/yolov7.engine"
    calibration_cache = "./weights/yolov7_calibration.cache"

    '''
    模型使用单精度量化，设置 FP16_mode = False & INT8_mode = False  (FP32)
    模型使用半精度量化，设置 FP16_mode = True  & INT8_mode = False  (FP16)
    模型使用 Int8量化，设置 FP16_mode = False & INT8_mode = True   (INT8)
    '''
    FP16_mode = False
    INT8_mode = False

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print('Load ONNX file from:%s \nStart export, Please wait a moment...' % (onnx_file))
    buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                img_height, img_wdith, calibration_images, calibration_cache)
    print('Export ENGINE success, Save as: ', engine_file)


if __name__ == '__main__':
    main()
