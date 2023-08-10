import base64

from PIL import Image, ImageDraw
import cv2
import numpy as np


# opencv 图片转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


# base64 转opencv图片
def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.frombuffer(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


def pre_process_image(img, polygon_points):
    image = cv_convert_pil(img)
    # 创建一个与原始图片相同大小的透明图片
    mask = Image.new("L", image.size, 0)
    # 使用多边形的坐标点创建一个ImageDraw对象
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)
    # 将透明图片作为蒙版，裁剪原始图片
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)
    # 保存裁剪后的图片
    rgb_im = result.convert('RGB')
    rgb_img = pil_convert_opencv(rgb_im)
    return rgb_img


def crop_image_with_polygon(image_path, polygon_points, output_path):
    # 打开原始图片
    image = Image.open(image_path)
    # 创建一个与原始图片相同大小的透明图片
    mask = Image.new("L", image.size, 0)
    # 使用多边形的坐标点创建一个ImageDraw对象
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)
    # 将透明图片作为蒙版，裁剪原始图片
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)
    # 保存裁剪后的图片
    rgb_im = result.convert('RGB')
    # rgb_im.save(output_path)
    return rgb_im


# PIL.Image转换成OpenCV格式
def pil_convert_opencv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# opencv转换成pil.image格式
def cv_convert_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# 调用示例
#
# image_path = "img.jpg"  # 原始图片路径
# polygon_points = [(200, 300), (1400, 300), (1400, 800), (200, 800)]  # 多边形的坐标点
# output_path = "cropped_image.jpg"  # 裁剪后的图片路径
# crop_image_with_polygon(image_path, polygon_points, output_path)
#
# image_path = '../../datasets/helmon/color/000027.jpg'
# img = cv2.imread(image_path)
# img_64 = image_to_base64(img)
# print(img_64)
# img_out = base64_to_image(img_64)
# cv2.imwrite("output.jpg", img_out)
