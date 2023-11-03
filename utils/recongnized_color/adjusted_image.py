import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils.recongnized_color import identify_color


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    """
        百分位数是统计中使用的度量，表示小于这个值的观察值的百分比
        a: 输入数组
        q: 要计算的百分位数，在 0 ~ 100 之间
        axis: 沿着它计算百分位数的轴
        keepdims :bool是否保持维度不变
        首先明确百分位数：第 p 个百分位数是这样一个值，它使得至少有 p% 的数据项小于或等于这个值，且至少有 (100-p)% 的数据项大于或等于这个值。
        【注】举个例子：高等院校的入学考试成绩经常以百分位数的形式报告。比如，假设某个考生在入学考试中的语文部分的原始分数为 54 分。相对于参加同一考试的其他学生来说，
        他的成绩如何并不容易知道。但是如果原始分数54分恰好对应的是第70百分位数，我们就能知道大约70%的学生的考分比他低，而约30%的学生考分比他高。这里的 p = 70。
    """
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """图像亮度增强"""
    # if get_lightness(src) > 130:
    #     print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 5, 250, cv2.NORM_MINMAX)
    # cv2.imwrite('../resources/images/out.png', out)
    return out


def adaptive_contrast_enhancement(image):
    # 将图像从BGR色彩空间转换为LAB色彩空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 分离L、A、B通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 对L通道进行自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    l_channel_eq = clahe.apply(l_channel)

    # 合并L、A、B通道
    lab_image_eq = cv2.merge([l_channel_eq, a_channel, b_channel])

    # 将图像从LAB色彩空间转换回BGR色彩空间
    enhanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)

    return enhanced_image

# 扩展图片增强
def expand_aug(src):
    print(identify_color.get_color(src))
    img_1_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    histogram(img_1_gray)

    img = aug(src)

    img_1_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    histogram(img_1_gray)
    print(identify_color.get_color(img))


def equalizeHist_image(src):
    # 应用直方图均衡化
    equalized_image = cv2.equalizeHist(src)

    # 显示原始图像和均衡化后的图像
    cv2.imshow('Original Image', src)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    return lightness


def histogram(img):
    img = img.reshape(-1)  # 将图像展开成一个一维的numpy数组
    plt.hist(img, 128)  # 将数据分为128组
    plt.show()


#  色阶调整算法
def levelAdjust(img, Sin=0, Hin=255, Mt=1.0, Sout=0, Hout=255):
    """
    Photoshop 的色阶调整分为输入色阶调整和输出色阶调整。
        输入色阶调整有 3 个调节参数：黑场阈值、白场阈值和灰场值：
        Sin ，输入图像的黑场阈值，input shadows
        Hin ，输入图像的白场阈值，input hithlight
        Mt，中间调，灰场调节值，midtone
        输入图像中低于黑场阈值的像素置 0 （黑色），高于白场阈值的像素置 255（白色）。灰场调节值默认值 1.0，调节范围 [0.01, 9.99]。灰场调节值增大的效果是加灰降对比度，
        减小的效果是减灰加对比度。
        输出色阶调整有 2个调节参数：黑场阈值 Sout白场阈值  Hout分别对应着输出图像的最小像素值、最大像素值。
    :param img:
    :param Sin:
    :param Hin:
    :param Mt:
    :param Sout:
    :param Hout:
    :return:
    """
    Sin = min(max(Sin, 0), Hin - 2)  # Sin, 黑场阈值, 0<=Sin<Hin
    Hin = min(Hin, 255)  # Hin, 白场阈值, Sin<Hin<=255
    Mt = min(max(Mt, 0.01), 9.99)  # Mt, 灰场调节值, 0.01~9.99
    Sout = min(max(Sout, 0), Hout - 2)  # Sout, 输出黑场阈值, 0<=Sout<Hout
    Hout = min(Hout, 255)  # Hout, 输出白场阈值, Sout<Hout<=255

    difIn = Hin - Sin
    difOut = Hout - Sout
    table = np.zeros(256, np.uint8)
    for i in range(256):
        V1 = min(max(255 * (i - Sin) / difIn, 0), 255)  # 输入动态线性拉伸
        V2 = 255 * np.power(V1 / 255, 1 / Mt)  # 灰场伽马调节
        table[i] = min(max(Sout + difOut * V2 / 255, 0), 255)  # 输出线性拉伸

    imgTone = cv2.LUT(img, table)
    return imgTone


def test_show_levelAdjust(img):
    equ1 = levelAdjust(img, 10, 225, 1.0, 10, 245)
    # equ1 = equ1.astype(np.uint8)
    print(identify_color.get_color(equ1))
    equ2 = levelAdjust(img, 10, 225, 1.5, 10, 245)
    # equ2 = equ2.astype(np.uint8)
    print(identify_color.get_color(equ2))
    plt.figure(figsize=(9, 6))
    plt.subplot(131), plt.title("origin"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("colorEqu1"), plt.axis('off')
    plt.imshow(cv2.cvtColor(equ1, cv2.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("colorEqu2"), plt.axis('off')
    plt.imshow(cv2.cvtColor(equ2, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def compare_gamma_log_clash(image, path='../../runs/'):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 标准化结果
    img1 = cv2.imread(image)
    img1 = aug(img1)
    print("normlize-" + str(image) + f"{identify_color.get_color(img1)}")
    cv2.imwrite(f"{path}normlize-" + image.split("/")[-1], img1)
    # gamma变换
    gamma, gain, scale = 0.7, 1, 255
    gamma_img = np.zeros_like(img)
    for i in range(3):
        gamma_img[:, :, i] = ((img[:, :, i] / scale) ** gamma) * scale * gain
    gamma_img = cv2.cvtColor(gamma_img, cv2.COLOR_RGB2HSV)
    gamma_img[:, :, 1] = gamma_img[:, :, 1] * 1.05
    gamma_img = cv2.cvtColor(gamma_img, cv2.COLOR_HSV2RGB)
    print("gamma-" + str(image) + f"{identify_color.get_color(gamma_img)}")
    cv2.imwrite(f"{path}gamma-" + image.split("/")[-1],  cv2.cvtColor(gamma_img, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    path = '../../datasets'
    images = os.listdir(path)
    for item in images:
        # result = identify_color.cut_rect_image(f'{path}/{item}')
        compare_gamma_log_clash(f'{path}/{item}')
        # img = cv2.imread(f'{path}/{item}')
        # equ2 = levelAdjust(img, 10, 225, 1.5, 10, 245)
        # print(f'{item}-{identify_color.get_color(equ2)}')
    # test_show_levelAdjust(img)
    # expand_aug(img)
