import uuid

import cv2
import numpy as np
from utils.recongnized_color import hsv_color_define

from PIL import Image

from utils.recongnized_color.adjusted_image import levelAdjust, aug

filename = '../../datasets/38.png'

change_txt = {
    "black": "black",
    "black1": "black",
    "gray": "gray",
    "gray1": "gray",
    "white": "white",
    "red": "red",
    "red2": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "cyan": "cyan",
    "blue": "blue",
    "purple": "purple"
}


# 抠图
def cut_img(frame):
    img2 = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    rect = (2, 1, frame.shape[0], frame.shape[1])
    # 背景模型： 用于grabcut计算，如果为null，则函数内部会自动创建一个bgModel,必须时单通道浮点型,且行数只能为1，列数只能为13*5
    bgdmodel = np.zeros((1, 65), np.float64)
    # 前景模型： 用于grabcut计算，如果为null，则函数内部会自动创建一个fgModel,必须时单通道浮点型,且行数只能为1，列数只能为13*5
    fgdmodel = np.zeros((1, 65), np.float64)
    """
        grabcut适用迭代图分割交互式前景提取,该算法适用了颜色、对比度和用户输入来获得结果，通过构建在算法之的应用程序的数量。
        参数说明：
            image: 进行grabcut操作的输入图像
            rect: 包括哟啊分割的区域边界框举行，仅在设置cv2.GC_INT_WITH_RECT时，适用此参数
            itercount: 运行迭代次数
            bgmodel,fgmodel: 算法适用两个大小为（1，65）的float64数组
            model:
                cv2.GC_INIT_WITH_RECT模式：意味着有rect参数指定正在初始化的grabcut算法适用矩形掩码
                cv2.GC_INIT_WITH_MASK模式：意味着正在适用mask输入参数指定的通用掩码（基本是一个二值图像）初始化Grabcut算法，蒙版具有与输入图像相同的形状
        返回参数：
            对掩码进行选择，mask只能取以下四种值：
            GCD_BGD(=0) 背景
            GCD_FGD(=1) 前景
            GCD_PR_BGD(=2) 可能背景
            GCD_PR_FGD(=3) 可能前景
    """
    cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 100, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    output = cv2.bitwise_and(img2, img2, mask=mask2)

    cv2.imshow('output', output)
    cv2.imshow('input', frame)

    print(get_color(output))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 处理图片
def get_color(frame):
    # cv2.imwrite('test.jpg', frame)
    frame = aug(frame)
    x, y, c = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    areas = {}
    color_dict = hsv_color_define.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        # cv2.imwrite(f'{d}.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        d = change_txt[d]
        if d not in areas:
            areas[d] = round(sum / (x * y), 4)
        else:
            areas[d] = max(areas[d], round(sum / (x * y), 4))
        if sum > maxsum:
            maxsum = sum
            color = d
    result = {
        'mainColor': color,
        'colorRatio': areas
    }
    return result


# 处理图片
def deal_color(frame, name):
    x, y, c = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    areas = {}
    color_dict = hsv_color_define.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(f'.\\runs\\color\\{name}-{d}.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        d = change_txt[d]
        if d not in areas:
            areas[d] = round(sum / (x * y), 4)
        else:
            areas[d] = max(areas[d], round(sum / (x * y), 4))
        if sum > maxsum:
            maxsum = sum
            color = d
    result = {
        'mainColor': color,
        'colorRatio': areas
    }
    print(f"{name}:{result}")


def check_pic(path):
    try:
        Image.open(path).load()
    except:
        return True
    else:
        return False


def cut_rect_image(filename):
    image = cv2.imread(filename)
    if not check_pic(filename):
        return image
    # 将图片从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义红色的HSV范围
    lower_red = np.array([0, 50, 45])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 50, 45])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    # 合并红色的掩码
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 定义橙色的HSV范围
    lower_orange = np.array([11, 105, 100])
    upper_orange = np.array([25, 255, 255])
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    # # 合并红色和橙色的掩码
    mask = cv2.bitwise_or(mask_red, mask3)

    # 去除黄色内容
    lower_yellow = np.array([26, 50, 50])
    upper_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_yellow))

    # 执行膨胀操作，将像素1的线框进行加粗处理
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 找到红色线框的轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的闭环红色线框
    max_contour = None
    max_area = 0
    y0 = 0
    y1 = 0
    x0 = 0
    x1 = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
            y0, y1, x0, x1 = get_rect_points(contour, image)

    # 创建一个与原始图像大小相同的空白图像，将闭环的红色线框内的内容抠出来
    mask = np.zeros_like(image)
    try:
        cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), -1)
        result = cv2.bitwise_and(image, mask)
        result = result[y0:y1, x0:x1]
        return result
    except:
        return image


def get_rect_points(contour, frame):
    weight = []
    height = []
    for item in contour:
        height.append(item[-1][1])
        weight.append(item[-1][0])
    frame = frame[min(height):max(height), min(weight):max(weight)]
    return min(height), max(height), min(weight), max(weight)


if __name__ == '__main__':
    cut_rect_image(filename)
