import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils.recongnized_color import hsv_color_define
import uuid

filename = '../../datasets/color/36.png'


# 抠图
def cut_img(frame):
    img2 = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    print(get_color(frame))
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


def get_mean_color(frame, num_colors):
    # 将图像转换为RGB格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 调整图像大小（可选）
    # image = cv2.resize(image, (800, 600))
    # 将图像转换为一维数组
    pixels = image.reshape(-1, 3)
    # 使用K均值聚类算法提取主要颜色
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # 获取聚类中心的RGB值
    colors = kmeans.cluster_centers_

    return colors.astype(int)


# 处理图片
def get_color(frame):
    # cv2.imwrite('test.jpg', frame)
    x, y, c = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    areas = {}
    color_dict = hsv_color_define.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        # cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if d == 'red2':
            d = 'red'
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


if __name__ == '__main__':
    frame = cv2.imread(filename)
    print(get_color(frame))
    # print(get_mean_color(frame, 2))
