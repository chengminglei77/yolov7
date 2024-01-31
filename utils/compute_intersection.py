import cv2
import numpy as np


def GetPolygon(point1, point2):
    Polygon1 = np.array(point1).astype(int)
    Polygon2 = np.array(point2).astype(int)
    ImH = 1440
    ImW = 2560
    return ImH, ImW, Polygon1, Polygon2


def DrawPolygon(ImShape, Polygon, Color):
    Im = np.zeros(ImShape, np.uint8)
    try:
        cv2.fillPoly(Im, Polygon, Color)  # 只使用这个函数可能会出错，不知道为啥
    except:
        try:
            cv2.fillConvexPoly(Im, Polygon, Color)
        except:
            print('cant fill\n')

    return Im


def Get2PolygonIntersectArea(ImShape, Polygon1, Polygon2):
    Im1 = DrawPolygon(ImShape[:-1], Polygon1, 122)  # 多边形1区域填充为122
    Im2 = DrawPolygon(ImShape[:-1], Polygon2, 133)  # 多边形2区域填充为133
    Im = Im1 + Im2
    ret, OverlapIm = cv2.threshold(Im, 200, 255, cv2.THRESH_BINARY)  # 根据上面的填充值，因此新图像中的像素值为255就为重叠地方
    ret1, OverlapIm1 = cv2.threshold(Im1, 120, 133, cv2.THRESH_BINARY)  # 根据上面的填充值，133
    ret2, OverlapIm2 = cv2.threshold(Im2, 120, 133, cv2.THRESH_BINARY)  # 根据上面的填充值，133

    IntersectArea = np.sum(np.greater(OverlapIm, 0))  # 求取两个多边形交叠区域面积
    IntersectArea1 = np.sum(np.greater(OverlapIm1, 0))  # 求取两个多边形交叠区域面积
    IntersectArea2 = np.sum(np.greater(OverlapIm2, 0))  # 求取两个多边形交叠区域面积

    # # 下面使用opencv自带的函数求取一下，最为对比
    # contours, hierarchy = cv2.findContours(OverlapIm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contourArea = cv2.contourArea(contours[0])
    # print('contourArea={}\n'.format(contourArea))
    # perimeter = cv2.arcLength(contours[0], True)
    # print('contourPerimeter={}\n'.format(perimeter))
    # RealContourArea = contourArea + perimeter
    # print('RealContourArea={}\n'.format(RealContourArea))
    return IntersectArea, OverlapIm, IntersectArea / IntersectArea2


# 多维度校验:目标和目标
def expand_check_in_area(point1, point2, height, weight, max_rate=1, min_rate=0.05):
    point1 = np.array(point1).astype(int)
    point2 = np.array(point2).astype(int)
    shape = (height, weight, 3)
    inter_area, over_area, rate = Get2PolygonIntersectArea(shape, point1, point2)
    if rate is None:
        return False, rate
    elif max_rate >= rate >= min_rate:
        return True, rate
    else:
        return False, rate


def check_in_area(point1, point2, image_h, image_w, max_rate=1, min_rate=0.05):
    p2 = [point2[0], [point2[0][0], point2[1][1]], point2[1], [point2[1][0], point2[0][1]]]
    p1 = [point1[0], [point1[0][0], point1[1][1]], point1[1], [point1[1][0], point1[0][1]]]
    flag, _ = expand_check_in_area(p1, p2, image_h, image_w, max_rate, min_rate)
    return flag


# 多维度校验:绘制区域和目标
def check_iou(points, point, weight=1920, height=1680, max_rate=1, min_rate=0.1):
    p2 = [point[0], [point[0][0], point[1][1]], point[1], [point[1][0], point[0][1]]]
    # 获取目标上下位置最大值和最小值
    o_x_max, _ = max((row[0], i) for i, row in enumerate(p2))
    o_x_min, _ = min((row[0], i) for i, row in enumerate(p2))
    for item in points:
        area_max, _ = max((row[0], i) for i, row in enumerate(item))
        area_min, _ = min((row[0], i) for i, row in enumerate(item))
        if check_in_area(item, p2, image_h=height, image_w=weight, max_rate=max_rate, min_rate=min_rate):
            return True
    return False


# 检查是否有交集且交集小于100,大于10
def check_intersection(points, point, weight=1920, height=1680, max_rate=1, min_rate=0.05):
    p2 = [point[0], [point[0][0], point[1][1]], point[1], [point[1][0], point[0][1]]]
    for item in points:
        p1 = [item[0], [item[0][0], item[1][1]], item[1], [item[1][0], item[0][1]]]
        if check_in_area(p1, p2, image_h=height, image_w=weight, max_rate=1, min_rate=0.05):
            return True
    return False


if __name__ == '__main__':
    points1 = [[50, 50], [50, 200], [200, 200], [200, 50]]
    # [[1103，600]，[1103，1282]，[1465，1282]，[1465，600]]
    points2 = [[20, 45], [45, 100], [55, 100], [55, 20]]

    ImH, ImW, Polygon1, Polygon2 = GetPolygon(points1, points2)
    ImShape = (ImH, ImW, 3)
    Im1 = DrawPolygon(ImShape, Polygon1, (255, 0, 0))
    Im2 = DrawPolygon(ImShape, Polygon2, (0, 255, 0))
    cv2.imshow('ColorPolygons', Im1 + Im2)

    IntersectArea, OverlapIm, rate = Get2PolygonIntersectArea(ImShape, Polygon1, Polygon2)
    print('rate={}\n'.format(rate))
    print('IntersectArea={}\n'.format(IntersectArea))
    cv2.imshow('OverlapIm', OverlapIm)
    cv2.waitKey(0)
    print(check_in_area(points1, points2, image_h=ImH, image_w=ImW, min_rate=0.45))
