"""
    提供线与线的关系判断
"""
import numpy as np

from numpy import linalg as LA


# 线对象
class Line(object):
    def __init__(self, point1, point2):
        self.startPoint = point1
        self.endPoint = point2


# 求向量ab和向量cd的叉乘
def xmult(point1, point2, point3, point4):
    vectorAx = point2[0] - point1[0]
    vectorAy = point2[1] - point1[1]
    vectorBx = point4[0] - point3[0]
    vectorBy = point4[1] - point3[1]
    return vectorAx * vectorBy - vectorAy * vectorBx


# 判断两直线是否相交
def intersectionLine(line: Line, line2: Line):
    return intersection(line.startPoint, line.endPoint, line2.startPoint, line2.endPoint)


# 两直线是否相交
def intersection(point1, point2, point3, point4):
    """

    :param point1: 直线1 开始点
    :param point2: 直线1 结束点
    :param point3: 直线2 开始点
    :param point4: 直线2 结束点
    :return:
    """
    # 快速排斥-如果两条线在x轴和y轴上的映射都没有相交的点，则这两条线必然不会相交
    if (max(point3[0], point4[0]) < min(point1[0], point2[0])
            or max(point1[0], point2[0]) < min(point3[0], point4[0])
            or max(point3[1], point4[1]) < min(point1[1], point2[1])
            or max(point1[1], point2[1]) < min(point3[1], point4[1])):
        return False
    # 以c为公共点，分别判断向量cd到向量ca与到向量cb的方向，记为xmult1和xmult2。
    # 若ab分布于cd两侧，xmult1 * xmult2应小于0。
    # 同理若cd分布于ab两侧，xmult3 * xmult4应小于0。
    xmult1 = xmult(point3, point4, point3, point1)
    xmult2 = xmult(point3, point4, point3, point2)
    xmult3 = xmult(point1, point2, point1, point3)
    xmult4 = xmult(point1, point2, point1, point4)
    return xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0


def line_vectorize(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return [a, b]


# 判断ponit3是在线[p1,p2]的顺时针方向还是逆时针方向
def judge_in(point1, point2, point3):
    # 以c为公共点，分别判断向量cd到向量ca与到向量cb的方向，记为xmult1和xmult2。
    # 若ab分布于cd两侧，xmult1 * xmult2应小于0。
    # 同理若cd分布于ab两侧，xmult3 * xmult4应小于0。
    xmult1 = xmult(point1, point2, point1, point3)
    return xmult1 < 0


def calcVectorAngele(point1, point2, point3, point4):
    u = np.array(line_vectorize(point1, point2))
    v = np.array(line_vectorize(point3, point4))
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
    if u[0] * v[1] - u[1] * v[0] < 0:
        return a
    else:
        return 360 - a


if __name__ == '__main__':
    p1 = [1, 2]
    p2 = [4, 8]
    p3 = [2, 2]
    p4 = [4, 9]
    print(intersection(p1, p2, p3, p4))
    print(calcVectorAngele(p1, p2, p3, p4))
    if judge_in(p1, p2, p3):
        print('顺时针')
    else:
        print('逆时针')
