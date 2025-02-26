import numpy as np
import collections


# 定义字典存放颜色分量上下限
# 例如：{颜色: [min分量, max分量]}
# {'red': [array([160,  43,  46]), array([179, 255, 255])]}

def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 38])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
    # 灰黑色
    lower_gray = np.array([0, 35, 0])
    upper_gray = np.array([180, 50, 75])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['black3'] = color_list
    # 添加蓝黑色
    lower_gray = np.array([96, 35, 0])
    upper_gray = np.array([130, 255, 120])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['black2'] = color_list
    # 添加红、橙色的黑色区域
    lower_black = np.array([0, 55, 38])
    upper_black = np.array([25, 255, 80])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black1'] = color_list

    # 灰色
    lower_gray = np.array([0, 35, 75])
    upper_gray = np.array([180, 50, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray'] = color_list
    # 添加蓝色的灰色区域
    lower_gray = np.array([98, 0, 100])
    upper_gray = np.array([130, 43, 190])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray1'] = color_list

    # 白色
    lower_white = np.array([0, 0, 205])
    upper_white = np.array([180, 68, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 蓝白色
    lower_blue = np.array([96, 0, 190])
    upper_blue = np.array([130, 43, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['white1'] = color_list

    # 红色
    lower_red = np.array([156, 55, 38])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 55, 85])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 55, 85])
    upper_orange = np.array([25, 255, 200])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([20, 30, 38])
    upper_yellow = np.array([40, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 38, 38])
    upper_green = np.array([78, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 38, 38])
    upper_cyan = np.array([96, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([96, 43, 45])
    upper_blue = np.array([130, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([126, 38, 38])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


if __name__ == '__main__':
    color_dict = getColorList()
    print(color_dict)

    num = len(color_dict)
    print('num=', num)

    for d in color_dict:
        print('key=', d)
        print('value=', color_dict[d][1])
