import os

import cv2

from utils.recongnized_color import identify_color

path = './datasets'


def out_color_of_image(path):
    images = os.listdir(path)
    for item in images:
        bk_img = cv2.imread(f'{path}/{item}')
        txt = identify_color.get_color(bk_img)
        h, w, c = bk_img.shape
        # 在图片上添加文字信息
        cv2.putText(bk_img, txt['mainColor'], (int(h / 2), int(w / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # 显示图片
        cv2.imshow(txt['mainColor'], bk_img)
        cv2.waitKey(0)


# 获取颜色
def check_color(path):
    images = os.listdir(path)
    for item in images:
        bk_img = cv2.imread(f'{path}/{item}')
        name = item.split('.')[0]
        identify_color.deal_color(bk_img, name)


# 截取图片
def cut_image(path):
    images = os.listdir(path)
    for item in images:
        bk_img = cv2.imread(f'{path}/{item}')
        result = identify_color.cut_rect_image(bk_img)
        cv2.imwrite(f'./runs/{item}', result)


if __name__ == '__main__':
    cut_image(path)
