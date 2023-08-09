import os

import cv2

from utils.recongnized_color import identify_color

path = './runs/detect/exp2/images/'


def out_color_of_image(path):
    images = os.listdir(path)
    for item in images:
        bk_img = cv2.imread(f'{path}/{item}')
        txt = identify_color.get_color(bk_img)
        h, w, c = bk_img.shape
        # 在图片上添加文字信息
        cv2.putText(bk_img, txt, (int(h / 2), int(w / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # 显示图片
        cv2.imshow(txt, bk_img)
        cv2.waitKey()


if __name__ == '__main__':
    out_color_of_image(path)
