import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_box_string(img, box, msg):
    """
    img: read by cv;
    box:[xmin, ymin, xmax, ymax];
    string: what you want to draw in img;
    return: img
    """
    x, y, x1, y1 = box

    cv2.rectangle(img, (x, y), (x1, y1), (48, 48, 255), 2)
    y -= 10
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simhei.ttf", 12, encoding="utf-8")
    p1_x, p1_y, p2_x, p2_y = font.getbbox(msg)
    chars_w = p2_x - p1_x
    chars_h = p2_y - p1_y
    coords = [(x, y), (x, y + chars_h),
              (x + chars_w, y + chars_h), (x + chars_w, y)]
    draw.polygon(coords, fill=(255, 24, 24))
    draw.text((x, y), msg, fill="white", font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


img = cv2.imread('test.jpg')
a = []
b = []
ps = [[450, 600], [780, 710]]

cv2.rectangle(img, ps[0], ps[1], (0, 255, 0), 3)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1080, 600)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyWindow('test')
