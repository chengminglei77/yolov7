import os
import cv2

cv2.namedWindow('show', cv2.WINDOW_NORMAL)
cv2.resizeWindow('show', 1080, 640)


def main():
    # 总的检测根目录
    path_root_labels = 'E:\\datasets\\zh_extra\\test\\labels'
    # 总的检测根目录
    path_root_imgs = 'E:\\datasets\\zh_extra\\test\\images'
    type_object = '.txt'

    for ii in os.walk(path_root_imgs):
        for j in ii[2]:
            type = j.split(".")[1]
            if type != 'jpg':
                continue
            path_img = os.path.join(path_root_imgs, j)
            print(path_img)
            label_name = j[:-4] + type_object
            path_label = os.path.join(path_root_labels, label_name)
            # print(path_label)
            f = open(path_label, 'r+', encoding='utf-8')
            if os.path.exists(path_label) == True:

                img = cv2.imread(path_img, cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
                w = img.shape[1]
                h = img.shape[0]
                new_lines = []
                img_tmp = img.copy()
                while True:
                    line = f.readline()
                    if line:
                        msg = line.split(" ")
                        # print(x_center,",",y_center,",",width,",",height)
                        x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                        y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                        x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                        y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                        print(x1, ",", y1, ",", x2, ",", y2)
                        cv2.rectangle(img_tmp, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    else:
                        break
            cv2.imshow("show", img_tmp)
            c = cv2.waitKey(0)


if __name__ == '__main__':
    main()
