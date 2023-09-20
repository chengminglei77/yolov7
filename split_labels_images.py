import os
import shutil


# 获取指定路劲下的文件名称
def get_paths(path: str, old_suffix=None, new_suffix=None):
    paths = os.listdir(path)
    if old_suffix is None or new_suffix is None:
        return os.listdir(path)
    return [item.replace(old_suffix, new_suffix) for item in paths]


# 移动没有标注的文件到新的目录下
def move_images_without_labels(old_img_path: str, old_label_path: str, new_path: str):
    if not os.path.exists(old_img_path) or not os.path.exists(old_label_path):
        raise ValueError("图片或者标注路径不存在,请确认后重试")
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    # 获取标注的图片名称
    image_with_labels = get_paths(old_label_path, old_suffix='txt', new_suffix='jpg')
    image_with_labels.extend(get_paths(old_label_path, old_suffix='txt', new_suffix='png'))
    image_with_labels.extend(get_paths(old_label_path, old_suffix='txt', new_suffix='JPG'))
    # 额外配置的label文件
    labels = []
    # 获取目录下的所有文件
    images = get_paths(old_img_path)
    for item in images:
        if item not in image_with_labels or item in labels:
            shutil.move(f'{old_img_path}/{item}', new_path)


def move_images_with_labels(label_path, image_path, to_path):
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    image_with_labels = get_paths(label_path, old_suffix='txt', new_suffix='jpg')
    image_with_labels.extend(get_paths(label_path, old_suffix='txt', new_suffix='png'))
    image_with_labels.extend(get_paths(label_path, old_suffix='txt', new_suffix='JPG'))
    # 获取目录下的所有文件
    images = get_paths(image_path)
    for item in image_with_labels:
        if item in images:
            shutil.move(f'{image_path}//{item}', to_path)


def clear_labels_without_images(old_img_path: str, old_label_path: str):
    if not os.path.exists(old_img_path) or not os.path.exists(old_label_path):
        raise ValueError("图片或者标注路径不存在,请确认后重试")
    # 获取标注的图片名称
    images = get_paths(old_img_path, old_suffix='jpg', new_suffix='txt')
    images.extend(get_paths(old_img_path, old_suffix='JPG', new_suffix='txt'))
    images.extend(get_paths(old_img_path, old_suffix='png', new_suffix='txt'))
    labels = get_paths(old_label_path)
    for item in labels:
        if item not in images:
            os.remove(f'{old_label_path}/{item}')


# 测试获取目录文件名称
def test_get_paths():
    paths = get_paths("D://project//python//datasets//safety_cap//train//labels", old_suffix='txt', new_suffix='jpg')
    print(paths)


def test_move_iamges_without_labels():
    move_images_without_labels(old_img_path="D://project//python//datasets//safety_cap//train//images",
                               old_label_path="D://project//python//datasets//safety_cap//train//labels",
                               new_path="D://project//python//datasets//safety_cap//train//test//")


def test_clear_labels_without_images(old_img_path="D://project//python//datasets//safety_cap//train//images",
                                     old_label_path="D://project//python//datasets//safety_cap//train//labels"):
    clear_labels_without_images(old_img_path=old_img_path, old_label_path=old_label_path)


if __name__ == '__main__':
    move_images_without_labels(old_img_path="D://data//datasets//train//images",
                                old_label_path="D://data//datasets//train//labels",
                               new_path="D://data//datasets//train//test")
