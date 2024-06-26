# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import argparse
import json
import logging

logging.basicConfig(level=logging.DEBUG)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='train.json',
                        help='path of label json')
    parser.add_argument('--result_json', type=str,
                        default='v7_labels.json',
                        help='path of result json')
    parser.add_argument('--ann_type', type=str, default='bbox', help='type of evaluation')
    args = parser.parse_args()
    return args


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return x


def convert_to_coco_keypoints(json_file, cocoGt):
    keypoints_openpose_map = ["nose", "Neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
                              "left_elbow", "left_wrist",
                              "right_hip", "right_knee", "right_ankle"]
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        keypoints = res["keypoints"]
        if len(keypoints) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        person_num = int(len(keypoints) / (len(keypoints_openpose_map) * 3))
        # print(len(keypoints), len(keypoints_openpose_map), person_num)
        for i in range(person_num):
            data = dict()
            data['image_id'] = int(image_id)
            data['category_id'] = 1
            data['keypoints'] = []
            score_list = []
            for point_name in cocoGt.dataset["categories"][0]["keypoints"]:
                point_id = keypoints_openpose_map.index(point_name)
                x = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3]
                y = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 1]
                score = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 2]
                data['keypoints'].append(x)
                data['keypoints'].append(y)
                data['keypoints'].append(score)
                score_list.append(score)
            data['score'] = float(sum(score_list) / len(score_list) + 1.25 * max(score_list))
            temp_json.append(data)
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)


def convert_to_coco_bbox(json_file, cocoGt):
    temp_json = []
    coco91class = coco80_to_coco91_class()
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        bboxes = res["bboxes"]
        if len(bboxes) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break

        for i in range(len(bboxes)):
            try:
                data = dict()
                data['image_id'] = int(image_id)
                data['category_id'] = coco91class[bboxes[i]['category_id']]
                data['bbox'] = bboxes[i]['bbox']
                data['score'] = bboxes[i]['score']
                temp_json.append(data)
            except:
                continue

    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)


def main(args):
    cocoGt = COCO(args.gt_path)

    if args.ann_type == 'keypoints':
        convert_to_coco_keypoints(args.result_json, cocoGt)
    if args.ann_type == 'bbox':
        convert_to_coco_bbox(args.result_json, cocoGt)

    cocoDt = cocoGt.loadRes('temp.json')
    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    iou_thrs = [0.45, 0.5]  # 设置要评测的IoU阈值
    cocoEval.params.iouThrs = iou_thrs

    # imgIds = sorted(cocoGt.getImgIds())
    # cocoEval.params.imgIds = imgIds
    for catId in cocoGt.getCatIds():  # 遍历每一个类别
        cocoEval.params.catIds = [catId]  # 设置要评测的类别为当前类别
        cocoEval.evaluate()  # 开始评估
        cocoEval.accumulate()  # 累积评估结果
        print(f"Category ID: {catId}")
        cocoEval.summarize()  # 输出当前类别的评估结果摘要

    print('-------')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = argsparser()
    main(args)
