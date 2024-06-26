import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt


# 计算iou函数
def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union


# 计算precision、 Recall、mAP函数
def compute_metrics(predictions, ground_truth, iou_threshold=0.5):
    all_precisions = []
    all_recalls = []
    all_ap = []

    # classes = list(set(ground_truth['class'].tolist()))
    classes = list(set([item.get('class') for item in ground_truth]))
    files = list(set([item.get('file') for item in ground_truth]))
    for file in files:
        ground_truth_temp = [item for item in ground_truth if
                             item.get('file', '') == file]  # ground_truth[ground_truth['file'] == file]
        predictions_temp = [item for item in predictions if
                            item.get('file', '') == file]  # predictions[predictions['file'] == file]

        for cls in classes:
            gt_cls = [item for item in ground_truth_temp if
                      item.get('class', '') == cls]  # ground_truth_temp[ground_truth_temp['class'] == cls]
            pred_cls = [item for item in predictions_temp if
                        item.get('class', '') == cls]  # predictions_temp[predictions_temp['class'] == cls]
            true_positives = []
            scores = []
            num_gt = len(gt_cls)
            for index, pred in enumerate(pred_cls):
                iou_max = 0
                match_index = -1
                detected = []
                for gt_index, gt in enumerate(gt_cls):
                    if gt_index not in detected:
                        # iou = calculate_iou(pred[['x_min', 'y_min', 'x_max', 'y_max']].tolist(),
                        #                     gt[['x_min', 'y_min', 'x_max', 'y_max']].tolist())
                        iou = calculate_iou([pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']],
                                            [gt['x_min'], gt['y_min'], gt['x_max'], gt['y_max']])
                        if iou > iou_max:
                            iou_max = iou
                            match_index = gt_index
                if iou_max >= iou_threshold:
                    true_positives.append(1)
                    detected.append(match_index)
                else:
                    true_positives.append(0)
                scores.append(pred['confidence'])
            if len(true_positives) > 0:
                precisions, recalls, _ = precision_recall_curve(true_positives, scores)
                ap = average_precision_score(true_positives, scores)

                all_precisions.append(precisions)
                all_recalls.append(recalls)
                all_ap.append(ap)

    mean_ap = np.mean(all_ap)

    return all_precisions, all_recalls, mean_ap


def read_data(path):
    images = os.listdir(path)
    labels = []
    for item in images:
        with open(f'{path}/{item}', 'r') as f:
            for line in f.readlines():
                ds = line.split(' ')
                temp = {
                    'file': f'{item}',
                    'class': ds[0],
                    'x_min': float(ds[1]),
                    'y_min': float(ds[2]),
                    'x_max': float(ds[3]),
                    'y_max': float(ds[4]),
                    'confidence': float(ds[5])
                }
                labels.append(temp)
    return labels


# 可视化分析
def plot_precision_recall(precisions, recalls, model_name):
    plt.figure(figsize=(10, 8))
    for i in range(len(precisions)):
        plt.plot(recalls[i], precisions[i], lw=2, label=f'Class {i + 1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for {model_name}')
    plt.legend(labels=['best'])
    plt.show()


if __name__ == '__main__':
    model_v7_labels = read_data('datasets/v7_labels')
    model_v8_labels = read_data('datasets/v9c-640')
    model_v10_labels = read_data('datasets/v9c-960')

    precisions2, recalls2, mean_ap2 = compute_metrics(model_v8_labels, model_v7_labels)
    precisions3, recalls3, mean_ap3 = compute_metrics(model_v10_labels, model_v7_labels)

    print(f'Model 1 - mAP: {mean_ap2}')
    print(f'Model 2 - mAP: {mean_ap3}')
    t = {}
    print(t.get('t1',[]))
    # plot_precision_recall(precisions1, recalls1, 'v8')
    # plot_precision_recall(precisions2, recalls2, 'v9')
    # plot_precision_recall(precisions3, recalls3, 'v10')
