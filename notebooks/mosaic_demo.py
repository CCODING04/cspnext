import glob
import os
import random

import cv2
import numpy as np

from cspnext.datasets import coco


def drawBox(image, labels, classes, colors):
    # height, width = image.shape[:2]
    height, width = 1, 1
    for label in labels:
        cls_idx, xc, yc, w, h = label.tolist()
        cls_idx = int(cls_idx)
        class_name = classes[cls_idx]
        color = colors[cls_idx]
        cv2.rectangle(
            image,
            (
                int(width * (xc - w / 2)),
                int(height * (yc - h / 2)),
            ),
            (
                int(width * (xc + w / 2)),
                int(height * (yc + h / 2)),
            ),
            color=color,
            thickness=2,
        )
    return image


def mosaic_gen(image_list, labels_list, scale=[640, 640]):
    # assert len(image_list) == 4
    mosaic_range = [scale[0] // 2, scale[1] // 2]
    mosaic_image = np.full(
        (2 * scale[1], 2 * scale[0], 3), 114, dtype=np.uint8
    )
    mosaic_labels = []
    xc, yc = [
        int(
            random.uniform(
                scale[0] - mosaic_range[0],
                scale[0] + mosaic_range[0],
            )
        ),
        int(
            random.uniform(
                scale[1] - mosaic_range[1],
                scale[1] + mosaic_range[1],
            )
        ),
    ]
    height, width = mosaic_image.shape[:2]
    # clockwise sequence
    for i, (image, labels) in enumerate(
        zip(image_list, labels_list)
    ):
        h, w = image.shape[:2]
        labels = np.array(labels)
        if i == 0:
            x1a, y1a, x2a, y2a = (
                max(0, xc - w),
                max(0, yc - h),
                xc,
                yc,
            )
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )
        elif i == 1:
            x1a, y1a, x2a, y2a = (
                xc,
                max(0, yc - h),
                min(width, xc + w),
                yc,
            )
            x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a
        elif i == 2:
            x1a, y1a, x2a, y2a = (
                xc,
                yc,
                min(width, xc + w),
                min(height, yc + h),
            )
            x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a
        elif i == 3:
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                yc,
                xc,
                min(height, yc + h),
            )
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                0,
                w,
                y2a - y1a,
            )
        mosaic_image[y1a:y2a, x1a:x2a] = image[
            y1b:y2b, x1b:x2b
        ]
        pad_x, pad_y = x1a - x1b, y1a - y1b
        if labels.shape[0] == 0:
            continue
        labels[:, 1::2] *= w
        labels[:, 2::2] *= h

        labels[:, 1] += pad_x
        labels[:, 2] += pad_y

        mosaic_labels.append(labels)
    mosaic_labels = np.concatenate(mosaic_labels, axis=0)
    mosaic_labels[:, 1::2] /= mosaic_image.shape[1]
    mosaic_labels[:, 2::2] /= mosaic_image.shape[0]
    # print(mosaic_labels.shape)
    # mosaic_labels[:, 1] = mosaic_labels[:, 1].clip(min=0, max=scale[1]*2)
    # mosaic_labels[:, 2] = mosaic_labels[:, 2].clip(min=0, max=scale[0]*2)
    # mosaic_labels = mosaic_labels[mosaic_labels[:, 1:].sum(axis=1) > 0]
    return mosaic_image, mosaic_labels


if __name__ == "__main__":
    imp_lst = glob.glob("assets/tiny_coco/images/*.jpg")
    scale = [480, 480]
    image_list = []
    labels_list = []
    for image_path in random.sample(imp_lst, k=4):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        image_list.append(image)
        label_path = image_path.replace(
            "images", "labels"
        ).replace(".jpg", ".txt")
        labels = [
            list(map(eval, x.strip().split(" ")))
            for x in open(label_path, "r").readlines()
        ]
        labels_list.append(labels)
    mosaic_image, mosaic_labels = mosaic_gen(
        image_list, labels_list, [640, 640]
    )
    height, width = mosaic_image.shape[:2]
    mosaic_labels[:, 1::2] *= width
    mosaic_labels[:, 2::2] *= height
    mosaic_image = drawBox(mosaic_image, mosaic_labels, coco.classes, coco.colors)
    cv2.imwrite('out.png', mosaic_image)
