import cv2
import numpy as np


def vis(
    img,
    boxes,
    scores,
    cls_ids,
    colors,
    class_names,
    conf_thr=0.5,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, score, cls_id in zip(
        boxes, scores, cls_ids
    ):
        idx = cls_id.item()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        x0, y0, x1, y1 = box.numpy().astype('int').tolist()
        print([x0, y0, x1, y1], text)
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_color = (0, 0, 0) if np.mean(colors[cls_id]) < 0.5 else (255, 255, 255)
        txt_bk_color = (np.array(colors[cls_id]) * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), colors[cls_id], 2)
        cv2.rectangle(
            img,
            (x0, y0+1),
            (x0 + txt_size[0]+1, y0+int(1.5*txt_size[1])),
            color=txt_bk_color,
            thickness=-1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img
