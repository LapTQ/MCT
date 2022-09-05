import numpy as np
import cv2

COLORS = [(121, 0, 148), (162, 206, 69), (253, 126, 251), (210, 116, 25), (20, 249, 29), (0, 255, 255), (0, 127, 255), (0, 0, 255)]

def plot_box(img, boxes):
    """
    boxes: [[frame, id, x1, y1, x2, y2, conf],...]
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.squeeze(np.array(boxes))

    assert len(boxes.shape) == 2 and boxes.shape[1] == 7, "Invalid 'boxes' shape"

    img = img.copy()
    ids = np.int32(boxes[:, 1])
    coords = np.int32(boxes[:, 2:6])
    confs = boxes[:, 6]
    # coords[:, 0] = np.clip(coords[:, 0], 0, img.shape[1])
    # coords[:, 1] = np.clip(coords[:, 1], 0, img.shape[0])
    # coords[:, 2] = np.clip(coords[:, 2], 0, img.shape[1])
    # coords[:, 3] = np.clip(coords[:, 3], 0, img.shape[0])

    # TODO parallel
    for i in range(boxes.shape[0]):
        if ids[i] == -1:
            color = COLORS[3]
        else:
            color = COLORS[ids[i] % len(COLORS)]

            cv2.putText(img, str(ids[i]) + (f' ({int(confs[i] * 100)})' if confs[i] != -1 else ''), coords[i, :2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)

        cv2.rectangle(img, coords[i, :2], coords[i, 2:4], color, 2)

    return img

