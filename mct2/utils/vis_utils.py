import numpy as np
import cv2

COLORS = [(121, 0, 148), (162, 206, 69), (253, 126, 251), (210, 116, 25), (20, 249, 29), (0, 255, 255), (0, 127, 255), (0, 0, 255)]

def plot_box(img, boxes):
    """
    boxes: [[frame (optional), id (optional), x1, y1, x2, y2],...]
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.squeeze(np.array(boxes))

    assert len(boxes.shape) == 2 and 4 <= boxes.shape[1] <= 6, "Invalid 'boxes' shape"

    img = img.copy()
    boxes = np.int32(boxes)
    coords = boxes[:, -4:]
    # coords[:, 0] = np.clip(coords[:, 0], 0, img.shape[1])
    # coords[:, 1] = np.clip(coords[:, 1], 0, img.shape[0])
    # coords[:, 2] = np.clip(coords[:, 2], 0, img.shape[1])
    # coords[:, 3] = np.clip(coords[:, 3], 0, img.shape[0])
    ids = boxes[:, -5] if boxes.shape[1] >= 5 else None

    # TODO parallel
    for i in range(boxes.shape[0]):
        if ids is None:
            color = COLORS[3]
        else:
            id = ids[i].item()
            color = COLORS[id % len(COLORS)]

            cv2.putText(img, str(id), coords[i, :2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)

        cv2.rectangle(img, coords[i, :2], coords[i, 2:], color, 2)

    return img

