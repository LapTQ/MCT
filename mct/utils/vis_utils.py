import numpy as np
import cv2

COLORS = [(51, 255, 221), (55, 250, 250), (255, 221, 21), (102, 255, 102), (83, 50, 250), (209, 240, 170), (83, 179, 36), (240, 120, 240), (51, 153, 204), (187, 125, 250), (51, 204, 255)]


def plot_box(img, boxes, thickness=2):
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
            space = 5
            expected_text_H = 15

            y_text = coords[i, 1] - space if coords[i, 1] - space - expected_text_H > 0 else coords[i, 1] + space + expected_text_H
            x_text = coords[i, 0] + space
            cv2.putText(img, str(ids[i]) + (f' ({int(confs[i] * 100)})' if confs[i] != -1 else ''), (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=thickness)
        cv2.rectangle(img, coords[i, :2], coords[i, 2:4], color=color, thickness=thickness)

    return img

def draw_track(img, track, color, radius=2, **kwargs):

    if not isinstance(track, np.ndarray):
        track = np.array(track)
    if len(track.shape) == 1:
        track = np.expand_dims(track, axis=0)
    assert len(track.shape) == 2 and track.shape[1] == 2, 'Invalid track dimension, must be (2,) or (N, 2)'

    img = img.copy()
    for i in range(len(track)):
        cv2.circle(img, np.int32(track[i]), radius=radius, color=color, thickness=-1)

        if i == len(track) - 1 and 'camid' in kwargs:
            cv2.putText(img, str(kwargs['camid']), (int(track[i, 0]), int(track[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)

        if i == 0:
            continue
        cv2.line(img, np.int32(track[i]), np.int32(track[i - 1]), color=color, thickness=2 * radius)

    return img



