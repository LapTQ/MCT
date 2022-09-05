import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy2xysr(boxes):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    assert boxes.shape[1] == 4, 'Invalid box dimension.'
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x = (boxes[:, 0] + boxes[:, 2]) / 2.
    y = (boxes[:, 1] + boxes[:, 3]) / 2.
    s = w * h
    r = w / np.float32(h)
    return np.squeeze(np.stack([x, y, s, r], axis=1))


def xysr2xyxy(boxes):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    assert boxes.shape[1] == 4, 'Invalid box dimension.'
    w = np.sqrt(boxes[:, 2] * boxes[:, 3])
    h = boxes[:, 2] / w
    x1 = boxes[:, 0] - w/2
    y1 = boxes[:, 1] - h/2
    x2 = x1 + w
    y2 = y1 + h
    return np.squeeze(np.stack([x1, y1, x2, y2], axis=1))


def iou_batch(boxes1, boxes2):
    """
    boxes1: [[x1, y1, x2, y2,...], ...]
    boxes2: [[x1, y1, x2, y2,...], ...]
    return matrix dim: len(boxes1) x len(boxes2)
    """
    boxes1 = np.expand_dims(boxes1, 1)
    boxes2 = np.expand_dims(boxes2, 0)

    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
              + (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) - wh)
    return (o)


def iou_associate(dets, preds, thresh):
    """
    dets: [[x1, y1, x2, y2,...], ...]
    preds: [[x1, y1, x2, y2,...], ...]
    return:
        - matched: [[dets_index, preds_index], ...]
        - unmatched_dets: [dets_index, ...]
        - unmatched_preds: [preds_index, ...]
    """
    if len(dets) == 0 or len(preds) == 0:
        return np.empty((0, 2)), np.arange(len(dets)), np.arange(len(preds))

    iou_matrix = iou_batch(dets, preds)    # dim: len(dets) x len(preds)

    a = np.int32(iou_matrix > thresh)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        m = a
    else:
        # TODO do thoi gian cua ham nay so voi lapjv
        d, p = linear_sum_assignment(-iou_matrix)
        b = np.zeros((len(dets), len(preds)), 'int32')
        b[(d, p)] = 1
        m = a * b

    return np.transpose(np.where(m)), np.where(1 - m.sum(1))[0], np.where(1 - m.sum(0))[0]
