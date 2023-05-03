import numpy as np
import cv2
from typing import Union


def load_roi(path, W, H) -> np.ndarray:
    roi = np.loadtxt(path)
    roi[:, 0] *= W
    roi[:, 1] *= H
    roi = roi.reshape(-1, 1, 2)
    return roi


def load_homo(matches_path) -> np.ndarray:
    """matches_path: path to file containing matching points"""
    matches = np.int32(np.loadtxt(matches_path))
    src, dst = matches[:, :2], matches[:, 2:]       # type: ignore
    H, mask = cv2.findHomography(src, dst)      # cv2.RANSAC
    return H


def calc_loc(X, loc_infer_mode: int, mid: Union[tuple[Union[float, int]], None] = None) -> np.ndarray:
    
    # if detection mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
    # if detection mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
    # return [[x, y], ...]
    
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    X = X.copy()
    assert len(X.shape) == 2, 'Invalid shape, must be (N, M)'

    # TODO
    if loc_infer_mode == 1:     # box, midpoint of box's bottom edge
        X[:, 2] += X[:, 4] / 2
        X[:, 3] += X[:, 5]
        return X[:, 2:4]
    
    elif loc_infer_mode == 2:   # box, intersection of box's bottom edge with the segment from box's center to the midpoint image's bottom edge
        assert mid is not None
        X = X[:, 2:6]
        X[:, 2:4] += X[:, :2]
        ret = np.empty(shape=(len(X), 2), dtype='float32')
        mx, my = mid                                                    # type: ignore
        for i, xyxy in enumerate(X):
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2

            a = (xyxy[2] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and xyxy[1] <= y <= xyxy[3]:
                ret[i] = xyxy[2:4]
                continue

            a = (xyxy[3] - cy) / (my - cy)
            x = a * mx + (1 - a) * cx
            if 0 <= a <= 1 and xyxy[0] <= x <= xyxy[2]:
                ret[i] = [x, xyxy[3]]
                continue

            a = (xyxy[0] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and xyxy[1] <= y <= xyxy[3]:
                ret[i] = xyxy[[0, 3]]
        return ret
    
    raise NotImplementedError()


if __name__ == '__main__':
    print(load_homo(''))