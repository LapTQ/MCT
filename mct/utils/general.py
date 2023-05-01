import numpy as np
import cv2


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


def calc_loc(x, loc_infer_mode: int) -> np.ndarray:
    
    # if detection mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
    # if detection mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
    
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x = x.copy()
    assert len(x.shape) == 2, 'Invalid x shape, must be (N, M)'

    # TODO
    if loc_infer_mode == 1:     # box, midpoint of bottom edge
        x[:, 2] += x[:, 4] / 2
        x[:, 3] += x[:, 5]
        return x[:, 2:4]
    
    raise NotImplementedError()