import cv2
import numpy as np
from pathlib import Path
from vis_utils import COLORS
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from ortools.linear_solver import pywraplp
from img_utils import iou_batch
from map_utils import hungarian, map_timestamp
import time

import sys
sys.path.append(sys.path[0] + '/../..')

HERE = Path(__file__).parent


video_version_to_db_name = {
    '2d_v1': '20221124143517_sct',  # 20221118235019_sct (ban dau), 20221124143517_sct (sau khi bo non-object, refine cam 1)
    '2d_v2': '20230123154857_sct', # 20221208170629_sct (gt, gan bay tay), 20230123154857_sct (gt, cai thien them tu 20221208170629_sct)
}

def get_box_repr(boxes, kind, **kwargs):
    """x1, y1, x2, y2. Return shape (N, 2)"""
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    assert boxes.shape[-1] == 4, 'Invalid boxes dimension, must be (4,) or (N, 4)'

    if kind == 'center':
        return np.concatenate([(boxes[:, 0:1] + boxes[:, 2:3]) / 2, (boxes[:, 1:2] + boxes[:, 3:4]) / 2], axis=1)
    elif kind == 'bottom':
        return np.concatenate([(boxes[:, 0:1] + boxes[:, 2:3]) / 2, boxes[:, 3:4]], axis=1)
    elif kind == 'foot':
        assert 'midpoint' in kwargs, 'midpoint must be specified for kind=="foot"'
        assert len(kwargs['midpoint']) == 2, 'Invalid midpoint dimension, must be (2,)'
        ret = np.empty(shape=(boxes.shape[0], 2), dtype='float32')
        mx, my = kwargs['midpoint']
        for i in range(len(boxes)):
            cx, cy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2

            a = (boxes[i, 2] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and boxes[i, 1] <= y <= boxes[i, 3]:
                ret[i] = [boxes[i, 2], boxes[i, 3]]
                continue

            a = (boxes[i, 3] - cy) / (my - cy)
            x = a * mx + (1 - a) * cx
            if 0 <= a <= 1 and boxes[i, 0] <= x <= boxes[i, 2]:
                ret[i] = [x, boxes[i, 3]]
                continue

            a = (boxes[i, 0] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and boxes[i, 1] <= y <= boxes[i, 3]:
                ret[i] = [boxes[i, 0], boxes[i, 3]]
        return ret

    else:
        raise ValueError('Invalid value for kind')


def trajectory_distance(traj1, traj2, kind):
    if not isinstance(traj1, np.ndarray):
        traj1 = np.array(traj1)
    if not isinstance(traj2, np.ndarray):
        traj2 = np.array(traj2)
    if len(traj1.shape) == 1:
        traj1 = np.expand_dims(traj1, axis=0)
    if len(traj2.shape) == 1:
        traj2 = np.expand_dims(traj2, axis=0)
    assert len(traj1.shape) == len(traj2.shape) == 2 and traj1.shape[1] == traj2.shape[
        1] == 2, f'Invalid trajectory dimension, must be (2,) or (N, 2), got {traj1.shape} and {traj2.shape}'
    assert len(traj1) == len(traj2), '2 trajectories must of the same length'

    if kind == 'euclid':
        return np.mean(
            np.sqrt(np.sum(
                np.square(traj1 - traj2),
                axis=1,
                keepdims=False))
        )
    # TODO Hausdorff, DTW

    else:
        raise ValueError('Invalid value for kind')


def get_homo(src, dst, video_version):

    # TODO any
    matches = np.loadtxt(str(HERE / f'../../data/recordings/{video_version}/matches_{src}_to_{dst}.txt'), dtype='int32')
    src_pts, dst_pts = matches[:, :2], matches[:, 2:]
    H, mask = cv2.findHomography(src_pts, dst_pts) # cv2.RANSAC
    return H


def get_roi(dst, video_version):

    # TODO any
    roi = np.loadtxt(str(HERE / f'../../data/recordings/{video_version}/roi_{dst}.txt'), dtype='float32')
    H, W = cv2.VideoCapture(str(list(Path(HERE / f'../../data/recordings/{video_version}/videos').glob(f'{dst}*.avi'))[0])).read()[1].shape[:2]
    roi[:, 0] *= W
    roi[:, 1] *= H
    roi = roi.reshape(-1, 1, 2).astype('int32')

    return roi


def input_sct_from(txt_path, delimeter, midpoint, box_repr_kind, **kwargs):

    seq = np.loadtxt(txt_path, delimiter=delimeter)

    ls_id = np.int32(np.unique(seq[:, 1]))

    N = np.max(ls_id) + 1           # this will work for both track_id starts from 0 or 1
    T = np.int32(np.max(seq[:, 0]) + 1)     # this will work for both frame_id starts from 0 or 1

    # mark temporal visibility
    OT = np.zeros((N, T), dtype='int32')

    # spatio position
    OX = np.full((N, T), -1, dtype='float32')  # cannot use np.empty due to nan value in later computation.
    OY = np.full((N, T), -1, dtype='float32')
    OX_no_trans = np.full((N, T), -1, dtype='float32')
    OY_no_trans = np.full((N, T), -1, dtype='float32')
    OS = np.full((N, T, 8), -1, dtype='float32')
    OS_no_trans = np.full((N, T, 8), -1, dtype='float32')

    for i, det in enumerate(seq):
        frame = np.int32(det[0])
        id = np.int32(det[1])

        det[4:6] += det[2:4]
        [[repr_x, repr_y]] = get_box_repr(det[2:6], kind=box_repr_kind, midpoint=midpoint)
        x1, y1, x2, y2, x3, y3, x4, y4 = det[2], det[3], det[2], det[5], det[4], det[5], det[4], det[3]

        OX_no_trans[id, frame] = repr_x
        OY_no_trans[id, frame] = repr_y
        OS_no_trans[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

        if 'homo' in kwargs:
            [[[repr_x, repr_y]],
             [[x1, y1]],
             [[x2, y2]],
             [[x3, y3]],
             [[x4, y4]]] = cv2.perspectiveTransform(
                np.array([[[repr_x, repr_y]],
                          [[x1, y1]],
                          [[x2, y2]],
                          [[x3, y3]],
                          [[x4, y4]]
                          ]),
                kwargs['homo'])

        OX[id, frame] = repr_x
        OY[id, frame] = repr_y
        OS[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

        if 'roi' in kwargs and not cv2.pointPolygonTest(kwargs['roi'], (repr_x, repr_y), True) >= -5:
            continue

        OT[id, frame] = 1


    return OT, OX, OY, OS, OX_no_trans, OY_no_trans, OS_no_trans


def input_mct_from(txt_path, delimeter, fps, midpoint, box_repr_kind, **kwargs):

    _, _, *record_time = os.path.splitext(os.path.split(txt_path)[-1])[0].split('_')
    record_time = datetime.strptime('_'.join(record_time), '%Y-%m-%d_%H-%M-%S-%f')

    seq = np.loadtxt(txt_path, delimiter=delimeter)

    ls_id = np.int32(np.unique(seq[:, 1]))

    N = np.max(ls_id) + 1   # this will work for both track_id starts from 0 or 1
    T = np.int32(np.max(seq[:, 0]) + 1)  # this will work for both frame_id starts from 0 or 1

    # mark temporal visibility
    OT = np.zeros((N, T), dtype='int32')
    OTT = np.zeros((T,), dtype='float64')    # timestamp

    # spatio position anskfsufhasufhaisuhfoaufhud
    OX = np.full((N, T), -1, dtype='float32')  # cannot use np.empty due to nan value in later computation.
    OY = np.full((N, T), -1, dtype='float32')
    OS = np.full((N, T, 8), -1, dtype='float32')
    OX_no_trans = np.full((N, T), -1, dtype='float32')
    OY_no_trans = np.full((N, T), -1, dtype='float32')
    OS_no_trans = np.full((N, T, 8), -1, dtype='float32')

    for i, det in enumerate(seq):
        frame = np.int32(det[0])
        id = np.int32(det[1])

        det[4:6] += det[2:4]
        [[repr_x, repr_y]] = get_box_repr(det[2:6], kind=box_repr_kind, midpoint=midpoint)
        x1, y1, x2, y2, x3, y3, x4, y4 = det[2], det[3], det[2], det[5], det[4], det[5], det[4], det[3]

        OX_no_trans[id, frame] = repr_x
        OY_no_trans[id, frame] = repr_y
        OS_no_trans[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

        if 'homo' in kwargs:
            [[[repr_x, repr_y]],
             [[x1, y1]],
             [[x2, y2]],
             [[x3, y3]],
             [[x4, y4]]] = cv2.perspectiveTransform(
                np.array([[[repr_x, repr_y]],
                          [[x1, y1]],
                          [[x2, y2]],
                          [[x3, y3]],
                          [[x4, y4]]
                          ]),
                kwargs['homo'])

        OX[id, frame] = repr_x
        OY[id, frame] = repr_y
        OS[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

        if 'roi' in kwargs and not cv2.pointPolygonTest(kwargs['roi'], (repr_x, repr_y), True) >= -5:
            continue

        OT[id, frame] = 1

    for i in range(T):
        OTT[i] = (record_time + timedelta(seconds=(i - 1) / fps)).timestamp() # frame id starts from 1

    return OT, OX, OY, OTT, OS, OX_no_trans, OY_no_trans, OS_no_trans


class IQRFilter:
    def __init__(self, q1=25, q2=75):
        self.q1 = q1
        self.q2 = q2

    def run(self, distances):
        print('[DEBUG] Calculating upper bound for distance using IQR')
        p1, p2 = np.percentile(distances, [self.q1, self.q2])
        iqr = p2 - p1
        upper_bound = p2 + 1.5 * iqr
        print('[DEBUG] Upper bound =', upper_bound)

        # filter out false matches due to missing detection boxes
        import matplotlib.pyplot as plt
        plt.hist(distances.flatten(), bins=42)
        plt.plot([upper_bound, upper_bound], plt.ylim())
        plt.show()

        return upper_bound


class GMMFilter:
    def __init__(self, n_components, std_coef=3):
        self.n_components = n_components
        self.std_coef = std_coef

    def run(self, distances):
        print('[DEBUG] Calculating upper bound for distance using GMM')
        np.random.seed(42)
        from sklearn.mixture import GaussianMixture
        gmm_error_handled = False
        reg_covar = 1e-6
        while not gmm_error_handled:
            try:
                print(f'[DEBUG] Trying GMM reg_covar = {reg_covar}')
                gm = GaussianMixture(n_components=self.n_components, covariance_type='diag', reg_covar=reg_covar).fit(distances)
                gmm_error_handled = True
            except:
                print(f'[DEBUG] Failed!')
                reg_covar *= 10
        smaller_component = np.argmin(gm.means_)
        upper_bound = gm.means_[smaller_component] + self.std_coef * np.sqrt(gm.covariances_[smaller_component])
        print(f'[DEBUG] smaller component has mean = {min(gm.means_)} and std = {np.sqrt(gm.covariances_[smaller_component])}')
        print('[DEBUG] Upper bound =', upper_bound)

        # filter out false matches due to missing detection boxes
        import matplotlib.pyplot as plt
        plt.hist(distances.flatten(), bins=42)
        plt.plot([upper_bound, upper_bound], plt.ylim())
        plt.show()

        return upper_bound


def make_true_sct_gttracker_correspondences_v2(
        gt_txt_path,
        tracker_txt_path,
        midpoint,
        filter,
        box_repr_kind,
        **kwargs):
    # options for kwargs: homo, roi, use_iou

    # CONCERN ON OVERLAPPING REGION ONLY
    OT, OX, OY, OS, OX_no_trans, OY_no_trans, OS_no_trans = input_sct_from(gt_txt_path, delimeter=',', midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)
    HT, HX, HY, HS, HX_no_trans, HY_no_trans, HS_no_trans = input_sct_from(tracker_txt_path, delimeter=None, midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)

    N = OT.shape[0]
    M = HT.shape[0]
    T = max(OT.shape[1], HT.shape[1])

    # pad 0 to equalize temporal dimension
    OT = np.pad(OT, ((0, 0), (0, T - OT.shape[1])), mode='constant', constant_values=0)
    OX = np.pad(OX, ((0, 0), (0, T - OX.shape[1])), mode='constant', constant_values=-1)
    OY = np.pad(OY, ((0, 0), (0, T - OY.shape[1])), mode='constant', constant_values=-1)
    HT = np.pad(HT, ((0, 0), (0, T - HT.shape[1])), mode='constant', constant_values=0)
    HX = np.pad(HX, ((0, 0), (0, T - HX.shape[1])), mode='constant', constant_values=-1)
    HY = np.pad(HY, ((0, 0), (0, T - HY.shape[1])), mode='constant', constant_values=-1)

    X = np.zeros((N, M, T), dtype='int32')
    distances = np.empty((N, M, T), dtype='float32')

    for t in range(T):

        O_present_list = np.where(OT[:, t])[0]
        H_present_list = np.where(HT[:, t])[0]

        cost = np.empty((len(O_present_list), len(H_present_list)), dtype='float32')
        for i, o in enumerate(O_present_list):
            for j, h in enumerate(H_present_list):
                if 'use_iou' not in kwargs or not kwargs['use_iou']:
                    cost[i, j] = np.sqrt((OX_no_trans[o, t] - HX_no_trans[h, t])**2 + (OY_no_trans[o, t] - HY_no_trans[h, t])**2)
                else:
                    cost[i, j] = - iou_batch([OS_no_trans[o, t, [0, 1, 4, 5]]], [HS_no_trans[h, t, [0, 1, 4, 5]]])

        i_matched_list, j_matched_list = hungarian(cost)
        for i, j in zip(i_matched_list, j_matched_list):
            o = O_present_list[i]
            h = H_present_list[j]
            X[o, h, t] = 1
            distances[o, h, t] = cost[i, j]

    if filter is not None:
        boundary = filter(distances[X == 1].reshape(-1, 1))
        X = np.where(distances > boundary, 0, X)

    print('[DEBUG] Number of matches:', np.sum(X))

    return OT, HT, OX, HX, OY, HY, OS, HS, OX_no_trans, HX_no_trans, OY_no_trans, HY_no_trans, OS_no_trans, HS_no_trans, np.eye(T, T, dtype='int32'), X, X


def make_true_mct_trackertracker_correspondences_v2(
        cam1_tracker_path, cam2_tracker_path,
        fps1, fps2,
        midpoint1, midpoint2,
        mct_gtgt_correspondences,
        cam1_sct_gttracker_correspondences,
        cam2_sct_gttracker_correspondences,
        homo,
        roi,
        box_repr_kind
):

    C1T, C1X, C1Y, C1TT, C1S, C1X_no_trans, C1Y_no_trans, C1S_no_trans = input_mct_from(cam1_tracker_path, delimeter=None, fps=fps1, midpoint=midpoint1, box_repr_kind=box_repr_kind, homo=homo, roi=roi)
    C2T, C2X, C2Y, C2TT, C2S, C2X_no_trans, C2Y_no_trans, C2S_no_trans = input_mct_from(cam2_tracker_path, delimeter=None, fps=fps2, midpoint=midpoint2, box_repr_kind=box_repr_kind, roi=roi)

    N1, T1 = C1T.shape
    N2, T2 = C2T.shape

    time_correspondences = map_timestamp(C1TT, C2TT, diff_thresh=1)

    X = np.zeros((N1, N2, T1), dtype='int32')

    for t1 in range(T1):
        if not np.any(time_correspondences[t1]):
            continue

        t2 = np.where(time_correspondences[t1])[0].item()

        for h1 in range(N1):
            for h2 in range(N2):
                if not (np.any(cam1_sct_gttracker_correspondences[:, h1, t1]) and
                        np.any(cam2_sct_gttracker_correspondences[:, h2, t2])
                ):
                    continue

                o1 = np.where(cam1_sct_gttracker_correspondences[:, h1, t1])[0].item()
                o2 = np.where(cam2_sct_gttracker_correspondences[:, h2, t2])[0].item()

                if (o1, o2) in mct_gtgt_correspondences:
                    X[h1, h2, t1] = 1

    return C1T, C2T, C1X, C2X, C1Y, C2Y, C1S, C2S, C1X_no_trans, C2X_no_trans, C1Y_no_trans, C2Y_no_trans, C1S_no_trans, C2S_no_trans, time_correspondences, X, X


def mct_mapping(
        cam1_tracker_path, cam2_tracker_path,
        fps1, fps2,
        midpoint1, midpoint2,
        homo,
        roi,
        filter,
        max_filter_iters,
        box_repr_kind,
        window_size=1,
        window_boundary=0,
        **kwargs
):
    """
    n_gmm_components can be None or 1, 2, 3,... In this function, it should be 2.
    window_size must be odd
    window_boundary limit sampling window_size frames in each direction
    """
    assert window_size % 2 == 1, 'window_size must be an odd number'

    start_preprocess = time.time()
    C1T, C1X, C1Y, C1TT, C1S, C1X_no_trans, C1Y_no_trans, C1S_no_trans = input_mct_from(cam1_tracker_path, delimeter=None, fps=fps1, midpoint=midpoint1, box_repr_kind=box_repr_kind, homo=homo, roi=roi)
    C2T, C2X, C2Y, C2TT, C2S, C2X_no_trans, C2Y_no_trans, C2S_no_trans = input_mct_from(cam2_tracker_path, delimeter=None, fps=fps2, midpoint=midpoint2, box_repr_kind=box_repr_kind, roi=roi)
    end_preprocess = time.time()

    print(f'[INFO] Preprocess time: {end_preprocess - start_preprocess:.2f}s')

    N1, T1 = C1T.shape
    N2, T2 = C2T.shape

    start_map_timestamp = time.time()
    time_correspondences = map_timestamp(C1TT, C2TT, diff_thresh=2)
    end_map_timestamp = time.time()

    print(f'[INFO] Mapping timestamp time: {end_map_timestamp - start_map_timestamp:.2f}s')

    map_frame_time = 0
    X_prev = np.zeros((N1, N2, T1), dtype='int32')  # T1 or T2 is either the same =)
    gate_prev = np.ones((N1, N2, T1), dtype='int32')
    for filter_iter in range(max_filter_iters):
        X = np.zeros((N1, N2, T1), dtype='int32')
        distances = np.empty((N1, N2, T1), dtype='float32')
        for t1 in range(T1):     # because I chose T1 as a dim of X
            start_map_frame = time.time()
            if not np.any(time_correspondences[t1]):
                continue

            t2 = np.where(time_correspondences[t1])[0].item()

            H1_present_list = np.where(C1T[:, t1])[0]
            H2_present_list = np.where(C2T[:, t2])[0]

            cost = np.empty((len(H1_present_list), len(H2_present_list)), dtype='float32')
            sub_gate = np.empty((len(H1_present_list), len(H2_present_list)), dtype='int32')
            for i1, h1 in enumerate(H1_present_list):
                for i2, h2 in enumerate(H2_present_list):
                    # sample up to window_size frames that h1, h2 co-occurs, but not exceed window_boundary in each direction
                    left_counter = 0
                    right_counter = 0
                    window_h1_loc = [[C1X[h1, t1], C1Y[h1, t1]]]
                    window_h2_loc = [[C2X[h2, t2], C2Y[h2, t2]]]
                    for i in range(1, window_boundary + 1):
                        if left_counter + 1 > (window_size - 1) / 2:
                            break

                        temp_t1 = t1 - i
                        if temp_t1 < 0 or not np.any(time_correspondences[temp_t1]):
                            continue
                        temp_t2 = np.where(time_correspondences[temp_t1])[0].item()

                        if not C1T[h1, temp_t1] or not C2T[h2, temp_t2]:
                            continue

                        left_counter += 1
                        window_h1_loc.append([C1X[h1, temp_t1], C1Y[h1, temp_t1]])
                        window_h2_loc.append([C2X[h2, temp_t2], C2Y[h2, temp_t2]])

                    for i in range(1, window_boundary + 1):
                        if right_counter + 1 > (window_size - 1) / 2:
                            break

                        temp_t1 = t1 + i
                        if temp_t1 >= T1 or not np.any(time_correspondences[temp_t1]):
                            continue
                        temp_t2 = np.where(time_correspondences[temp_t1])[0].item()

                        if not C1T[h1, temp_t1] or not C2T[h2, temp_t2]:
                            continue

                        right_counter += 1
                        window_h1_loc.append([C1X[h1, temp_t1], C1Y[h1, temp_t1]])
                        window_h2_loc.append([C2X[h2, temp_t2], C2Y[h2, temp_t2]])

                    # cost[i1, i2] = np.sqrt((C1X[h1, t1] - C2X[h2, t2])**2 + (C1Y[h1, t1] - C2Y[h2, t2])**2)
                    cost[i1, i2] = np.mean(
                        np.sqrt(np.sum(
                            np.square(np.array(window_h1_loc) - np.array(window_h2_loc)),
                            axis=1,
                            keepdims=False
                        ))
                    )

                    sub_gate[i1, i2] = gate_prev[h1, h2, t1]

            i1_matched_list, i2_matched_list = hungarian(cost, sub_gate)
            for i1, i2 in zip(i1_matched_list, i2_matched_list):
                h1 = H1_present_list[i1]
                h2 = H2_present_list[i2]
                X[h1, h2, t1] = 1
                distances[h1, h2, t1] = cost[i1, i2]
            end_map_frame = time.time()
            map_frame_time = 0.5 * map_frame_time + 0.5 * (end_map_frame - start_map_frame)

        if filter is None:
            break

        start_filter_time = time.time()
        boundary = filter(distances[X == 1].reshape(-1, 1))
        gate_prev = np.where(np.logical_and(X == 1, distances > boundary), 0, 1)
        X = np.where(distances > boundary, 0, X)
        if np.all(X == X_prev):
            break
        if filter_iter == 1:
            for t1 in range(T1):
                if np.any(X[:, :, t1] != X_prev[:, :, t1]):
                    print(f'[DEBUG] change in frame {t1} after re-mapping after FP elimination with cost', distances[:, :, t1])
        X_prev = X

        end_filter_time = time.time()
        print('[DEBUG] Number of matches:', np.sum(X))

    print(f'[INFO] Mapping object per frame time: {map_frame_time:.2f}s')


    print(f'[INFO] Filter time: {end_filter_time - start_filter_time:.2f}s')
    print(f'[INFO] Total mapping time: {end_filter_time - start_preprocess:.2f}s')

    # just to show without timestamp details
    # X_notime = np.any(X, axis=2)
    # X_notime = list(zip(*np.where(X_notime == 1)))

    # for error analysis
    if 'true_mct_trackertracker_correspondences' in kwargs:
        # ASSUMING it is either Python list [(_, _),...] or Numpy of shape (N1, N2, T1)
        if isinstance(kwargs['true_mct_trackertracker_correspondences'], list):
            X_true = np.zeros_like(X)
        else:
            X_true = kwargs['true_mct_trackertracker_correspondences']
        X_eval = np.zeros_like(X)
        for t1 in range(T1):
            if not np.any(time_correspondences[t1]):
                continue

            t2 = np.where(time_correspondences[t1])[0].item()

            true_H1_present_list = np.where(C1T[:, t1])[0]
            true_H2_present_list = np.where(C2T[:, t2])[0]

            if isinstance(kwargs['true_mct_trackertracker_correspondences'], list):
                for h1 in true_H1_present_list:
                    for h2 in true_H2_present_list:
                        if (h1, h2) in kwargs['true_mct_trackertracker_correspondences']:
                            X_true[h1, h2, t1] = 1

            for h1 in range(N1):
                for h2 in range(N2):
                    if X[h1, h2, t1] == X_true[h1, h2, t1] == 1:        # true positive
                        X_eval[h1, h2, t1] = 1
                    elif X[h1, h2, t1] == 1 and X_true[h1, h2, t1] == 0: # false positive
                        X_eval[h1, h2, t1] = 2
                    elif X[h1, h2, t1] == 0 and X_true[h1, h2, t1] == 1: # false negative
                        X_eval[h1, h2, t1] = 3

        return C1T, C2T, C1X, C2X, C1Y, C2Y, C1S, C2S, C1X_no_trans, C2X_no_trans, C1Y_no_trans, C2Y_no_trans, C1S_no_trans, C2S_no_trans, time_correspondences, X, X_eval
    else:
        return C1T, C2T, C1X, C2X, C1Y, C2Y, C1S, C2S, C1X_no_trans, C2X_no_trans, C1Y_no_trans, C2Y_no_trans, C1S_no_trans, C2S_no_trans, time_correspondences, X, X


def error_analysis_mct_mapping(
        cap1, cap2,
        homo,
        roi,
        C1T, C2T,
        C1X, C2X,
        C1Y, C2Y,
        C1S, C2S,
        C1X_no_trans, C2X_no_trans,
        C1Y_no_trans, C2Y_no_trans,
        C1S_no_trans, C2S_no_trans,
        time_correspondences,
        X_pred,
        X_eval,
        display,
        export_video,
        checking_true_gttracker=False,
        log_file=None
):

    # read predecessed frames of cam 2 because X chose T1 as the z-dim
    if np.any(time_correspondences[0]):
        for _ in range(np.where(time_correspondences[1])[0].item() - 1):    # index 1 of time_corr and -1 in range is because of frame_id in pretrained tracker starting from 1
            cap2.read()

    N1, T1 = C1T.shape
    N2, T2 = C2T.shape

    if display:
        cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    if export_video is not None:
        writer_created = False

    TP = 0
    FP = 0
    FN = 0

    for t1 in range(1, T1):

        _, frame1 = cap1.read()

        if not np.any(time_correspondences[t1]):
            t2 = None
            __, frame2 = True, np.full_like(frame1, 0)
        else:
            t2 = np.where(time_correspondences[t1])[0].item()
            if not checking_true_gttracker:
                __, frame2 = cap2.read()
            else:
                __, frame2 = _, frame1

        if not (_ and __ and frame1 is not None and frame2 is not None):
            if display:
                cv2.destroyAllWindows()
            break

        if display or export_video is not None:
            H, W = frame2.shape[:2]
            frame1_no_trans = frame1.copy()
            if homo is not None:

                homo_inv = np.linalg.inv(homo)
                roi_inv_trans = cv2.perspectiveTransform(
                    np.float32(roi),
                    homo_inv).astype('int32')

            if checking_true_gttracker:
                frame2 = frame1.copy()
            black = np.full_like(frame2, 0)
            cv2.drawContours(black, [roi], -1, (255, 255, 255), 2)
            cv2.drawContours(frame2, [roi_inv_trans if checking_true_gttracker and homo is not None else roi], -1, (255, 255, 255), 2)
            cv2.drawContours(frame1_no_trans, [roi_inv_trans if homo is not None else roi], -1, (255, 255, 255), 2)
            cv2.putText(black, f'frame {t1}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=2)

            for h1 in range(N1):
                if C1X[h1, t1] == -1:
                    continue

                color1 = COLORS[h1 % len(COLORS)]
                cv2.circle(black, (np.int32(C1X[h1, t1]), np.int32(C1Y[h1, t1])), radius=6, color=color1, thickness=-1)
                cv2.putText(black, f'1-{h1}', (np.int32(C1X[h1, t1]), np.int32(C1Y[h1, t1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, thickness=2)

                cv2.polylines(frame1_no_trans, [np.int32(C1S_no_trans[h1, t1]).reshape(-1, 1, 2)], True, color=color1, thickness=2)
                cv2.circle(frame1_no_trans, (np.int32(C1X_no_trans[h1, t1]), np.int32(C1Y_no_trans[h1, t1])), radius=6, color=color1, thickness=-1)
                cv2.putText(frame1_no_trans, f'1-{h1}', (np.int32(C1X_no_trans[h1, t1]), np.int32(C1Y_no_trans[h1, t1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, thickness=2)

            for h2 in range(N2):
                if t2 is None or C2X[h2, t2] == -1:
                    continue

                color2 = COLORS[h2 % len(COLORS)]
                cv2.polylines(frame2, [np.int32(C2S_no_trans[h2, t2] if checking_true_gttracker and homo is not None else C2S[h2, t2]).reshape(-1, 1, 2)], True, color=color2, thickness=2)
                cv2.circle(frame2, (np.int32(C2X_no_trans[h2, t2]), np.int32(C2Y_no_trans[h2, t2])) if checking_true_gttracker and homo is not None else (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])), radius=6, color=color2, thickness=-1)
                cv2.circle(black, (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])), radius=6, color=color2, thickness=-1)
                cv2.putText(frame2, f'2-{h2}', (np.int32(C2X_no_trans[h2, t2]), np.int32(C2Y_no_trans[h2, t2]) - 10) if checking_true_gttracker and homo is not None else (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, thickness=2)
                cv2.putText(black, f'2-{h2}', (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, thickness=2)

        for h1 in range(N1):
            for h2 in range(N2):

                if X_eval[h1, h2, t1] == 1:
                    connection_color = (0, 255, 0)
                    TP += 1
                elif X_eval[h1, h2, t1] == 2:
                    connection_color = (0, 0, 255)
                    FP += 1
                elif X_eval[h1, h2, t1] == 3:
                    connection_color = (11, 185, 255)
                    FN += 1

                if (display or export_video is not None) and X_eval[h1, h2, t1] > 0:
                    cv2.line(black,
                             (np.int32(C1X[h1, t1]), np.int32(C1Y[h1, t1])),
                             (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])),
                             color=connection_color, thickness=3)

        if display or export_video is not None:
            if homo is not None:
                frame1 = cv2.warpPerspective(frame1_no_trans, homo, (W, H))
            else:
                frame1 = frame1_no_trans

            show_img = np.concatenate(
                [np.concatenate([frame1, frame2], axis=1),
                 np.concatenate([frame1_no_trans, black], axis=1)],
                axis=0
            )

            if display:
                cv2.imshow('show', show_img)
                key = cv2.waitKey(50)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key == ord('e'):
                    exit(0)
                elif key == ord(' '):
                    cv2.waitKey(0)
            if export_video is not None:
                if not writer_created:
                    writer = cv2.VideoWriter(
                        str(HERE / f'../../output/{export_video}'),
                        cv2.VideoWriter_fourcc(*'XVID'),
                        cap1.get(cv2.CAP_PROP_FPS),
                        (show_img.shape[1], show_img.shape[0])
                    )
                    writer_created = True
                writer.write(show_img)

    if display:
        cv2.destroyAllWindows()

    print('TP:', TP, file=log_file)
    print('FP:', FP, file=log_file)
    print('FN:', FN, file=log_file)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print('Pre:', precision, file=log_file)
    print('Rec:', recall, file=log_file)
    F1 = 2 * precision * recall / (precision + recall)
    print('F1:', F1, file=log_file)


def detect_IDSW(
        C1T, C2T,
        C1X, C2X,
        C1Y, C2Y,
        C1S, C2S,
        C1X_no_trans, C2X_no_trans,
        C1Y_no_trans, C2Y_no_trans,
        C1S_no_trans, C2S_no_trans,
        time_correspondences,
        X_pred,
        X_eval,
        log_file=None
):
    N1, T1 = C1T.shape
    N2, T2 = C2T.shape

    pairs = np.transpose(np.where(np.any(X_pred, axis=2))) # [[h1, h2], ...]

    def _process_track(label, h1, maps_of_h1, X_pred, C2T, time_correspondences):
        for h2a in maps_of_h1:
            for h2b in maps_of_h1:
                if h2a <= h2b:
                    continue

                h1_map_h2a = X_pred[h1, h2a]
                h1_map_h2b = X_pred[h1, h2b]
                if label == 'CAM_1':
                    both_h2_present = ((C2T[h2a] * C2T[h2b]).reshape(1, -1) \
                                       @ time_correspondences) \
                        .reshape(-1)
                else:
                    both_h2_present = C2T[h2a] * C2T[h2b]

                # if h2_a, h2_b co-occur when one of them is mapped to h1
                # then there is an object swap
                h1_h2a_h2b_present_and_map = np.logical_or(h1_map_h2a, h1_map_h2b) * both_h2_present
                object_swap_found = np.any(h1_h2a_h2b_present_and_map)

                if not object_swap_found:
                    continue

                print(f'{label} ID {h1} - {"CAM_2" if label == "CAM_1" else "CAM_1"} ID {h2a} and {h2b}, co-occur at frame {np.where(h1_h2a_h2b_present_and_map)[0][0]} (w.r.t cam 1)', file=log_file)

                time_ascending_maps = sorted(list(
                    zip(*np.where(np.stack(
                        [h1_map_h2a, h1_map_h2b],
                        axis=0
                    )))
                ), key=lambda x: x[1])
                for i in range(0, len(time_ascending_maps)):

                        # current_map and previous_map are now 0, 1
                        current_map, current_time = time_ascending_maps[i]
                        current_map = h2a if current_map == 0 else h2b
                        if i == 0:
                            start_time_of_previous_map = current_time
                            continue

                        previous_map, previous_time = time_ascending_maps[i - 1]
                        previous_map = h2a if previous_map == 0 else h2b
                        if current_map != previous_map:
                            print(
                                f'\t switched from {h1} - {previous_map} (at frame {start_time_of_previous_map}) to {h1} - {current_map} (at frame {current_time}) (w.r.t cam 1)', file=log_file)
                            start_time_of_previous_map = current_time

    for h1 in range(N1):
        maps_of_h1 = pairs[pairs[:, 0] == h1][:, 1]
        _process_track('CAM_1', h1, maps_of_h1, X_pred, C2T, time_correspondences.T)
    for h2 in range(N2):
        maps_of_h2 = pairs[pairs[:, 1] == h2][:, 0]
        _process_track('CAM_2', h2, maps_of_h2, np.swapaxes(X_pred, 0, 1), C1T, time_correspondences)


# TODO (tomorrow):
#  1. hàm tính khoảng cách 2 track (euclid, Hausdorff, DTW)
#  2. hàm match các điểm của 2 track
#  3. hàm smooth bằng kalman/moving average cho track


if __name__ == '__main__':

    VIDEO_SET = {
        '2d_v1': {'cam_id1': 21, 'cam_id2': 27, 'box_repr_kind': 'foot', 'range_': range(0, 16)},
        '2d_v2': {'cam_id1': 21, 'cam_id2': 27, 'box_repr_kind': 'foot', 'range_': range(19, 25)},
        '2d_v3': {'cam_id1': 121, 'cam_id2': 127, 'box_repr_kind': 'bottom', 'range_': range(6, 7)},
    }
    TRACKER_SET = [
        'YOLOv5l_pretrained-640-ByteTrack',
        'YOLOv8l_pretrained-640-ByteTrack',
        'YOLOv8l_pretrained-640-StrongSORT',
        'YOLOv8m_pretrained-640-ByteTrack',
        'YOLOv8s_pretrained-640-ByteTrack',
        'YOLOXl_pretrained-640-ByteTrack',
        'YOLOXm_pretrained-640-ByteTrack',
        'YOLOXs_pretrained-640-ByteTrack'
    ]
    FILTER_CHOICE = [None, 'IQR', 'GMM']
    WINDOW_CHOICE = [(1, 0), (11, 5)]

    video_version = '2d_v3'
    TRACKER_NAME = 'YOLOv8l_pretrained-640-ByteTrack'
    filter_type = 'IQR'  # None, 'GMM', 'IQR'
    max_filter_iters = 2
    IQR_lower = 25
    IQR_upper = 75
    window_size = 1
    window_boundary = 0


    cam1_id = VIDEO_SET[video_version]['cam_id1']
    cam2_id = VIDEO_SET[video_version]['cam_id2']
    box_repr_kind = VIDEO_SET[video_version]['box_repr_kind']
    range_ = VIDEO_SET[video_version]['range_']
    n_0 = 5
    # video_id = 19
    # log_file is set to None if stdout

    gttracker_filter = IQRFilter(25, 75).run
    if filter_type == 'GMM':
        filter = GMMFilter(n_components=2, std_coef=3).run
        # gttracker_filter = GMMFilter(n_components=1, std_coef=3).run
    elif filter_type == 'IQR':
        filter = IQRFilter(IQR_lower, IQR_upper).run
        # gttracker_filter = IQRFilter(25, 75).run
    else:
        filter = None
        # gttracker_filter = None

    cf = f'{("GMM" if filter_type == "GMM" else "IQR" + str(IQR_lower) + str(IQR_upper)) if filter_type else "noFilter"}_windowsize{window_size}_windowboundary{window_boundary}'
    log_file = None #open(str(HERE / f'../../data/recordings/{video_version}/{TRACKER_NAME}/log_error_analysis_pred_mct_trackertracker_correspondences_v2_{cf}.txt'), 'w')
    IDSW_log_file = None #open(str(HERE / f'../../data/recordings/{video_version}/{TRACKER_NAME}/IDSW_{cf}.txt'), 'w')
    print(f'================= ERROR ANALYSIS FOR TRACKER {TRACKER_NAME} VIDEO VERSION {video_version} WITH{" " + ("GMM" if filter_type == "GMM" else "IQR" + str(IQR_lower) + str(IQR_upper)) if filter_type else "OUT"} FILTER, WINDOW_SIZE = {window_size}, WINDOW_BOUNDARY = {window_boundary} ================', file=log_file)

    for video_id in tqdm(range_):

        gt_txt1 = str(list((HERE / f'../../data/recordings/{video_version}/gt').glob(f"{cam1_id}_{('00000' + str(video_id))[-n_0:]}_*_*.txt"))[0])
        print(gt_txt1)
        tracker_txt1 = str(list((HERE / f'../../data/recordings/{video_version}/{TRACKER_NAME}/sct').glob(f"{cam1_id}_{('00000' + str(video_id))[-n_0:]}_*_*.txt"))[0])
        print(tracker_txt1)
        gt_txt2 = str(list((HERE / f'../../data/recordings/{video_version}/gt').glob(f"{cam2_id}_{('00000' + str(video_id))[-n_0:]}_*_*.txt"))[0])
        print(gt_txt2)
        tracker_txt2 = str(list((HERE / f'../../data/recordings/{video_version}/{TRACKER_NAME}/sct').glob(f"{cam2_id}_{('00000' + str(video_id))[-n_0:]}_*_*.txt"))[0])
        print(tracker_txt2)


        cap1 = cv2.VideoCapture(
            str(list((HERE / f'../../data/recordings/{video_version}/videos').glob(f"{cam1_id}_{('00000' + str(video_id))[-n_0:]}_*_*.avi"))[0])
        )
        midpoint1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps1 = cap1.get(cv2.CAP_PROP_FPS)

        cap2 = cv2.VideoCapture(
            str(list((HERE / f'../../data/recordings/{video_version}/videos').glob(f"{cam2_id}_{('00000' + str(video_id))[-n_0:]}_*_*.avi"))[0])
        )
        midpoint2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        homo = get_homo(cam1_id, cam2_id, video_version)
        roi = get_roi(cam2_id, video_version)


        ################### MAKE TRUE GT TRACKER CORRESPONDENCES V2 #############################################
        ret1 = make_true_sct_gttracker_correspondences_v2(
            gt_txt1, tracker_txt1,
            midpoint1,
            filter=gttracker_filter,
            box_repr_kind=box_repr_kind,
            homo=homo, # must have
            roi=roi,
            use_iou=False
        )
        ret2 = make_true_sct_gttracker_correspondences_v2(
            gt_txt2, tracker_txt2,
            midpoint2,
            filter=gttracker_filter,
            box_repr_kind=box_repr_kind,
            roi=roi,
            use_iou=False
        )

        # error_analysis_mct_mapping(
        #     cap1, cap1,
        #     homo,
        #     roi,
        #     *ret1,
        #     display=False,
        #     export_video=f'true_sct_gttracker_correspondences_{TRACKER_NAME}_{cam1_id}_{video_id}.avi',
        #     checking_true_gttracker=True
        # )
        # error_analysis_mct_mapping(
        #     cap2, cap2,
        #     None,
        #     roi,
        #     *ret2,
        #     display=False,
        #     export_video=f'true_sct_gttracker_correspondences_{TRACKER_NAME}_{cam2_id}_{video_id}.avi',
        #     checking_true_gttracker=True
        # )
        # continue
        # for t in range(ret1[-2].shape[-1]):
        #     if np.any(ret1[-2][:, :, t]):
        #         print(t, list(zip(*np.where(ret1[-2][:, :, t]))))
        # for t in range(ret2[-2].shape[-1]):
        #     if np.any(ret2[-2][:, :, t]):
        #         print(t, list(zip(*np.where(ret2[-2][:, :, t]))))
        ######################################################################################################



        #################### MAKE TRUE MCT TRACKER TRACKER CORRESPONDENCES V2 ###################################
        with open(f'../../data/recordings/{video_version}/true_mct_gtgt_correspondences.txt', 'r') as f:
            mct_gtgt_correspondences = f.read().strip().split('\n')
            mct_gtgt_correspondences = [eval(l) for l in mct_gtgt_correspondences]
            mct_gtgt_correspondences = [(l[2], l[5]) for l in mct_gtgt_correspondences if
                                      l[0] == cam1_id and l[3] == cam2_id and l[1] == video_id]
        ret = make_true_mct_trackertracker_correspondences_v2(
            tracker_txt1, tracker_txt2,
            fps1, fps2,
            midpoint1, midpoint2,
            mct_gtgt_correspondences,
            ret1[-1],
            ret2[-1],
            homo,
            roi,
            box_repr_kind=box_repr_kind
        )

        # error_analysis_mct_mapping(
        #     cap1, cap2,
        #     homo,
        #     roi,
        #     *ret,
        #     display=False,
        #     export_video=f'true_mct_trackertracker_correspondences_{TRACKER_NAME}_{cam1_id}_{cam2_id}_{video_id}.avi',
        # )
        # for t in range(ret[-2].shape[-1]):
        #     if np.any(ret[-2][:, :, t]):
        #         print(t, list(zip(*np.where(ret[-2][:, :, t]))))
        ######################################################################################################


        ######################## PREDICT MCT TRACKER TRACKER CORRESPONDENCES V2 ########################################
        print(f'\n\n[INFO]\t VIDEO {video_id} PREDICT MCT TRACKER TRACKER CORRESPONDENCES V2', file=log_file)
        ret = mct_mapping(
            tracker_txt1, tracker_txt2,
            fps1, fps2,
            midpoint1, midpoint2,
            homo,
            roi,
            filter=filter,
            max_filter_iters=max_filter_iters,
            box_repr_kind=box_repr_kind,
            window_size=window_size,
            window_boundary=window_boundary,
            true_mct_trackertracker_correspondences=ret[-1], # COMMENT OUT IF NOT ERROR ANALYSIS OR DETECT IDSW
        )
        # # COMMENT OUT IF NOT ERROR ANALYSIS
        error_analysis_mct_mapping(
            cap1, cap2,
            homo,
            roi,
            *ret,
            display=False,
            export_video=None, #f'pred_mct_trackertracker_correspondences_{TRACKER_NAME}_{cam1_id}_{cam2_id}_{video_id}_{cf}.avi', # None
            log_file=log_file
        )

        # for t in range(ret[-2].shape[2]):
        #     print(list(zip(*np.where(ret[-2][:, :, t]))))

        # detect_IDSW(*ret, log_file=IDSW_log_file)

        ##############################################################################################################

    # f.close()
    # '''


