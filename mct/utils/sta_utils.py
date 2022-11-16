import cv2
import numpy as np
from pathlib import Path
from vis_utils import COLORS

HERE = Path(__file__).parent



def get_box_repr(boxes, kind, **kwargs):
    """x1, y1, x2, y2"""
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
        1] == 2, 'Invalid trajectory dimension, must be (2,) or (N, 2)'
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


from mct.utils.db_utils import Pymongo
from scipy.optimize import linear_sum_assignment


def bipartite_match(seq1, seq2):
    n1 = len(seq1)
    n2 = len(seq2)
    if not isinstance(seq1, np.ndarray):
        seq1 = np.array(seq1)
    if not isinstance(seq2, np.ndarray):
        seq2 = np.array(seq2)

    assert len(seq1.shape) == len(seq2.shape) == 1, 'Invalid seq dimension, must be (N,)'
    seq1 = np.expand_dims(seq1, axis=1)
    seq2 = np.expand_dims(seq2, axis=0)

    cost_matrix = np.abs(np.subtract(seq1, seq2))
    list_idx1, list_idx2 = linear_sum_assignment(cost_matrix)

    list_unmatched_idx1 = np.array([i for i in range(n1) if i not in list_idx1])
    list_unmatched_idx2 = np.array([i for i in range(n2) if i not in list_idx2])

    return (list_idx1, list_idx2), list_unmatched_idx1, list_unmatched_idx2


def map_tracks(vid_id):

    mongo = Pymongo.Builder('localhost', 1111).set_database('sct_db').set_collection('20221115154937_sct').get_product()

    list_tracks = list(mongo.collection.find({'videoid': vid_id}))

    mongo.close()

    # categorize tracks by camid
    by_camid = {}
    for track in list_tracks:
        camid = track['camid']
        if camid not in by_camid:
            by_camid[camid] = {}
        by_camid[camid][track['trackid']] = track['detections']

    cap21 = cv2.VideoCapture(
        str(list(Path('../../data/recordings').glob(f'21_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    cap27 = cv2.VideoCapture(
        str(list(Path('../../data/recordings').glob(f'27_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    H_21_27 = np.loadtxt(str(HERE / '../../data/21_to_27.txt'))

    # sample time correspondences of 2 tracks
    traj_dist_matrix = np.empty((len(by_camid[21].keys()), len(by_camid[27].keys())), dtype='float32')
    # TODO CHUY: cac trackid dang bat dau tu 1
    # TODO: 3 cam overlap
    for cam21_trackid in by_camid[21].keys():
        for cam27_trackid in by_camid[27].keys():

            traj1_time_series = [detection['time'].timestamp() for detection in by_camid[21][cam21_trackid]]
            traj2_time_series = [detection['time'].timestamp() for detection in by_camid[27][cam27_trackid]]

            # TODO thu nghiem anh Manh: map tren tung frag de xem vung nao ko nen dung homo
            (list_idx21, list_idx27), list_unmatched_idx21, list_unmatched_idx27 = bipartite_match(traj1_time_series, traj2_time_series)
            print(f'[INFO] Number of points sampled from trackid {cam21_trackid} and {cam27_trackid}: {len(list_idx21)}')
            sampled_boxes21 = [by_camid[21][cam21_trackid][idx]['box'] for idx in list_idx21]
            sampled_boxes27 = [by_camid[27][cam27_trackid][idx]['box'] for idx in list_idx27]

            repr_points21 = get_box_repr(sampled_boxes21, kind='foot', midpoint=(cap21.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap21.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            repr_points27 = get_box_repr(sampled_boxes27, kind='foot', midpoint=(cap27.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap27.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            repr_points21_trans = cv2.perspectiveTransform(repr_points21.reshape(-1, 1, 2), H_21_27)
            if repr_points21_trans is None:
                repr_points21_trans = []
            else:
                repr_points21_trans = repr_points21_trans.reshape(-1, 2)

            # TODO CHUY: cac trackid dang bat dau tu 1
            # TODO nghien cuu so luong match (VD iou), threshold
            iou = len(list_idx21) / (len(list_idx21) + len(list_unmatched_idx21) + len(list_unmatched_idx27))
            traj_dist_matrix[cam21_trackid - 1, cam27_trackid - 1] = 1/iou * trajectory_distance(repr_points21_trans, repr_points27, kind='euclid')

    print(traj_dist_matrix)
    # TODO CHUY: cac trackid dang bat dau tu 1
    list_matched_track21, list_matched_track27 = linear_sum_assignment(traj_dist_matrix)
    ret = []
    for trackid21, trackid27 in zip(list_matched_track21, list_matched_track27):
        ret.append(f'21,{vid_id},{trackid21 + 1},27,{vid_id},{trackid27 + 1}')
    return '\n'.join(ret)


    # # ======================== VISUALIZATION ==========================
    # n_frames = int(min(cap21.get(cv2.CAP_PROP_FRAME_COUNT), cap27.get(cv2.CAP_PROP_FRAME_COUNT)))
    # window_name = 'show'
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # white = np.full((int(cap27.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap27.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 255, dtype='uint8')
    # history21 = {}
    # history27 = {}
    # for frame_count in range(1, n_frames + 1):
    #     success, frame21 = cap21.read()
    #     success, frame27 = cap27.read()
    #
    #     dets21 = []
    #     dets27 = []
    #     for trackid in by_camid[21].keys():
    #         for detection in by_camid[21][trackid]:
    #             if detection['frameid'] == frame_count:
    #                 dets21.append([frame_count, trackid] + detection['box'] + [detection['score']])
    #
    #     for trackid in by_camid[27].keys():
    #         for detection in by_camid[27][trackid]:
    #             if detection['frameid'] == frame_count:
    #                 dets27.append([frame_count, trackid] + detection['box'] + [detection['score']])
    #
    #     dets21 = np.array(dets21).reshape(-1, 7)
    #     dets27 = np.array(dets27).reshape(-1, 7)
    #
    #     points21 = get_box_repr(dets21[:, 2:6], kind='foot', midpoint=(frame21.shape[1] // 2, frame21.shape[0]))
    #     points27 = get_box_repr(dets27[:, 2:6], kind='foot', midpoint=(frame27.shape[1] // 2, frame27.shape[0]))
    #
    #     points21_trans = cv2.perspectiveTransform(points21.reshape(-1, 1, 2), H_21_27)
    #     if points21_trans is None:
    #         points21_trans = []
    #     else:
    #         points21_trans = points21_trans.reshape(-1, 2)
    #
    #     frame21_trans = cv2.warpPerspective(frame21, H_21_27, (frame27.shape[1], frame27.shape[0]))
    #
    #     for box, id in zip(points21_trans, dets21[:, 1]):
    #         cv2.circle(frame21_trans, np.int32(box), radius=3, color=COLORS[(int(id) + 5) % len(COLORS)], thickness=-1)
    #
    #         # write full trajectory
    #         cv2.circle(white, np.int32(box), radius=3, color=COLORS[(int(id) + 5) % len(COLORS)], thickness=-1)
    #         if int(id) in history21:
    #             cv2.line(white, np.int32(box), history21[int(id)], color=COLORS[(int(id) + 5) % len(COLORS)], thickness=1)
    #         history21[int(id)] = np.int32(box)
    #
    #     for box, id in zip(points27, dets27[:, 1]):
    #         cv2.circle(frame27, np.int32(box), radius=3, color=COLORS[int(id) % len(COLORS)], thickness=-1)
    #
    #         # write full trajectory
    #         cv2.circle(white, np.int32(box), radius=3, color=COLORS[int(id) % len(COLORS)], thickness=-1)
    #         if int(id) in history27:
    #             cv2.line(white, np.int32(box), history27[int(id)], color=COLORS[int(id) % len(COLORS)], thickness=1)
    #         history27[int(id)] = np.int32(box)
    #
    #     show_img = np.concatenate([frame21_trans, frame27, white], axis=1)
    #
    #     cv2.imshow(window_name, show_img)
    #     key = cv2.waitKey(50)
    #     if key == 27:
    #         break
    #     elif key == ord(' '):
    #         cv2.waitKey(0)
    # # =====================================================================



def evaluate(true_path, pred_path):
    with open(true_path, 'r') as f:
        true = set(l[:-1] for l in f.readlines())
        print(true)
    with open(pred_path, 'r') as f:
        pred = set(l[:-1] for l in f.readlines())
        print(pred)

    TP = true.intersection(pred)
    FP = pred.difference(true)
    FN = true.difference(pred)
    print('TP', len(TP))
    print('FP', len(FP))
    print('FN', len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))
    print('Precision', precision)
    print('Recall', recall)


# TODO (tomorrow):
#  1. hàm tính khoảng cách 2 track (euclid, Hausdorff, DTW)
#  2. hàm match các điểm của 2 track
#  3. hàm smooth bằng kalman/moving average cho track



if __name__ == '__main__':

    # ret = []
    # for vid_id in range(16):
    #     ret.append(map_tracks(vid_id))
    # with open('../../output/mct.txt', 'w') as f:
    #     print('\n'.join(ret), file=f)

    evaluate('../../data/recordings/correspondences.txt', '../../output/mct.txt')



