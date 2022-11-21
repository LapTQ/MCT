import cv2
import numpy as np
from pathlib import Path
from vis_utils import COLORS
from tqdm import tqdm

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


def get_homo(src, dst):

    # TODO any
    return np.loadtxt(str(HERE / '../../data/21_to_27.txt'))

def map_tracks(cam1, cam2, vid_id):

    mongo = Pymongo.Builder('localhost', 1111).set_database('tracking').set_collection('20221118235019_sct').get_product()

    list_tracks = list(mongo.collection.find({'videoid': vid_id, 'camid': {'$in': [cam1, cam2]}}))

    mongo.close()

    # categorize tracks by camid
    by_camid = {cam1: {}, cam2: {}}
    for track in list_tracks:
        by_camid[track['camid']][track['trackid']] = track['detections']

    # just to get the frame height and width
    # TODO: get frame height and width from database/config
    cap1 = cv2.VideoCapture(
        str(list(Path('../../data/recordings').glob(f'{cam1}_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    cap2 = cv2.VideoCapture(
        str(list(Path('../../data/recordings').glob(f'{cam2}_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    homo = get_homo(cam1, cam2)

    distance_matrix = np.empty((len(by_camid[cam1]), len(by_camid[cam2])), dtype='float32')
    iou_matrix = np.empty((len(by_camid[cam1]), len(by_camid[cam2])), dtype='float32')
    # sample time correspondences of 2 tracks
    track_indexes1 = {}
    track_indexes2 = {}
    for idx1, id1 in enumerate(by_camid[cam1]):
        for idx2, id2 in enumerate(by_camid[cam2]):

            track_indexes1[idx1] = id1
            track_indexes2[idx2] = id2

            timestamps1 = [det['time'].timestamp() for det in by_camid[cam1][id1]]
            timestamps2 = [det['time'].timestamp() for det in by_camid[cam2][id2]]

            # TODO thu nghiem anh Manh: map tren tung frag de xem vung nao ko nen dung homo
            (matched_indexes1, matched_indexes2), unmatched_indexes1, unmatched_indexes2 = bipartite_match(timestamps1, timestamps2)
            # print(f'[INFO] Number of points sampled from trackid {id1} and {id2}: {len(matched_indexes1)}')
            sampled_boxes1 = [by_camid[cam1][id1][idx]['box'] for idx in matched_indexes1]
            sampled_boxes2 = [by_camid[cam2][id2][idx]['box'] for idx in matched_indexes2]

            repr_points1 = get_box_repr(sampled_boxes1, kind='foot', midpoint=(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            repr_points2 = get_box_repr(sampled_boxes2, kind='foot', midpoint=(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            repr_points1_trans = cv2.perspectiveTransform(repr_points1.reshape(-1, 1, 2), homo)
            if repr_points1_trans is None:
                repr_points1_trans = []
            else:
                repr_points1_trans = repr_points1_trans.reshape(-1, 2)

            # TODO nghien cuu so luong match (VD iou), threshold
            distance_matrix[idx1, idx2] = trajectory_distance(repr_points1_trans, repr_points2, kind='euclid')
            iou_matrix[idx1, idx2] = len(matched_indexes1) / (len(matched_indexes1) + len(unmatched_indexes1) + len(unmatched_indexes2))

    cost_matrix = distance_matrix / iou_matrix

    # print(distance_matrix)
    # print(iou_matrix)
    # print(cost_matrix)

    matched_track_indexes1, matched_track_indexes2 = linear_sum_assignment(cost_matrix)
    matched_track_ids1 = [track_indexes1[idx] for idx in matched_track_indexes1]
    matched_track_ids_2 = [track_indexes2[idx] for idx in matched_track_indexes2]
    ret = []
    for id1, id2 in zip(matched_track_ids1, matched_track_ids_2):
        ret.append(f'{cam1},{vid_id},{id1},{cam2},{vid_id},{id2}')

    # ======================== VISUALIZATION ==========================
    # n_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    # window_name = 'show'
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # white = np.full((int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 255, dtype='uint8')
    # history1 = {}
    # history2 = {}
    # for frame_count in range(1, n_frames + 1):
    #     success, frame1 = cap1.read()
    #     success, frame2 = cap2.read()
    #
    #     dets1 = []
    #     dets2 = []
    #     for trackid in by_camid[cam1].keys():
    #         for detection in by_camid[cam1][trackid]:
    #             if detection['frameid'] == frame_count:
    #                 dets1.append([frame_count, trackid] + detection['box'] + [detection['score']])
    #
    #     for trackid in by_camid[cam2].keys():
    #         for detection in by_camid[cam2][trackid]:
    #             if detection['frameid'] == frame_count:
    #                 dets2.append([frame_count, trackid] + detection['box'] + [detection['score']])
    #
    #     dets1 = np.array(dets1).reshape(-1, 7)
    #     dets2 = np.array(dets2).reshape(-1, 7)
    #
    #     points1 = get_box_repr(dets1[:, 2:6], kind='foot', midpoint=(frame1.shape[1] // 2, frame1.shape[0]))
    #     points2 = get_box_repr(dets2[:, 2:6], kind='foot', midpoint=(frame2.shape[1] // 2, frame2.shape[0]))
    #
    #     points1_trans = cv2.perspectiveTransform(points1.reshape(-1, 1, 2), homo)
    #     if points1_trans is None:
    #         points1_trans = []
    #     else:
    #         points1_trans = points1_trans.reshape(-1, 2)
    #
    #     frame1_trans = cv2.warpPerspective(frame1, homo, (frame2.shape[1], frame2.shape[0]))
    #
    #     for box, id in zip(points1_trans, dets1[:, 1]):
    #         cv2.circle(frame1_trans, np.int32(box), radius=3, color=COLORS[(int(id) + 5) % len(COLORS)], thickness=-1)
    #
    #         # write full trajectory
    #         cv2.circle(white, np.int32(box), radius=3, color=COLORS[(int(id) + 5) % len(COLORS)], thickness=-1)
    #         if int(id) in history1:
    #             cv2.line(white, np.int32(box), history1[int(id)], color=COLORS[(int(id) + 5) % len(COLORS)], thickness=1)
    #         history1[int(id)] = np.int32(box)
    #
    #     for box, id in zip(points2, dets2[:, 1]):
    #         cv2.circle(frame2, np.int32(box), radius=3, color=COLORS[int(id) % len(COLORS)], thickness=-1)
    #
    #         # write full trajectory
    #         cv2.circle(white, np.int32(box), radius=3, color=COLORS[int(id) % len(COLORS)], thickness=-1)
    #         if int(id) in history2:
    #             cv2.line(white, np.int32(box), history2[int(id)], color=COLORS[int(id) % len(COLORS)], thickness=1)
    #         history2[int(id)] = np.int32(box)
    #
    #     show_img = np.concatenate([frame1_trans, frame2, white], axis=1)
    #
    #     cv2.imshow(window_name, show_img)
    #     key = cv2.waitKey(50)
    #     if key == 27:
    #         break
    #     elif key == ord('e'):
    #         exit(0)
    #     elif key == ord(' '):
    #         cv2.waitKey(0)
    # =====================================================================

    return '\n'.join(ret)



def evaluate(true_path, pred_path):
    with open(true_path, 'r') as f:
        true = set(l[:-1] for l in f.readlines())
    with open(pred_path, 'r') as f:
        pred = set(l[:-1] for l in f.readlines())

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

    ret = []
    for vid_id in tqdm(range(16)):
        ret.append(map_tracks(21, 27, vid_id))
    with open('../../output/mct.txt', 'w') as f:
        print('\n'.join(ret), file=f)

    evaluate('../../data/recordings/correspondences_mapped.txt', '../../output/mct.txt')

    # map_tracks(0)

    # for each (cam1, cam2) in homo_db:




