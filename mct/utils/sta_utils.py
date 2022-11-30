import cv2
import numpy as np
from pathlib import Path
from vis_utils import COLORS
from tqdm import tqdm
from datetime import datetime, timedelta
import os

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

import sys
sys.path.append(sys.path[0] + '/../..')
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


from mct.utils.vis_utils import draw_track

def get_homo(src, dst):

    # TODO any
    return np.loadtxt(str(HERE / '../../data/21_to_27.txt'))

def map_tracks(cam1, cam2, vid_id, export_video=False):

    mongo = Pymongo.Builder('localhost', 1111).set_database('tracking').set_collection('20221124143517_sct').get_product()

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
    matched_track_ids2 = [track_indexes2[idx] for idx in matched_track_indexes2]
    ret = []
    for id1, id2 in zip(matched_track_ids1, matched_track_ids2):
        ret.append(f'{cam1},{vid_id},{id1},{cam2},{vid_id},{id2}')

    # ======================== VISUALIZATION ==========================
    correspondence = np.stack([matched_track_ids1, matched_track_ids2], axis=1)
    LENGTH = 30
    n_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    window_name = 'show'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    history = {cam1: {}, cam2: {}}
    writer_created = False
    for frame_count in tqdm(range(1, n_frames + 1)):
        white = np.full((int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 0,
                        dtype='uint8')
        success, frame1 = cap1.read()
        success, frame2 = cap2.read()

        dets1 = []
        dets2 = []
        for trackid in by_camid[cam1].keys():
            for detection in by_camid[cam1][trackid]:
                if detection['frameid'] == frame_count:
                    dets1.append([frame_count, trackid] + detection['box'] + [detection['score']])

        for trackid in by_camid[cam2].keys():
            for detection in by_camid[cam2][trackid]:
                if detection['frameid'] == frame_count:
                    dets2.append([frame_count, trackid] + detection['box'] + [detection['score']])

        dets1 = np.array(dets1).reshape(-1, 7)
        dets2 = np.array(dets2).reshape(-1, 7)

        coords1 = np.int32(np.stack([dets1[:, 2], dets1[:, 3], dets1[:, 2], dets1[:, 5], dets1[:, 4], dets1[:, 5], dets1[:, 4], dets1[:, 3]], axis=1).reshape(-1, 4, 1, 2))
        coords2 = np.int32(np.stack([dets2[:, 2], dets2[:, 3], dets2[:, 2], dets2[:, 5], dets2[:, 4], dets2[:, 5], dets2[:, 4], dets2[:, 3]], axis=1).reshape(-1, 4, 1, 2))

        points1 = get_box_repr(dets1[:, 2:6], kind='foot', midpoint=(frame1.shape[1] // 2, frame1.shape[0]))
        points2 = get_box_repr(dets2[:, 2:6], kind='foot', midpoint=(frame2.shape[1] // 2, frame2.shape[0]))

        points1_trans = cv2.perspectiveTransform(points1.reshape(-1, 1, 2), homo)
        coords1_trans = cv2.perspectiveTransform(np.float32(coords1).reshape(-1, 1, 2), homo)
        if points1_trans is None:
            points1_trans = []
            coords1_trans = []
        else:
            points1_trans = points1_trans.reshape(-1, 2)
            coords1_trans = coords1_trans.reshape(-1, 4, 1, 2).astype('int32')

        frame1_trans = cv2.warpPerspective(frame1, homo, (frame2.shape[1], frame2.shape[0]))

        for id in history[cam1]:
            if not np.any(dets1[:, 1] == id):
                history[cam1][id] = []
        for id in history[cam2]:
            if not np.any(dets2[:, 1] == id):
                history[cam2][id] = []

        for pts, id, box_trans, box in zip(points1_trans, dets1[:, 1], coords1_trans, coords1):
            id = int(id)
            if id not in history[cam1]:
                history[cam1][id] = []
            history[cam1][id].append(pts)
            if len(history[cam1][id]) > LENGTH:
                del history[cam1][id][0]
            color = COLORS[id % len(COLORS)]
            cv2.polylines(frame1_trans, [box_trans], True, color, thickness=2)
            cv2.polylines(frame1, [box], True, color, thickness=2)
            frame1_trans = draw_track(frame1_trans, history[cam1][id], color=color, radius=5)
            white = draw_track(white, history[cam1][id], color=color, radius=3, camid=cam1)

        for pts, id, box in zip(points2, dets2[:, 1], coords2):
            id = int(id)
            if id not in history[cam2]:
                history[cam2][id] = []
            history[cam2][id].append(pts)
            if len(history[cam2][id]) > LENGTH:
                del history[cam2][id][0]
            color = COLORS[(correspondence[correspondence[:, 1] == id][0, 0]) % len(COLORS)]
            cv2.polylines(frame2, [box], True, color, thickness=2)
            frame2 = draw_track(frame2, history[cam2][id], color=color, radius=5)
            white = draw_track(white, history[cam2][id], color=color, radius=3, camid=cam2)

        show_img = np.concatenate([
            np.concatenate([frame1_trans, frame2], axis=1),
            np.concatenate([frame1, white], axis=1)
        ], axis=0)

        if export_video:
            if not writer_created:
                writer = cv2.VideoWriter(
                    str(HERE/f'../../output/map_{cam1}_{cam2}_{vid_id}.avi'),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    10.0,
                    (show_img.shape[1], show_img.shape[0])
                )
                writer_created = True
            else:
                writer.write(show_img)

        cv2.imshow(window_name, show_img)
        key = cv2.waitKey(50)
        if key == 27:
            break
        elif key == ord('e'):
            exit(0)
        elif key == ord(' '):
            cv2.waitKey(0)
    # =====================================================================

    return '\n'.join(ret)


def analyze_homo(cam1, cam2, vid_id, correspondence, vis=False, export_video=False):

    mongo = Pymongo.Builder('localhost', 1111).set_database('tracking').set_collection(
        '20221124143517_sct').get_product()

    list_tracks = list(mongo.collection.find({'videoid': vid_id, 'camid': {'$in': [cam1, cam2]}}))

    mongo.close()

    # categorize tracks by camid
    by_camid = {cam1: {}, cam2: {}}
    for track in list_tracks:
        by_camid[track['camid']][track['trackid']] = track['detections']

    # just to get the frame height and width
    # TODO: get frame height and width from database/config
    vid1_path = str(list(Path('../../data/recordings').glob(f'{cam1}_{("00000" + str(vid_id))[-5:]}*.avi'))[0])
    vid2_path = str(list(Path('../../data/recordings').glob(f'{cam2}_{("00000" + str(vid_id))[-5:]}*.avi'))[0])
    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)
    homo = get_homo(cam1, cam2)

    fps = cap1.get(cv2.CAP_PROP_FPS)
    vid1_starttime = datetime.strptime('_'.join(os.path.splitext(os.path.split(vid1_path)[1])[0].split('_')[2:]), '%Y-%m-%d_%H-%M-%S-%f')
    n_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    vid1_endtime = vid1_starttime + timedelta(seconds=(n_frames1 - 1) / fps)
    vid2_starttime = datetime.strptime('_'.join(os.path.splitext(os.path.split(vid2_path)[1])[0].split('_')[2:]), '%Y-%m-%d_%H-%M-%S-%f')
    n_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    vid2_endtime = vid2_starttime + timedelta(seconds=(n_frames2 - 1) / fps)

    min_timestamp = min(vid1_starttime.timestamp(), vid2_starttime.timestamp())
    max_timestamp = max(vid1_endtime.timestamp(), vid2_endtime.timestamp())
    frame_diff = int((vid2_starttime.timestamp() - vid1_starttime.timestamp()) * fps)
    n_timestamp = int((max_timestamp - min_timestamp) * fps + 1)

    # n_timestamp = max([len(seq) for cam in by_camid for seq in by_camid[cam].values()])
    # min_timestamp = min([det['time'].timestamp() for cam in by_camid for id in by_camid[cam] for det in by_camid[cam][id]])
    # max_timestamp = max(
    #     [det['time'].timestamp() for cam in by_camid for id in by_camid[cam] for det in by_camid[cam][id]])
    timestampsa = np.linspace(min_timestamp, max_timestamp, n_timestamp)

    distance_tensor = np.full((len(by_camid[cam1]), len(by_camid[cam2]), n_timestamp), 1e9, dtype='float32')
    mask_tensor = np.full_like(distance_tensor, False, dtype='bool')

    # sample time correspondences of 2 tracks
    track_indexes1 = {}
    track_indexes2 = {}
    points_records = {}
    for idx1, id1 in enumerate(by_camid[cam1]):
        for idx2, id2 in enumerate(by_camid[cam2]):

            track_indexes1[idx1] = id1
            track_indexes2[idx2] = id2

            timestamps1 = [det['time'].timestamp() for det in by_camid[cam1][id1]]
            timestamps2 = [det['time'].timestamp() for det in by_camid[cam2][id2]]

            (matched_indexes1, matched_indexesa1), _, _ = bipartite_match(timestamps1, timestampsa)
            (matched_indexes2, matched_indexesa2), _, _ = bipartite_match(timestamps2, timestampsa)

            overlap = [(time_idxa1, time_idx1, time_idx2)
                       for (time_idx1, time_idxa1) in zip(matched_indexes1, matched_indexesa1)
                       for (time_idx2, time_idxa2) in zip(matched_indexes2, matched_indexesa2)
                       if time_idxa1 == time_idxa2]
            for time_idxa, time_idx1, time_idx2 in overlap:
                box1 = by_camid[cam1][id1][time_idx1]['box']
                box2 = by_camid[cam2][id2][time_idx2]['box']

                repr_points1 = get_box_repr([box1], kind='foot', midpoint=(
                    cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                repr_points2 = get_box_repr([box2], kind='foot', midpoint=(
                    cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                repr_points1_trans = cv2.perspectiveTransform(repr_points1.reshape(-1, 1, 2), homo).reshape(-1, 2)

                distance_tensor[idx1, idx2, time_idxa] = trajectory_distance(repr_points1_trans, repr_points2, kind='euclid')
                mask_tensor[idx1, idx2, time_idxa] = True


                coords1 = np.int32(
                    [box1[0], box1[1], box1[0], box1[3], box1[2], box1[3], box1[2],
                     box1[1]]).reshape(4, 1, 2)
                coords2 = np.int32(
                    [box2[0], box2[1], box2[0], box2[3], box2[2], box2[3], box2[2],
                     box2[1]]).reshape(4, 1, 2)
                coords1_trans = cv2.perspectiveTransform(np.float32(coords1), homo).astype('int32')


                points_records[(time_idxa, id1, id2)] = (coords1, coords1_trans, coords2, repr_points1[0], repr_points1_trans[0], repr_points2[0])


    if vis:
        cv2.namedWindow(str(vid_id), cv2.WINDOW_NORMAL)
    writer_created = False


    count_total = 0
    count_false = 0
    frame_skipped = False
    distance_tensor = distance_tensor[:, :, abs(frame_diff):]
    mask_tensor = mask_tensor[:, :, abs(frame_diff):]
    n_timestamp -= abs(frame_skipped)
    for frame_count in tqdm(range(n_timestamp)):
        matches = []
        matched_track_indexes1, matched_track_indexes2 = linear_sum_assignment(distance_tensor[:, :, frame_count])
        matched_track_ids1 = [track_indexes1[idx] for idx in matched_track_indexes1]
        matched_track_ids2 = [track_indexes2[idx] for idx in matched_track_indexes2]

        if not frame_skipped:
            for _ in range(abs(frame_diff)):
                if frame_diff > 0:
                    _, frame1 = cap1.read()
                else:
                    __, frame2 = cap2.read()
            frame_skipped = True

        _, frame1 = cap1.read()
        __, frame2 = cap2.read()

        if not (_ and __ and frame1 is not None and frame2 is not None):
            if vis:
                cv2.destroyAllWindows()
            return

        frame1_trans = cv2.warpPerspective(frame1, homo, (frame2.shape[1], frame2.shape[0]))

        white = np.full((int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 0,
                        dtype='uint8')

        for idx1, idx2, id1, id2 in zip(matched_track_indexes1, matched_track_indexes2, matched_track_ids1, matched_track_ids2):

            if mask_tensor[idx1, idx2, frame_count]:
                matches.append((id1, id2))
                count_total += 1

                c1, c1_trans, c2, p1, p1_trans, p2 = points_records[(frame_count + abs(frame_diff), id1, id2)]

                if (id1, id2) not in correspondence:
                    count_false += 1
                    tf_color = (0, 0, 255)
                else:
                    tf_color = (0, 255, 0)


                cv2.line(white, np.int32(p1_trans), np.int32(p2), color=tf_color, thickness=3)

                color1 = COLORS[id1 % len(COLORS)]
                color2 = COLORS[id2 % len(COLORS)]

                cv2.polylines(frame1_trans, [c1_trans], True, color1, thickness=2)
                cv2.polylines(frame1, [c1], True, color1, thickness=2)
                cv2.polylines(frame2, [c2], True, color2, thickness=2)
                cv2.circle(frame1_trans, np.int32(p1_trans), radius=5, color=color1, thickness=-1)
                cv2.circle(frame1, np.int32(p1), radius=5, color=color1, thickness=-1)
                cv2.circle(frame2, np.int32(p2), radius=5, color=color2, thickness=-1)

                cv2.circle(white, np.int32(p1_trans), radius=5, color=color1, thickness=-1)
                cv2.putText(white, f'{cam1} ({id1})', (int(p1_trans[0]), int(p1_trans[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, thickness=2)

                cv2.circle(white, np.int32(p2), radius=5, color=color2, thickness=-1)
                cv2.putText(white, f'{cam2} ({id2})', (int(p2[0]), int(p2[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, thickness=2)

        if vis or export_video:
            show_img = np.concatenate(
                [np.concatenate([frame1_trans, frame2], axis=1),
                 np.concatenate([frame1, white], axis=1)],
                axis=0
            )
            if vis:
                cv2.imshow(str(vid_id), show_img)
                key = cv2.waitKey(50)
                if key == 27:
                    cv2.destroyAllWindows()
                    return
                elif key == ord('e'):
                    exit(0)
                elif key == ord(' '):
                    cv2.waitKey(0)
            if export_video:
                if not writer_created:
                    writer = cv2.VideoWriter(
                        str(HERE / f'../../output/check_frame_{cam1}_{cam2}_{vid_id}.avi'),
                        cv2.VideoWriter_fourcc(*'XVID'),
                        10.0,
                        (show_img.shape[1], show_img.shape[0])
                    )
                    writer_created = True
                else:
                    writer.write(show_img)
        # print(matches, correspondence)

    if vis:
        cv2.destroyAllWindows()

    print(f'False: {count_false}/{count_total} =', count_false/count_total)



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

    # ret = []
    # for vid_id in tqdm(range(16)):
    #     ret.append(map_tracks(21, 27, vid_id))
    # with open('../../output/mct.txt', 'w') as f:
    #     print('\n'.join(ret), file=f)
    #
    # evaluate('../../data/recordings/correspondences.txt', '../../output/mct.txt')

    # from multiprocessing import Pool
    # pool = Pool(16)
    # pool.starmap(map_tracks, [(21, 27, vid_id, True) for vid_id in range(16)])

    cam1 = 21
    cam2 = 27
    for vid_id in range(16):
        with open('../../data/recordings/correspondences.txt', 'r') as f:
            true = [eval(l[:-1]) for l in f.readlines()]
            true = [(p[2], p[5]) for p in true if p[0] == cam1 and p[3] == cam2 and p[1] == vid_id]
        print('VID', vid_id, end=': ')
        analyze_homo(cam1, cam2, vid_id, correspondence=true, vis=True, export_video=False)



    # for each (cam1, cam2) in homo_db:
    # vis 5 video, quan sat xem co van de khong (them cam id, them box cho cam 1)
    # vis ket qua theo tung frame, tong chi phi, bo qua van de dong bo => danh gia vung nao tren anh homo khong tot, uu nhuoc diem cua homo



