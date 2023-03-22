import cv2
import numpy as np
from pathlib import Path
from vis_utils import COLORS
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from ortools.linear_solver import pywraplp
from img_utils import iou_batch
import time


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


def get_homo(src, dst, video_version):

    # TODO any
    return np.loadtxt(str(HERE / f'../../data/recordings/{video_version}/homo_{src}_to_{dst}.txt'))


def get_roi(dst, video_version):

    # TODO any
    roi = np.loadtxt(str(HERE / f'../../data/recordings/{video_version}/roi_{dst}.txt'), dtype='float32')
    H, W = cv2.VideoCapture(str(list(Path(HERE / f'../../data/recordings/{video_version}/videos').glob(f'{dst}*.avi'))[0])).read()[1].shape[:2]
    roi[:, 0] *= W
    roi[:, 1] *= H
    roi = roi.reshape(-1, 1, 2).astype('int32')

    return roi

def map_tracks(cam1, cam2, vid_id, video_version, sample_roi, use_iou, box_repr_kind, vis=False, export_video=False):

    mongo = Pymongo.Builder('localhost', 1111).set_database('tracking').set_collection(video_version_to_db_name[video_version]).get_product()

    list_tracks = list(mongo.collection.find({'videoid': vid_id, 'camid': {'$in': [cam1, cam2]}}))

    mongo.close()

    # just to get the frame height and width
    # TODO: get frame height and width from database/config
    cap1 = cv2.VideoCapture(
        str(list(Path(HERE/f'../../data/recordings/{video_version}/videos').glob(f'{cam1}_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    cap2 = cv2.VideoCapture(
        str(list(Path(HERE/f'../../data/recordings/{video_version}/videos').glob(f'{cam2}_{("00000" + str(vid_id))[-5:]}*.avi'))[0]))
    homo = get_homo(cam1, cam2, video_version)
    roi = get_roi(cam2, video_version)

    # categorize tracks by camid, and sample detections in ROI
    by_camid = {cam1: {}, cam2: {}}
    for track in list_tracks:
        by_camid[track['camid']][track['trackid']] = track['detections']

    distance_matrix = np.empty((len(by_camid[cam1]), len(by_camid[cam2])), dtype='float32')
    iou_matrix = np.empty((len(by_camid[cam1]), len(by_camid[cam2])), dtype='float32')

    track_indexes1 = {}
    track_indexes2 = {}
    for idx1, id1 in enumerate(by_camid[cam1]):
        for idx2, id2 in enumerate(by_camid[cam2]):

            track_indexes1[idx1] = id1
            track_indexes2[idx2] = id2

            # filter 1: sample time correspondences of 2 tracks
            timestamps1 = [det['time'].timestamp() for det in by_camid[cam1][id1]]
            timestamps2 = [det['time'].timestamp() for det in by_camid[cam2][id2]]

            (matched_indexes1, matched_indexes2), unmatched_indexes1, unmatched_indexes2 = bipartite_match(timestamps1, timestamps2)
            # print(f'[INFO] Number of points sampled from trackid {id1} and {id2}: {len(matched_indexes1)}')
            sampled_boxes1 = [by_camid[cam1][id1][idx]['box'] for idx in matched_indexes1]
            sampled_boxes2 = [by_camid[cam2][id2][idx]['box'] for idx in matched_indexes2]

            repr_points1 = get_box_repr(sampled_boxes1, kind=box_repr_kind, midpoint=(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            repr_points2 = get_box_repr(sampled_boxes2, kind=box_repr_kind, midpoint=(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            repr_points1_trans = cv2.perspectiveTransform(repr_points1.reshape(-1, 1, 2), homo)

            if repr_points1_trans is None:
                repr_points1_trans = np.empty((0, 2))
            else:
                repr_points1_trans = repr_points1_trans.reshape(-1, 2)

            # filter 2: sample points in ROI
            if sample_roi:
                repr_points1_trans_roi = []
                repr_points2_roi = []
                for p1, p2 in zip(repr_points1_trans, repr_points2):
                    if cv2.pointPolygonTest(roi, p1, True) >= -5 and cv2.pointPolygonTest(roi, p2, True) >= -5:
                        repr_points1_trans_roi.append(p1)
                        repr_points2_roi.append(p2)
                repr_points1_trans = np.array(repr_points1_trans_roi).reshape(-1, 2)
                repr_points2 = np.array(repr_points2_roi).reshape(-1, 2)

            # TODO nghien cuu so luong match (VD iou), threshold
            distance_matrix[idx1, idx2] = trajectory_distance(repr_points1_trans, repr_points2, kind='euclid')
            iou_matrix[idx1, idx2] = len(matched_indexes1) / (len(matched_indexes1) + len(unmatched_indexes1) + len(unmatched_indexes2))

            # after 2 filters, distance matrix may have nan

    if use_iou:
        cost_matrix = distance_matrix / iou_matrix
    else:
        cost_matrix = distance_matrix

    print(track_indexes1, track_indexes2)
    print(distance_matrix)
    print(iou_matrix)
    print(cost_matrix)

    nan_mask = np.isnan(distance_matrix)
    cost_matrix[nan_mask] = 1e9

    matched_track_indexes1, matched_track_indexes2 = linear_sum_assignment(cost_matrix)

    matched_track_ids1 = []
    matched_track_ids2 = []
    for idx1, idx2 in zip(matched_track_indexes1, matched_track_indexes2):
        if not nan_mask[idx1, idx2]:
            matched_track_ids1.append(track_indexes1[idx1])
            matched_track_ids2.append(track_indexes2[idx2])

    ret = []
    for id1, id2 in zip(matched_track_ids1, matched_track_ids2):
        ret.append(f'{cam1},{vid_id},{id1},{cam2},{vid_id},{id2}')

    # ======================== VISUALIZATION ==========================
    if vis or export_video:
        correspondence = np.stack([matched_track_ids1, matched_track_ids2], axis=1)
        LENGTH = 35
        n_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
        if vis:
            window_name = 'show'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        history = {cam1: {}, cam2: {}}
        writer_created = False
        for frame_count in tqdm(range(1, n_frames + 1)):

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

            points1 = get_box_repr(dets1[:, 2:6], kind=box_repr_kind, midpoint=(frame1.shape[1] // 2, frame1.shape[0]))
            points2 = get_box_repr(dets2[:, 2:6], kind=box_repr_kind, midpoint=(frame2.shape[1] // 2, frame2.shape[0]))

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

            white = np.full((int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 0,
                            dtype='uint8')
            cv2.drawContours(white, [roi], -1, (255, 255, 255), 2)
            cv2.drawContours(frame1_trans, [roi], -1, (255, 255, 255), 2)
            cv2.drawContours(frame2, [roi], -1, (255, 255, 255), 2)

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
                frame1_trans = draw_track(frame1_trans, history[cam1][id], id=id, color=color, radius=5)
                white = draw_track(white, history[cam1][id], id=id, color=color, radius=3, camid=cam1)

            for pts, id, box in zip(points2, dets2[:, 1], coords2):
                id = int(id)
                if id not in history[cam2]:
                    history[cam2][id] = []
                history[cam2][id].append(pts)
                if len(history[cam2][id]) > LENGTH:
                    del history[cam2][id][0]
                if len(correspondence[correspondence[:, 1] == id]) > 0:
                    color = COLORS[correspondence[correspondence[:, 1] == id][0, 0] % len(COLORS)]
                else:
                    color = COLORS[(correspondence[:, 0].max() + 2 + id - correspondence[:, 1].min()) % len(COLORS)]
                cv2.polylines(frame2, [box], True, color, thickness=2)
                frame2 = draw_track(frame2, history[cam2][id], id=id, color=color, radius=5)
                white = draw_track(white, history[cam2][id], id=id, color=color, radius=3, camid=cam2)

            cv2.putText(white, f"{correspondence}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

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

            if vis:
                cv2.imshow(window_name, show_img)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('e'):
                    exit(0)
                elif key == ord(' '):
                    cv2.waitKey(0)
    # =====================================================================

    return '\n'.join(ret)


def analyze_homo(cam1, cam2, vid_id, video_version, correspondence, box_repr_kind, vis=False, export_video=False):

    mongo = Pymongo.Builder('localhost', 1111).set_database('tracking').set_collection(
        video_version_to_db_name[video_version]).get_product()

    list_tracks = list(mongo.collection.find({'videoid': vid_id, 'camid': {'$in': [cam1, cam2]}}))

    mongo.close()

    # just to get the frame height and width
    # TODO: get frame height and width from database/config
    vid1_path = str(list(Path(HERE/f'../../data/recordings/{video_version}/videos').glob(f'{cam1}_{("00000" + str(vid_id))[-5:]}*.avi'))[0])
    vid2_path = str(list(Path(HERE/f'../../data/recordings/{video_version}/videos').glob(f'{cam2}_{("00000" + str(vid_id))[-5:]}*.avi'))[0])
    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)
    homo = get_homo(cam1, cam2, video_version)
    roi = get_roi(cam2, video_version)

    # categorize tracks by camid
    by_camid = {cam1: {}, cam2: {}}
    for track in list_tracks:
        by_camid[track['camid']][track['trackid']] = track['detections']

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

    timestampsa = np.linspace(min_timestamp, max_timestamp, n_timestamp)

    distance_tensor = np.full((len(by_camid[cam1]), len(by_camid[cam2]), n_timestamp), 1e9, dtype='float32')
    mask_tensor = np.full_like(distance_tensor, False, dtype='bool')
    roi_tensor = np.full_like(distance_tensor, False, dtype='bool')

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

                repr_points1 = get_box_repr([box1], kind=box_repr_kind, midpoint=(
                    cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                repr_points2 = get_box_repr([box2], kind=box_repr_kind, midpoint=(
                    cap1.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                repr_points1_trans = cv2.perspectiveTransform(repr_points1.reshape(-1, 1, 2), homo).reshape(-1, 2)

                if cv2.pointPolygonTest(roi, repr_points1_trans[0], True) >= -5 and cv2.pointPolygonTest(roi, repr_points2[0], True) >= -5:
                    distance_tensor[idx1, idx2, time_idxa] = trajectory_distance(repr_points1_trans, repr_points2,
                                                                                 kind='euclid')
                    roi_tensor[idx1, idx2, time_idxa] = True
                else:
                    distance_tensor[idx1, idx2, time_idxa] = 1e9

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
    false_distances = []
    true_distances = []
    frame_skipped = False
    distance_tensor = distance_tensor[:, :, abs(frame_diff):]
    mask_tensor = mask_tensor[:, :, abs(frame_diff):]
    roi_tensor = roi_tensor[:, :, abs(frame_diff):]
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
            break

        frame1_trans = cv2.warpPerspective(frame1, homo, (frame2.shape[1], frame2.shape[0]))

        white = np.full((int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 0,
                        dtype='uint8')
        cv2.drawContours(white, [roi], -1, (255, 255, 255), 2)
        cv2.drawContours(frame1_trans, [roi], -1, (255, 255, 255), 2)
        cv2.drawContours(frame2, [roi], -1, (255, 255, 255), 2)

        for idx1, idx2, id1, id2 in zip(matched_track_indexes1, matched_track_indexes2, matched_track_ids1, matched_track_ids2):

            if mask_tensor[idx1, idx2, frame_count]:
                matches.append((id1, id2))
                count_total += 1

                c1, c1_trans, c2, p1, p1_trans, p2 = points_records[(frame_count + abs(frame_diff), id1, id2)]
                distance = np.sqrt(np.sum(np.square(p1_trans - p2)))

                if (id1, id2) not in correspondence:
                    count_false += 1
                    tf_color = (0, 0, 255)
                    false_distances.append(distance)
                else:
                    tf_color = (0, 255, 0)
                    true_distances.append(distance)


                if roi_tensor[idx1, idx2, frame_count]:
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
                    break
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

    print(f'{vid_id} -> False: {count_false}/{count_total} =', count_false/count_total)
    print(f'true distances: {true_distances}')
    print(f'false distances: {false_distances}')


def input_sct_from(txt_path, delimeter, midpoint, box_repr_kind, **kwargs):

    seq = np.loadtxt(txt_path, delimiter=delimeter)

    ls_id = np.int32(np.unique(seq[:, 1]))

    N = np.max(ls_id) + 1           # this will work for both track_id starts from 0 or 1
    T = np.int32(np.max(seq[:, 0]) + 1)     # this will work for both frame_id starts from 0 or 1

    # mark temporal visibility
    OT = np.zeros((N, T), dtype='int32')

    # spatio position
    OX = np.zeros((N, T), dtype='float32')  # cannot use np.empty due to nan value in later computation.
    OY = np.zeros((N, T), dtype='float32')
    OX_no_trans = np.zeros((N, T), dtype='float32')
    OY_no_trans = np.zeros((N, T), dtype='float32')
    OS = np.zeros((N, T, 8), dtype='float32')
    OS_no_trans = np.zeros((N, T, 8), dtype='float32')

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

        if 'roi' in kwargs and not cv2.pointPolygonTest(kwargs['roi'], (repr_x, repr_y), True) >= -5:
            continue

        OT[id, frame] = 1
        OX[id, frame] = repr_x
        OY[id, frame] = repr_y
        OS[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

    return OT, OX, OY, OS, OX_no_trans, OY_no_trans, OS_no_trans


def make_true_sct_gttracker_correspondences_v1(gt_txt_path, tracker_txt_path, midpoint, box_repr_kind, **kwargs):
    # options for kwargs: homo, roi, use_iou
    # output of this function is saved in true_sct_gttracker_correspondences.txt

    # CONCERN ON OVERLAPPING REGION ONLY
    OT, OX, OY, OS, OX_no_trans, OY_no_trans, OS_no_trans = input_sct_from(gt_txt_path, delimeter=',', midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)
    HT, HX, HY, HS, HX_no_trans, HY_no_trans, HS_no_trans = input_sct_from(tracker_txt_path, delimeter=None, midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)

    N = OT.shape[0]
    M = HT.shape[0]
    T = max(OT.shape[1], HT.shape[1])

    # pad 0 to equalize temporal dimension
    OT = np.pad(OT, ((0, 0), (0, T - OT.shape[1])), mode='constant', constant_values=0)
    OX = np.pad(OX, ((0, 0), (0, T - OX.shape[1])), mode='constant', constant_values=0)
    OY = np.pad(OY, ((0, 0), (0, T - OY.shape[1])), mode='constant', constant_values=0)
    OS = np.pad(OS, ((0, 0), (0, T - OS.shape[1]), (0, 0)), mode='constant', constant_values=0)
    OX_no_trans = np.pad(OX_no_trans, ((0, 0), (0, T - OX_no_trans.shape[1])), mode='constant', constant_values=0)
    OY_no_trans = np.pad(OY_no_trans, ((0, 0), (0, T - OY_no_trans.shape[1])), mode='constant', constant_values=0)
    OS_no_trans = np.pad(OS_no_trans, ((0, 0), (0, T - OS_no_trans.shape[1]), (0, 0)), mode='constant',
                         constant_values=0)
    HT = np.pad(HT, ((0, 0), (0, T - HT.shape[1])), mode='constant', constant_values=0)
    HX = np.pad(HX, ((0, 0), (0, T - HX.shape[1])), mode='constant', constant_values=0)
    HY = np.pad(HY, ((0, 0), (0, T - HY.shape[1])), mode='constant', constant_values=0)
    HS = np.pad(HS, ((0, 0), (0, T - HS.shape[1]), (0, 0)), mode='constant', constant_values=0)
    HX_no_trans = np.pad(HX_no_trans, ((0, 0), (0, T - HX_no_trans.shape[1])), mode='constant', constant_values=0)
    HY_no_trans = np.pad(HY_no_trans, ((0, 0), (0, T - HY_no_trans.shape[1])), mode='constant', constant_values=0)
    HS_no_trans = np.pad(HS_no_trans, ((0, 0), (0, T - HS_no_trans.shape[1]), (0, 0)), mode='constant',
                         constant_values=0)

    OH_overlap = np.zeros((N, M), dtype='bool')
    for i in range(N):
        for j in range(M):
            if np.any(OT[i] * HT[j]):
                OH_overlap[i, j] = True

    # compute cost when map Oi to Hj
    cost = np.empty((N, M), dtype='float32')
    for i in range(N):
        for j in range(M):
            # TODO xem lai cho cost va weight nay
            # if Oi and Hj have overlap
            if OH_overlap[i, j]:
                idx = np.where(OT[i] * HT[j])
                iou = sum(np.logical_and(OT[i], HT[j])) / sum(np.logical_or(OT[i], HT[j]))   # IoU is better

                if 'use_iou' not in kwargs or not kwargs['use_iou']:
                    np.mean(np.square(OX[i, idx] - HX[j, idx]) + np.square(OY[i, idx] - HY[j, idx]))
                else:
                    cost[i, j] = - np.mean(
                        [iou_batch(
                            OS[i, t, [0, 1, 4, 5]].reshape(1, 4),
                            HS[j, t, [0, 1, 4, 5]].reshape(1, 4))
                        for t in idx[0]])

                cost[i, j] /= iou
            # if Oi and Hj have no overlap
            else:
                cost[i, j] = 1e9

    solver = pywraplp.Solver.CreateSolver('SCIP')

    X = np.array([[solver.IntVar(0, 1, f'X[{i}, {j}]') for j in range(M)] for i in range(N)])

    # constraint 1: each Oi is mapped to >= 0 Hj
    # for i in range(N):
    #     solver.Add(sum(X[i, j] for j in range(M)) >= 0)

    # constraint 2: each Hj is mapped to <= 1 Oi
    for j in range(M):
        solver.Add(sum(X[i, j] for i in range(N)) <= 1)

    # # constraint 3: every Hj1 and Hj2 mapped to a same Oi must not overlap (????????)
    # for i in range(N):
    #     for k in range(M - 1):
    #         for q in range(k + 1, M):
    #             for t in range(T):
    #                     solver.Add(X[i, k] * HT[k, t] + X[i, q] * HT[q, t] <= 1)

    # constraint 4: force minimum independent Hj
    for t in range(T):
        n_O_present = np.sum(OT[:, t])
        n_H_present = np.sum(HT[:, t])
        O_present = np.where(OT[:, t])[0].tolist()
        H_present = np.where(HT[:, t])[0].tolist()
        for j in H_present:
            for i in range(N):
                if i not in O_present and OH_overlap[i, j]:
                    O_present.append(i)
                    n_O_present += 1
        if n_H_present > 0:
            solver.Add(sum(X[i, j] for i in range(N) for j in H_present) == min(n_O_present, n_H_present))

    # objective function
    # chỉ xét những cặp được map với nhau (X[i, j])
    Y = sum(X[i, j] * cost[i, j] for i in range(N) for j in range(M))
    # - 1e4 * sum(X[i, j] for i in range(N) for j in range(M)) #

    solver.Minimize(Y)

    print('Solving')
    status = solver.Solve()
    print('Done')

    if status == pywraplp.Solver.OPTIMAL:
        objective_value = solver.Objective().Value()
        optimal_solution = [(i, j) for i in range(N) for j in range(M) if X[i, j].solution_value() == 1] # {i: [j for j in range(M) if X[i, j].solution_value() == 1] for i in range(N)}
        print(optimal_solution)
        print(objective_value)
    else:
        print("INFEASIBLE")
        optimal_solution = []

    return optimal_solution


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
    OX = np.zeros((N, T), dtype='float32')  # cannot use np.empty due to nan value in later computation.
    OY = np.zeros((N, T), dtype='float32')
    OS = np.zeros((N, T, 8), dtype='float32')
    OX_no_trans = np.zeros((N, T), dtype='float32')
    OY_no_trans = np.zeros((N, T), dtype='float32')
    OS_no_trans = np.zeros((N, T, 8), dtype='float32')

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

        if 'roi' in kwargs and not cv2.pointPolygonTest(kwargs['roi'], (repr_x, repr_y), True) >= -5:
            continue

        OT[id, frame] = 1
        OX[id, frame] = repr_x
        OY[id, frame] = repr_y
        OS[id, frame] = [x1, y1, x2, y2, x3, y3, x4, y4]

    for i in range(T):
        OTT[i] = (record_time + timedelta(seconds=(i - 1) / fps)).timestamp() # frame id starts from 1

    return OT, OX, OY, OTT, OS, OX_no_trans, OY_no_trans, OS_no_trans


def map_timestamp(ATT, BTT, diff_thresh=None):
    # all params must be in seconds (not milliseconds or anything else)

    T1 = len(ATT)
    T2 = len(BTT)

    if diff_thresh is None:
        diff_thresh = float('inf')

    if not isinstance(ATT, np.ndarray):
        ATT = np.array(ATT)
    if not isinstance(BTT, np.ndarray):
        BTT = np.array(BTT)

    assert len(ATT.shape) == len(BTT.shape) == 1, 'Invalid seq dimension, must be (N,)'

    X = np.zeros((T1, T2), dtype='int32')

    valid_pairs = [(abs(ATT[i] - BTT[j]), i, j)
                   for i in range(T1)
                   for j in range(T2)
                   if abs(ATT[i] - BTT[j]) <= diff_thresh]
    valid_pairs = sorted(valid_pairs)

    def _is_crossing(i, j):

        for i_optimal, j_optimal in zip(*np.where(X)):
            if (ATT[i] - ATT[i_optimal])*(BTT[j] - BTT[j_optimal]) <= 0:
                return True

        return False

    for _, i, j in valid_pairs:
        if not np.any(X[i, :]) and not np.any(X[:, j]) and not _is_crossing(i, j):
            X[i, j] = 1

    return X


def make_true_mct_trackertracker_correspondences_v1(
        cam1_tracker_path, cam2_tracker_path,
        fps1, fps2,
        midpoint1, midpoint2,
        mct_gtgt_correspondences,
        sct_gttracker_correspondences,
        homo,
        roi,
        box_repr_kind
):

    C1T, C1X, C1Y, C1TT, _, _, _, _ = input_mct_from(cam1_tracker_path, delimeter=None, fps=fps1, midpoint=midpoint1, box_repr_kind=box_repr_kind, homo=homo, roi=roi)
    C2T, C2X, C2Y, C2TT, _, _, _, _ = input_mct_from(cam2_tracker_path, delimeter=None, fps=fps2, midpoint=midpoint2, box_repr_kind=box_repr_kind, roi=roi)

    T1 = C1T.shape[1]
    T2 = C2T.shape[1]

    time_correspondences = map_timestamp(C1TT, C2TT, diff_thresh=1)

    ret = []

    for o_c1, o_c2 in mct_gtgt_correspondences:
        o_c1_h_list = [pair[1] for pair in sct_gttracker_correspondences[0] if pair[0] == o_c1]
        o_c2_h_list = [pair[1] for pair in sct_gttracker_correspondences[1] if pair[0] == o_c2]

        for h_c1 in o_c1_h_list:
            for h_c2 in o_c2_h_list:

                h_c1_present = C1T[h_c1].reshape(-1, 1) @ np.ones((1, T2), dtype='int32')
                h_c2_present = np.ones((T1, 1), dtype='int32') @ C2T[h_c2].reshape(1, -1)
                h_c1_and_h_c2_present = h_c1_present * h_c2_present * time_correspondences
                if np.any(h_c1_and_h_c2_present):
                    ret.append((h_c1, h_c2))

    return ret



class IQRFilter:
    def __init__(self, q1=25, q2=75):
        self.q1 = q1
        self.q2 = q2

    def run(self, distances):
        p1, p2 = np.percentile(distances, [self.q1, self.q2])
        iqr = p2 - p1
        upper_bound = p2 + 1.5 * iqr
        return upper_bound


class GMMFilter:
    def __init__(self, n_components, std_coef=3):
        self.n_components = n_components
        self.std_coef = std_coef

    def run(self, distances):
        np.random.seed(24)
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
        return upper_bound


def make_true_sct_gttracker_correspondences_v2(
        gt_txt_path,
        tracker_txt_path,
        midpoint,
        filter,
        box_repr_kind,
        **kwargs):
    # options for kwargs: homo, roi, use_iou
    """n_gmm_components can be None or 1, 2, 3,... In this function, it should be 1."""

    # CONCERN ON OVERLAPPING REGION ONLY
    OT, OX, OY, OS, OX_no_trans, OY_no_trans, OS_no_trans = input_sct_from(gt_txt_path, delimeter=',', midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)
    HT, HX, HY, HS, HX_no_trans, HY_no_trans, HS_no_trans = input_sct_from(tracker_txt_path, delimeter=None, midpoint=midpoint, box_repr_kind=box_repr_kind, **kwargs)

    N = OT.shape[0]
    M = HT.shape[0]
    T = max(OT.shape[1], HT.shape[1])

    # pad 0 to equalize temporal dimension
    OT = np.pad(OT, ((0, 0), (0, T - OT.shape[1])), mode='constant', constant_values=0)
    OX = np.pad(OX, ((0, 0), (0, T - OX.shape[1])), mode='constant', constant_values=0)
    OY = np.pad(OY, ((0, 0), (0, T - OY.shape[1])), mode='constant', constant_values=0)
    HT = np.pad(HT, ((0, 0), (0, T - HT.shape[1])), mode='constant', constant_values=0)
    HX = np.pad(HX, ((0, 0), (0, T - HX.shape[1])), mode='constant', constant_values=0)
    HY = np.pad(HY, ((0, 0), (0, T - HY.shape[1])), mode='constant', constant_values=0)

    X = np.zeros((N, M, T), dtype='int32')
    distances = np.empty((N, M, T), dtype='float32')

    for t in range(T):

        O_present_list = np.where(OT[:, t])[0]
        H_present_list = np.where(HT[:, t])[0]

        cost = np.empty((len(O_present_list), len(H_present_list)), dtype='float32')
        for i, o in enumerate(O_present_list):
            for j, h in enumerate(H_present_list):
                if 'use_iou' not in kwargs or not kwargs['use_iou']:
                    cost[i, j] = np.sqrt((OX[o, t] - HX[h, t])**2 + (OY[o, t] - HY[h, t])**2)
                else:
                    ox1, oy1, ox2, oy2 = OS[o, t, [0, 1, 4, 5]]
                    hx1, hy1, hx2, hy2 = HS[h, t, [0, 1, 4, 5]]
                    if 'homo' in kwargs:
                        homo_inv = np.linalg.inv(kwargs['homo'])
                        [[[ox1, oy1]], [[ox2, oy2]]] = cv2.perspectiveTransform(
                            np.array([ox1, oy1, ox2, oy2]).reshape(-1, 1, 2),
                            homo_inv)
                        [[[hx1, hy1]], [[hx2, hy2]]] = cv2.perspectiveTransform(
                            np.array([hx1, hy1, hx2, hy2]).reshape(-1, 1, 2),
                            homo_inv)
                    cost[i, j] = - iou_batch([[ox1, oy1, ox2, oy2]], [[hx1, hy1, hx2, hy2]])

        i_matched_list, j_matched_list = linear_sum_assignment(cost)
        for i, j in zip(i_matched_list, j_matched_list):
            o = O_present_list[i]
            h = H_present_list[j]
            X[o, h, t] = 1
            distances[o, h, t] = cost[i, j]

    # filter out false matches due to missing detection boxes
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.kdeplot(distances[X == 1].flatten())
    # plt.show()
    if filter is not None:
        boundary = filter(distances[X == 1].reshape(-1, 1))
        X = np.where(distances > boundary, 0, X)

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

    X = np.zeros((N1, N2, T1), dtype='int32')   # T1 or T2 is either the same =)
    distances = np.empty((N1, N2, T1), dtype='float32')

    map_frame_time = 0
    for t1 in range(T1):     # because I chose T1 as a dim of X
        start_map_frame = time.time()
        if not np.any(time_correspondences[t1]):
            continue

        t2 = np.where(time_correspondences[t1])[0].item()

        H1_present_list = np.where(C1T[:, t1])[0]
        H2_present_list = np.where(C2T[:, t2])[0]

        cost = np.empty((len(H1_present_list), len(H2_present_list)), dtype='float32')
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

        i1_matched_list, i2_matched_list = linear_sum_assignment(cost)
        for i1, i2 in zip(i1_matched_list, i2_matched_list):
            h1 = H1_present_list[i1]
            h2 = H2_present_list[i2]
            X[h1, h2, t1] = 1
            distances[h1, h2, t1] = cost[i1, i2]
        end_map_frame = time.time()
        map_frame_time = 0.5 * map_frame_time + 0.5 * (end_map_frame - start_map_frame)
    print(f'[INFO] Mapping object per frame time: {map_frame_time:.2f}s')

    # filter out false matches due to missing detection boxes
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.kdeplot(distances[X == 1].flatten())
    # plt.show()
    start_filter_time = time.time()
    if filter is not None:
        print('Sum before filter', np.sum(X))
        boundary = filter(distances[X == 1].reshape(-1, 1))
        X = np.where(distances > boundary, 0, X)
        print('Sum after filter', np.sum(X))
    end_filter_time = time.time()

    print(f'[INFO] Filter time: {end_filter_time - start_filter_time:.2f}s')
    print(f'[INFO] Total mapping time: {end_filter_time - start_preprocess:.2f}s')

    # just to show without timestamp details
    X_notime = np.any(X, axis=2)
    X_notime = list(zip(*np.where(X_notime == 1)))

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
        cv2.drawContours(frame2, [roi], -1, (255, 255, 255), 2)
        cv2.drawContours(frame1_no_trans, [roi_inv_trans], -1, (255, 255, 255), 2)
        cv2.putText(black, f'frame {t1}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

        for h1 in range(N1):
            if C1T[h1, t1] == 0:
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
            if t2 is None or C2T[h2, t2] == 0:
                continue

            color2 = COLORS[h2 % len(COLORS)]
            cv2.polylines(frame2, [np.int32(C2S[h2, t2]).reshape(-1, 1, 2)], True, color=color2, thickness=2)
            cv2.circle(frame2, (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])), radius=6, color=color2, thickness=-1)
            cv2.circle(black, (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])), radius=6, color=color2, thickness=-1)
            cv2.putText(frame2, f'2-{h2}', (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2]) - 10),
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

                if X_eval[h1, h2, t1] > 0:
                    cv2.line(black,
                             (np.int32(C1X[h1, t1]), np.int32(C1Y[h1, t1])),
                             (np.int32(C2X[h2, t2]), np.int32(C2Y[h2, t2])),
                             color=connection_color, thickness=3)

        frame1 = cv2.warpPerspective(frame1_no_trans, homo, (W, H))

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
        C1S_no_trans, C2S_no_trans,
        time_correspondences,
        X_pred,
        X_eval,
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
                both_h2_present = ((C2T[h2a] * C2T[h2b]).reshape(1, -1) \
                                   @ time_correspondences) \
                    .reshape(-1)

                # if h2_a, h2_b co-occur when one of them is mapped to h1
                # then there is an object swap
                h1_h2a_h2b_present_and_map = np.logical_or(h1_map_h2a, h1_map_h2b) * both_h2_present
                object_swap_found = np.any(h1_h2a_h2b_present_and_map)

                if not object_swap_found:
                    continue

                print(f'{label} ID {h1} maps to {h2a} and {h2b} while they co-occur at {np.where(h1_h2a_h2b_present_and_map)[0][0] / 10}')

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
                                f'\t switched from {previous_map} (at {start_time_of_previous_map / 10}) to {current_map} (at {current_time / 10})')
                            start_time_of_previous_map = current_time

    for h1 in range(N1):
        maps_of_h1 = pairs[pairs[:, 0] == h1][:, 1]
        _process_track('CAM_1', h1, maps_of_h1, X_pred, C2T, time_correspondences.T)
    for h2 in range(N2):
        maps_of_h2 = pairs[pairs[:, 1] == h2][:, 0]
        _process_track('CAM_2', h2, maps_of_h2, np.swapaxes(X_pred, 0, 1), C1T, time_correspondences)






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
    print('F1', 2*precision*recall/(precision + recall))
    print(FP)
    print(FN)


# TODO (tomorrow):
#  1. hàm tính khoảng cách 2 track (euclid, Hausdorff, DTW)
#  2. hàm match các điểm của 2 track
#  3. hàm smooth bằng kalman/moving average cho track



if __name__ == '__main__':

    video_version = '2d_v3'
    TRACKER_NAME = 'YOLOv8l_pretrained-640-StrongSORT'
    box_repr_kind = 'bottom'

    #'''
    cam1_id = 121
    cam2_id = 127
    n_0 = 5
    # video_id = 19
    # log_file is set to None if stdout
    filter_type = 'IQR' # None, 'GMM', 'IQR'
    window_size = 1
    window_boundary = 0
    gttracker_filter = IQRFilter(25, 75).run
    if filter_type == 'GMM':
        filter = GMMFilter(n_components=2, std_coef=3).run
        # gttracker_filter = GMMFilter(n_components=1, std_coef=3).run
    elif filter_type == 'IQR':
        filter = IQRFilter(25, 75).run
        # gttracker_filter = IQRFilter(25, 75).run
    else:
        filter = None
        # gttracker_filter = None

    cf = f'{filter_type if filter_type else "noFilter"}_windowsize{window_size}_windowboundary{window_boundary}'
    log_file = open(str(HERE / f'../../data/recordings/{video_version}/{TRACKER_NAME}/log_error_analysis_pred_mct_trackertracker_correspondences_v2_{cf}.txt'), 'w')
    print(f'================= ERROR ANALYSIS FOR TRACKER {TRACKER_NAME} VIDEO VERSION {video_version} WITH{" " + filter_type if filter_type else "OUT"} FILTER, WINDOW_SIZE = {window_size}, WINDOW_BOUNDARY = {window_boundary} ================', file=log_file)

    for video_id in tqdm(range(1, 13)):

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


        ################## MAKE TRUE GT TRACKER CORRESPONDENCES V1 #############################################
        """
        ret1 = make_true_sct_gttracker_correspondences_v1(
            gt_txt1, tracker_txt1,
            midpoint1,
            box_repr_kind=box_repr_kind,
            homo=homo,
            roi=roi,
            use_iou=True
        )
        ret2 = make_true_sct_gttracker_correspondences_v1(
            gt_txt2, tracker_txt2,
            midpoint2,
            box_repr_kind=box_repr_kind,
            roi=roi,
            use_iou=True
        )
        for i, j in ret1:
            print(f'{cam1_id},{video_id},{i},{j}')
        for i, j in ret2:
            print(f'{cam2_id},{video_id},{i},{j}')
        """
        ######################################################################################################

        ################### MAKE TRUE GT TRACKER CORRESPONDENCES V2 #############################################
        ret1 = make_true_sct_gttracker_correspondences_v2(
            gt_txt1, tracker_txt1,
            midpoint1,
            filter=gttracker_filter,
            box_repr_kind=box_repr_kind,
            homo=homo,
            roi=roi,
            use_iou=True
        )
        ret2 = make_true_sct_gttracker_correspondences_v2(
            gt_txt2, tracker_txt2,
            midpoint2,
            filter=gttracker_filter,
            box_repr_kind=box_repr_kind,
            roi=roi,
            use_iou=True
        )

        """
        error_analysis_mct_mapping(
            cap1, cap1,
            homo,
            roi,
            *ret1,
            display=False,
            export_video=f'true_mct_gttracker_correspondences_{cam1_id}_{video_id}_{cf}.avi',
            checking_true_gttracker=True
        )
        error_analysis_mct_mapping(
            cap2, cap2,
            None,
            roi,
            *ret2,
            display=False,
            export_video=f'true_mct_gttracker_correspondences_{cam2_id}_{video_id}_{cf}.avi',
            checking_true_gttracker=True
        )
        for t in range(ret1[-2].shape[-1]):
            if np.any(ret1[-2][:, :, t]):
                print(t, list(zip(*np.where(ret1[-2][:, :, t]))))
        for t in range(ret2[-2].shape[-1]):
            if np.any(ret2[-2][:, :, t]):
                print(t, list(zip(*np.where(ret2[-2][:, :, t]))))
        """
        ######################################################################################################


        #################### MAKE TRUE MCT TRACKER TRACKER CORRESPONDENCES V1 ###################################
        # with open(f'../../data/recordings/{video_version}/true_mct_gtgt_correspondences.txt', 'r') as f:
        #     mct_gtgt_correspondences = f.read().strip().split('\n')
        #     mct_gtgt_correspondences = [eval(l) for l in mct_gtgt_correspondences]
        #     mct_gtgt_correspondences = [(l[2], l[5]) for l in mct_gtgt_correspondences if
        #                               l[0] == cam1_id and l[3] == cam2_id and l[1] == video_id]
        # with open(f'../../data/recordings/{video_version}/{TRACKER_NAME}/true_sct_gttracker_correspondences_v1.txt', 'r') as f:
        #     sct_gttracker_correspondences = f.read().strip().split('\n')
        #     sct_gttracker_correspondences = [eval(l) for l in sct_gttracker_correspondences]
        #     sct_gttracker_correspondences = [
        #         [(l[2], l[3]) for l in sct_gttracker_correspondences if l[0] == cam1_id and l[1] == video_id],
        #         [(l[2], l[3]) for l in sct_gttracker_correspondences if l[0] == cam2_id and l[1] == video_id]
        #     ]
        # ret = make_true_mct_trackertracker_correspondences_v1(
        #     tracker_txt1, tracker_txt2,
        #     fps1, fps2,
        #     midpoint1, midpoint2,
        #     mct_gtgt_correspondences,
        #     sct_gttracker_correspondences,
        #     homo,
        #     roi,
        #     box_repr_kind=box_repr_kind,
        # )
        # for h1, h2 in ret:
        #     print(f'{cam1_id},{video_id},{h1},{cam2_id},{video_id},{h2}')

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
        #     export_video=f'true_mct_trackertracker_correspondences_{cam1_id}_{cam2_id}_{video_id}_{cf}.avi',
        # )
        # for t in range(ret[-2].shape[-1]):
        #     if np.any(ret[-2][:, :, t]):
        #         print(t, list(zip(*np.where(ret[-2][:, :, t]))))
        ######################################################################################################

        ######################## PREDICT MCT TRACKER TRACKER CORRESPONDENCES V1 ########################################
        # # for error analysis
        # with open(f'../../data/recordings/{video_version}/{TRACKER_NAME}/true_mct_trackertracker_correspondences.txt', 'r') as ff:
        #     true_mct_trackertracker_correspondences = ff.read().strip().split('\n')
        #     true_mct_trackertracker_correspondences = [eval(l) for l in true_mct_trackertracker_correspondences]
        #     true_mct_trackertracker_correspondences = [(l[2], l[5]) for l in true_mct_trackertracker_correspondences if
        #                                                l[0] == cam1_id and l[3] == cam2_id and l[1] == video_id]
        #
        # ret = mct_mapping(
        #     tracker_txt1, tracker_txt2,
        #     fps1, fps2,
        #     midpoint1, midpoint2,
        #     homo,
        #     roi,
        #     box_repr_kind=box_repr_kind,
        #     true_mct_trackertracker_correspondences=true_mct_trackertracker_correspondences, # COMMENT OUT IF NOT ERROR ANALYSIS
        # )
        #
        # # COMMENT OUT IF NOT ERROR ANALYSIS
        # error_analysis_mct_mapping(
        #     cap1, cap2,
        #     homo,
        #     roi,
        #     *ret,
        #     display=False,
        #     export_video=f'pred_mct_trackertracker_correspondences_{cam1_id}_{cam2_id}_{video_id}_{cf}.avi'
        # )
        ##############################################################################################################

        ######################## PREDICT MCT TRACKER TRACKER CORRESPONDENCES V2 ########################################
        print(f'\n\n[INFO]\t VIDEO {video_id} PREDICT MCT TRACKER TRACKER CORRESPONDENCES V2', file=log_file)
        ret = mct_mapping(
            tracker_txt1, tracker_txt2,
            fps1, fps2,
            midpoint1, midpoint2,
            homo,
            roi,
            filter=filter,
            box_repr_kind=box_repr_kind,
            window_size=window_size,
            window_boundary=window_boundary,
            true_mct_trackertracker_correspondences=ret[-1], # COMMENT OUT IF NOT ERROR ANALYSIS OR DETECT IDSW
        )
        # COMMENT OUT IF NOT ERROR ANALYSIS
        error_analysis_mct_mapping(
            cap1, cap2,
            homo,
            roi,
            *ret,
            display=False,
            export_video=None, #f'pred_mct_trackertracker_correspondences_{cam1_id}_{cam2_id}_{video_id}_{cf}.avi', # None
            log_file=log_file
        )

        # for t in range(ret.shape[2]):
        #     print(list(zip(*np.where(ret[:, :, t]))))

        # detect_IDSW(*ret)

        ##############################################################################################################

    # f.close()
    # '''
    # evaluate(f'../../data/recordings/{video_version}/{TRACKER_NAME}/true_mct_trackertracker_correspondences.txt',
    #          f'../../data/recordings/{video_version}/{TRACKER_NAME}/pred_mct_trackertracker_correspondences.txt')



