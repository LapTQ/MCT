import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import time
import sys
import logging
from pipeline import main
import yaml
from map_utils import map_mono, hungarian
from datetime import datetime
from vis_utils import plot_box, plot_loc, plot_roi, plot_skeleton_kpts
from general import load_roi, load_homo

sys.path.append(sys.path[0] + '/../..')

HERE = Path(__file__).parent


def make_pseudotrue_mct_trackertracker(
        true_mct_gtgt_path,
        meta_path_1,
        meta_path_2,
        pseudotrue_sct_gttracker_path_1,
        pseudotrue_sct_gttracker_path_2,
        out_path
):
    with open(meta_path_1, 'r') as f:
        meta_1 = yaml.safe_load(f)
    with open(meta_path_2, 'r') as f:
        meta_2 = yaml.safe_load(f)

    cam_id_1 = meta_1['cam_id']
    cam_id_2 = meta_2['cam_id']
    video_id_1 = meta_1['video_id']
    video_id_2 = meta_2['video_id']
    
    with open(true_mct_gtgt_path, 'r') as f:
        true_mct_gtgt = f.read().strip().split('\n')
        true_mct_gtgt = [eval(i) for i in true_mct_gtgt]
        true_mct_gtgt = np.array(true_mct_gtgt)
        true_mct_gtgt = true_mct_gtgt[np.all(true_mct_gtgt[:, [0, 1, 3, 4]] == [cam_id_1, video_id_1, cam_id_2, video_id_2], axis=1)]
    
    with open(pseudotrue_sct_gttracker_path_1, 'r') as f:
        pseudotrue_sct_gttracker_1 = f.read().strip().split('\n')
        pseudotrue_sct_gttracker_1 = [eval(i) for i in pseudotrue_sct_gttracker_1]
        pseudotrue_sct_gttracker_1 = {
            i['frame_id_1']: {
                'locs': i['locs'],
                'locs_in_roi': i['locs_in_roi'],
                'matches': {gt: tracker for gt, tracker in i['matches']}
            }
            for i in pseudotrue_sct_gttracker_1
        }
    
    with open(pseudotrue_sct_gttracker_path_2, 'r') as f:
        pseudotrue_sct_gttracker_2 = f.read().strip().split('\n')
        pseudotrue_sct_gttracker_2 = [eval(i) for i in pseudotrue_sct_gttracker_2]
        pseudotrue_sct_gttracker_2 = {
            i['frame_id_1']: {
                'locs': i['locs'],
                'locs_in_roi': i['locs_in_roi'],
                'matches': {gt: tracker for gt, tracker in i['matches']}
            }
            for i in pseudotrue_sct_gttracker_2
        }

    frame_ids_1 = np.arange(meta_1['start_frame_id'], meta_1['start_frame_id'] + meta_1['frame_count'])
    frame_ids_2 = np.arange(meta_2['start_frame_id'], meta_2['start_frame_id'] + meta_2['frame_count'])

    frame_times_1 = [datetime.strptime(meta_1['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (i - meta_1['start_frame_id']) / meta_1['fps'] for i in frame_ids_1]
    frame_times_2 = [datetime.strptime(meta_2['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (i - meta_2['start_frame_id']) / meta_2['fps'] for i in frame_ids_2]

    idx1, idx2 = map_mono(frame_times_1, frame_times_2, diff_thresh=0.1)
    frame_ids_1 = np.array(frame_ids_1)[idx1]
    frame_ids_2 = np.array(frame_ids_2)[idx2]

    parent, _ = os.path.split(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    f = open(out_path, 'w')
    
    for i, (f1, f2) in enumerate(zip(frame_ids_1, frame_ids_2)):

        out_item = {
            'frame_id_1': f1,
            'frame_id_2': f2,
            'locs': (pseudotrue_sct_gttracker_1[f1]['locs'][1], pseudotrue_sct_gttracker_2[f2]['locs'][1]),
            'locs_in_roi': (pseudotrue_sct_gttracker_1[f1]['locs_in_roi'][1], pseudotrue_sct_gttracker_2[f2]['locs_in_roi'][1]),
            'matches': []
        }
        for gt_id1, tracker_id1 in pseudotrue_sct_gttracker_1[f1]['matches'].items():
            for gt_id2, tracker_id2 in pseudotrue_sct_gttracker_2[f2]['matches'].items():
                if np.squeeze(true_mct_gtgt[true_mct_gtgt[:, 2] == gt_id1])[5] == gt_id2:
                    out_item['matches'].append([tracker_id1, tracker_id2])

        if i > 0:
            f.write('\n')
        f.write(str(out_item))

    f.close()


def validate_pred_mct_trackertracker(
        pseudotrue_mct_trackertracker_path,
        pred_mct_trackertracker_path,
        out_path
):
    with open(pseudotrue_mct_trackertracker_path, 'r') as f:
        pseudotrue = f.read().strip().split('\n')
        pseudotrue = [eval(i) for i in pseudotrue]

    with open(pred_mct_trackertracker_path, 'r') as f:
        pred = f.read().strip().split('\n')
        pred = [eval(i) for i in pred]

    parent, _ = os.path.split(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    f = open(out_path, 'w')

    for i, (T, P) in enumerate(zip(pseudotrue, pred)):
        assert T['frame_id_1'] == P['frame_id_1'] and T['frame_id_2'] == P['frame_id_2']

        out_item = {
            'frame_id_1': P['frame_id_1'],
            'frame_id_2': P['frame_id_2'],
            'locs': P['locs'],
            'locs_in_roi': P['locs_in_roi'],
            'matches': {
                'TP': [],
                'FP': [],
                'FN': []
            }
        }

        t = set((id1, id2) for id1, id2 in T['matches'])
        p = set((id1, id2) for id1, id2 in P['matches'])

        out_item['matches']['TP'].extend(list(t.intersection(p)))
        out_item['matches']['FP'].extend(list(p.difference(t)))
        out_item['matches']['FN'].extend(list(t.difference(p)))

        if i > 0:
            f.write('\n')
        f.write(str(out_item))


def prf(
        paths,
        out_path
):
    
    parent, _ = os.path.split(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    f = open(out_path, 'w')

    STP = 0
    SFP = 0
    SFN = 0

    for i, (validate_path, msg) in enumerate(paths):

        if i > 0:
            f.write('\n\n\n')

        f.write(f'==================== {msg} ===================\n')

        TP = 0
        FP = 0
        FN = 0

        with open(validate_path, 'r') as ff:
            result = ff.read().strip().split('\n')
            result = [eval(i) for i in result]

        for P in result:
            TP += len(P['matches']['TP'])
            FP += len(P['matches']['FP'])
            FN += len(P['matches']['FN'])
        
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * pre * rec / (pre + rec)

        f.write(f'TP: {TP}\n')
        f.write(f'FP: {FP}\n')
        f.write(f'FN: {FN}\n')
        f.write(f'Pre: {pre}\n')
        f.write(f'Rec: {rec}\n')
        f.write(f'F1: {F1}\n')

        STP += TP
        SFP += FP
        SFN += FN
    
    f.write('\n\n\n============== IN TOTAL ==============\n')

    pre = STP / (STP + SFP)
    rec = STP / (STP + SFN)
    F1 = 2 * pre * rec / (pre + rec)
            
    f.write(f'TP: {STP}\n')
    f.write(f'FP: {SFP}\n')
    f.write(f'FN: {SFN}\n')
    f.write(f'Pre: {pre}\n')
    f.write(f'Rec: {rec}\n')
    f.write(f'F1: {F1}\n')

    f.close()


def visualize_sta_result(
        cam_1_path,
        cam_2_path,
        sct_1_path,
        sct_2_path,
        validate_sta_path,
        roi_path,
        matches_path,
        out_path,
        mode # 'box', 'pose'
):
    cap_1 = cv2.VideoCapture(cam_1_path)
    cap_2 = cv2.VideoCapture(cam_2_path)

    fps = cap_1.get(cv2.CAP_PROP_FPS)

    sct_1 = np.loadtxt(sct_1_path)
    sct_2 = np.loadtxt(sct_2_path)

    with open(validate_sta_path, 'r') as f:
        sta = f.read().strip().split('\n')
        sta = [eval(i) for i in sta]

    H, W = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_2 = load_roi(roi_path, W, H)
    homo = load_homo(matches_path)
    homo_inv = np.linalg.inv(homo)
    roi_1 = cv2.perspectiveTransform(roi_2, homo_inv)

    parent, _ = os.path.split(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    writer_created = False
    
    fid_1 = 1
    fid_2 = 1
    p = 0
    while True:

        if fid_1 > sta[p]['frame_id_1'] or fid_2 > sta[p]['frame_id_2']:
            p += 1
            continue
        elif fid_1 < sta[p]['frame_id_1']:
            fid_1 += 1
            cap_1.read()
        elif fid_2 < sta[p]['frame_id_2']:
            fid_2 += 1
            cap_2.read()
        
        ret_1, fim_1 = cap_1.read()
        if not ret_1:
            break

        ret_2, fim_2 = cap_2.read()
        if not ret_2:
            break

        if not writer_created:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (2*W, 2*H)
            )
            writer_created = True

        if 1057 <= p <= 1156 or 1662 <= p <= 1895 or 2937 <= p <= 3095:
        
            black = np.zeros_like(fim_2)
            
            fim_1 = plot_roi(fim_1, roi_1)
            fim_2 = plot_roi(fim_2, roi_2)
            black = plot_roi(black, roi_2)

            dets_1 = sct_1[sct_1[:, 0] == fid_1]
            dets_2 = sct_2[sct_2[:, 0] == fid_2]

            fim_1 = plot_box(fim_1, dets_1)
            fim_2 = plot_box(fim_2, dets_2)

            if mode == 'pose':
                kpts_1 = dets_1[:, 10:]
                kpts_2 = dets_2[:, 10:]

                for kpt in kpts_1:
                    fim_1 = plot_skeleton_kpts(fim_1, kpt.T, 3)
                for kpt in kpts_2:
                    fim_2 = plot_skeleton_kpts(fim_2, kpt.T, 3)
            
            locs_1 = np.array([[-1, k, x, y] for k, (x, y) in sta[p]['locs'][0].items()]).reshape(-1, 4)    # type: ignore
            locs_2 = np.array([[-1, k, x, y] for k, (x, y) in sta[p]['locs'][1].items()]).reshape(-1, 4)    # type: ignore
            if len(locs_1) == 0:
                locs_1_ori = locs_1
            else:
                locs_1_ori = np.concatenate([locs_1[:, :2], cv2.perspectiveTransform(locs_1[:, 2:].reshape(-1, 1, 2), homo_inv).reshape(-1, 2)], axis=1)
            texts_1 = [f'{k} (0)' for k in sta[p]['locs'][0]]
            texts_2 = [f'{k} (1)' for k in sta[p]['locs'][1]]
            fim_1 = plot_loc(fim_1, locs_1_ori)
            fim_2 = plot_loc(fim_2, locs_2)
            black = plot_loc(black, locs_1, texts=texts_1)
            black = plot_loc(black, locs_2, texts=texts_2)

            for k, v in sta[p]['matches'].items():
                if k == 'TP':
                    color = (0, 255, 0)
                elif k == 'FP':
                    color = (0, 0, 255)
                else:
                    color = (11, 185, 255)
                for id_1, id_2 in v:
                    cv2.line(
                        black,
                        np.int32(sta[p]['locs'][0][id_1]),
                        np.int32(sta[p]['locs'][1][id_2]),
                        color=color,
                        thickness=3
                    )

            fim_1 = cv2.resize(fim_1, (W, H))
            collage = np.concatenate(
                [
                    np.concatenate([fim_1, fim_2], axis=1),
                    np.concatenate([np.zeros_like(black), black], axis=1)
                ],
                axis=0
            )

            if writer_created:
                writer.write(collage)   # type: ignore

        fid_1 += 1
        fid_2 += 1
        p += 1

        if p == len(sta):
            break
    
    if writer_created:
        writer.release()    # type: ignore


def frame2track(
        cam1_id,
        cam2_id,
        video_id,
        pred_mct_trackertracker_path,
        validate_pred_mct_trackertracker_path
    ):

    with open(pred_mct_trackertracker_path, 'r') as f:
        pred = f.read().strip().split('\n')
        pred = [eval(i) for i in pred]

    with open(validate_pred_mct_trackertracker_path, 'r') as f:
        true = f.read().strip().split('\n') 
        true = [eval(i) for i in true]

    matches = {}
    ID1s = []
    ID2s = []
    co_occurs = ([], [])
    for item in pred:
        for m in item['matches']:
            m = tuple(m)
            matches[m] = matches.get(m, 0) + 1

            if m[0] not in ID1s:
                ID1s.append(m[0])
            if m[1] not in ID2s:
                ID2s.append(m[1])

            for i in range(2):
                ids = list(item['locs'][i].keys())
                for id1 in ids:
                    for id2 in ids:
                        if id1 < id2 and (id1, id2) not in co_occurs[i]:
                            co_occurs[i].append((id1, id2))
    
    for id in ID1s:
        sub_m = {}
        for m in matches:
            if m[0] == id:
                sub_m[m[1]] = matches[m]
        for id1 in sub_m:
            for id2 in sub_m:
                if id1 < id2 and (id1, id2) in co_occurs[1]:
                    id3 = id1 if sub_m[id1] < sub_m[id2] else id2
                    matches[(id, id3)] = -1
    
    for id in ID2s:
        sub_m = {}
        for m in matches:
            if m[1] == id:
                sub_m[m[0]] = matches[m]
        for id1 in sub_m:
            for id2 in sub_m:
                if id1 < id2 and (id1, id2) in co_occurs[0]:
                    id3 = id1 if sub_m[id1] < sub_m[id2] else id2
                    matches[(id3, id)] = -1

    # print(matches)
    matches = [k for k, v in matches.items() if v > 0]
    # for k in matches:
    #     print(f'{cam1_id},{video_id},{k[0]},{cam2_id},{video_id},{k[1]}')

    true_matches = []
    for item in true:
        mm = item['matches']['TP'] + item['matches']['FN']
        for m in mm:
            m = tuple(m)
            if m not in true_matches:
                true_matches.append(m)
    
    print(true_matches)

    matches = set(matches)
    true_matches = set(true_matches)
    TP = matches.intersection(true_matches)
    FP = matches.difference(true_matches)
    FN = true_matches.difference(matches)
    print(f'TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}')


def eval_reid(
        cam1_id,
        cam2_id,
        video_id,
        pred_reid_path,
        validate_pred_mct_trackertracker_path
):
    with open(validate_pred_mct_trackertracker_path, 'r') as f:
        true = f.read().strip().split('\n') 
        true = [eval(i) for i in true]

    true_matches = []
    for item in true:
        mm = item['matches']['TP'] + item['matches']['FN']
        for m in mm:
            m = tuple(m)
            if m not in true_matches:
                true_matches.append(m)
    
    # print(true_matches)

    with open(pred_reid_path, 'r') as f:
        pred = f.read().strip().split('\n')[1:]
        pred = [i.split(',') for i in pred]
        pred = [(str(i[0]), int(i[1]), str(i[2])) for i in pred]
        pred = [i for i in pred if i[0] == f'CAM_{cam1_id}' or i[0] == f'CAM_{cam2_id}']
    
    matches = {}
    for item in pred:
        if item[2] not in matches:
            matches[item[2]] = []
        matches[item[2]].append((int(item[0][4:]), item[1]))
    # print(matches)

    pred_matches = []
    for k, v in matches.items():
        for i in v:
            for j in v:
                if i[0] == cam1_id and j[0] == cam2_id:
                    pred_matches.append((i[1], j[1]))
    
    print(pred_matches)

    matches = set(pred_matches)
    true_matches = set(true_matches)
    TP = matches.intersection(true_matches)
    FP = matches.difference(true_matches)
    FN = true_matches.difference(matches)
    print(f'TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}')
            

    



if __name__ == '__main__':

    VIDEO_SET = {
        '2d_v1': {'cam_id1': 21, 'cam_id2': 27, 'range_': range(0, 16)},
        '2d_v2': {'cam_id1': 21, 'cam_id2': 27, 'range_': range(19, 25)},
        '2d_v3': {'cam_id1': 121, 'cam_id2': 127, 'range_': range(1, 13)},
        '2d_v4': {'cam_id1': 41, 'cam_id2': 42, 'range_': range(1, 13)},
        #'2d_v4': {'cam_id1': 42, 'cam_id2': 43, 'range_': range(1, 13)},
    }
    for video_set in VIDEO_SET:
        VIDEO_SET[video_set]['video_set_dir'] = str(HERE / '../../data/recordings' / video_set)

    TRACKER_SET = [
        'YOLOv5l_pretrained-640-ByteTrack',
        'YOLOv8l_pretrained-640-ByteTrack',
        'YOLOv8l_pretrained-640-StrongSORT',
        'YOLOv8m_pretrained-640-ByteTrack',
        'YOLOv8s_pretrained-640-ByteTrack',
        'YOLOXl_pretrained-640-ByteTrack',
        'YOLOXm_pretrained-640-ByteTrack',
        'YOLOXs_pretrained-640-ByteTrack',
        'YOLOv7pose_pretrained-640-ByteTrack',
        'YOLOv7box_pretrained-640-ByteTrack',
        'YOLOv7pose_pretrained-640-ByteTrack-IDfixed',
        'YOLOv7box_pretrained-640-ByteTrack-IDfixed'
    ]
    

    video_set = '2d_v4'
    tracker_name = 'YOLOv7pose_pretrained-640-ByteTrack-IDfixed'
    for config_pred_option in [18]:
    

    
        video_set_dir = VIDEO_SET[video_set]['video_set_dir']
        cam1_id = VIDEO_SET[video_set]['cam_id1']
        cam2_id = VIDEO_SET[video_set]['cam_id2']
        range_ = VIDEO_SET[video_set]['range_']

        true_mct_gtgt_path = str(Path(video_set_dir) / ('true_mct_gtgt.txt' if 'pose' not in tracker_name else 'true_mct_gtgt_pose.txt'))
        config_true_path = str(Path(video_set_dir) / tracker_name / 'config_pseudotrue_sct_gttracker.yaml')
        config_pred_path = str(Path(video_set_dir) / tracker_name / f'config_pred_mct_trackertracker_{config_pred_option}.yaml')
        out_eval_path = str(Path(video_set_dir) / tracker_name / 'pred' / f'{config_pred_option}_eval_{cam1_id}_{cam2_id}.txt')

        paths = []

        for video_id in tqdm(range_):

            ################# retrieve paths #############
            vid1_path = list((Path(video_set_dir) / 'videos').glob(f"{cam1_id}_{('00000' + str(video_id))[-5:]}_*_*.avi"))
            assert len(vid1_path) == 1
            vid1_path = str(vid1_path[0])
            vid1_name = os.path.split(vid1_path)[1]
            vid1_basename = os.path.splitext(vid1_name)[0]
            vid2_path = list((Path(video_set_dir) / 'videos').glob(f"{cam2_id}_{('00000' + str(video_id))[-5:]}_*_*.avi"))
            assert len(vid2_path) == 1
            vid2_path = str(vid2_path[0])
            vid2_name = os.path.split(vid2_path)[1]
            vid2_basename = os.path.splitext(vid2_name)[0]
            
            gt_txt1_path = str(Path(video_set_dir) / ('gt' if 'pose' not in tracker_name else 'gt_pose') / (vid1_basename + '.txt'))
            gt_txt2_path = str(Path(video_set_dir) / ('gt' if 'pose' not in tracker_name else 'gt_pose') / (vid2_basename + '.txt'))
            tracker_txt1_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid1_basename + '.txt'))
            tracker_txt2_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid2_basename + '.txt'))

            meta1_path = str(Path(video_set_dir) / 'meta' / (vid1_basename + '.yaml'))
            meta2_path = str(Path(video_set_dir) / 'meta' / (vid2_basename + '.yaml'))

            roi_path = str(Path(video_set_dir) / f'roi_{cam2_id}.txt')
            matches_path = str(Path(video_set_dir) / f'matches_{cam1_id}_to_{cam2_id}.txt')

            
            ###############################################

            ################# make ground truth ##################
            # make pseudotrue sct gt-tracker
            out_pseudotrue_sct_gttracker1_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'sct_gttracker' / f'{cam1_id}_{video_id}.txt')
            # main({
            #     'config': config_true_path,
            
            #     'meta_1': meta1_path,
            #     'meta_2': meta1_path,
            #     'camera_1': vid1_path,
            #     'camera_2': vid1_path,
                
            #     'sct_1': gt_txt1_path,
            #     'sct_2': tracker_txt1_path,

            #     'roi': roi_path,
            #     'matches': matches_path,

            #     'out_sta_txt': out_pseudotrue_sct_gttracker1_path,
            #     'out_sct_vid_1': None,
            #     'out_sct_vid_2': None,
            #     'out_sta_vid': None
            # })

            out_pseudotrue_sct_gttracker2_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'sct_gttracker' / f'{cam2_id}_{video_id}.txt')
            # main({
            #     'config': config_true_path,
            
            #     'meta_1': meta2_path,
            #     'meta_2': meta2_path,
            #     'camera_1': vid2_path,
            #     'camera_2': vid2_path,
                
            #     'sct_1': gt_txt2_path,
            #     'sct_2': tracker_txt2_path,

            #     'roi': roi_path,
            #     'matches': None,

            #     'out_sta_txt': out_pseudotrue_sct_gttracker2_path,
            #     'out_sct_vid_1': None,
            #     'out_sct_vid_2': None,
            #     'out_sta_vid': None
            # })

            # make pseudotrue mct tracker-tracker
            out_pseudotrue_mct_trackertracker_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'mct_trackertracker' / f'{cam1_id}_{cam2_id}_{video_id}.txt')
            # make_pseudotrue_mct_trackertracker(
            #     true_mct_gtgt_path,
            #     meta1_path,
            #     meta2_path,
            #     out_pseudotrue_sct_gttracker1_path,
            #     out_pseudotrue_sct_gttracker2_path,
            #     out_pseudotrue_mct_trackertracker_path
            # )

            # predict mct tracker-tracker
            out_pred_mct_trackertracker_path = str(Path(video_set_dir) / tracker_name / 'pred' / f'{config_pred_option}' / f'{cam1_id}_{cam2_id}_{video_id}.txt')
            # main({
            #     'config': config_pred_path,
            
            #     'meta_1': meta1_path,
            #     'meta_2': meta2_path,
            #     'camera_1': vid1_path,
            #     'camera_2': vid2_path,
                
            #     'sct_1': tracker_txt1_path,
            #     'sct_2': tracker_txt2_path,

            #     'roi': roi_path,
            #     'matches': matches_path,

            #     'out_sta_txt': out_pred_mct_trackertracker_path,
            #     'out_sct_vid_1': None,
            #     'out_sct_vid_2': None,
            #     'out_sta_vid': None
            # })

            # find TP, FP, FN
            out_validate_pred_mct_trackertracker_path = str(Path(video_set_dir) / tracker_name / 'pred' / f'{config_pred_option}_val' / f'{cam1_id}_{cam2_id}_{video_id}.txt')
            # validate_pred_mct_trackertracker(
            #     out_pseudotrue_mct_trackertracker_path,
            #     out_pred_mct_trackertracker_path,
            #     out_validate_pred_mct_trackertracker_path
            # )

            # export video
            out_video_path = str(Path(video_set_dir) / tracker_name / 'pred' / f'{config_pred_option}_val' / f'{cam1_id}_{cam2_id}_{video_id}.avi')
            # visualize_sta_result(
            #     vid1_path,
            #     vid2_path,
            #     tracker_txt1_path,
            #     tracker_txt2_path,
            #     out_validate_pred_mct_trackertracker_path,
            #     roi_path,
            #     matches_path,
            #     out_video_path,
            #     mode='box' if 'pose' not in tracker_name else 'pose'
            # )

            # frame2track(
            #     cam1_id,
            #     cam2_id,
            #     video_id,
            #     out_pred_mct_trackertracker_path, 
            #     out_validate_pred_mct_trackertracker_path
            # )

            pred_reid_path = str(Path(video_set_dir) / 'Re-ID' / vid1_name[3:-4] / 'all_cams_reid.txt')
            eval_reid(cam1_id, cam2_id, video_id, pred_reid_path, out_validate_pred_mct_trackertracker_path)

            paths.append([out_validate_pred_mct_trackertracker_path, f'CAM_ID_1 = {cam1_id}, CAM_ID_2 = {cam2_id}, VIDEO_ID = {video_id}, CONFIG = {config_pred_option}, TIME = {datetime.now()}'])

        # prf(paths, out_path=out_eval_path)


# cuts:
# 2: 1108 <= p <= 1235 or 1906 <= p <= 2000
# 8: 390 <= p <= 570 or 1033 <= p <= 1127 or 1532 <= p <= 1771 or 1911 <= p <= 2187
# 9: 1945 <= p <= 2110 or 2352 <= p <= 2472
# 10: 1018 <= p <= 1126 or 1538 <= p <= 1683 or 2475 <= p <= 2600
# 11: 1084 <= p <= 1149 or 1703 <= p <= 1917 or 2137 <= p <= 2310
# 12: 1057 <= p <= 1156 or 1662 <= p <= 1895 or 2937 <= p <= 3095
