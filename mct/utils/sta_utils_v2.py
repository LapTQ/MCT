import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import time
import sys
import logging
from pipeline import main as main
import yaml
from map_utils import map_mono
from datetime import datetime

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

    frame_times_1 = np.array([datetime.strptime(meta_1['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (i - meta_1['start_frame_id']) / meta_1['fps'] for i in frame_ids_1])
    frame_times_2 = np.array([datetime.strptime(meta_2['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (i - meta_2['start_frame_id']) / meta_2['fps'] for i in frame_ids_2])

    idx1, idx2 = map_mono(frame_times_1, frame_times_2, diff_thresh=0.1)
    frame_ids_1 = frame_ids_1[idx1]
    frame_ids_2 = frame_ids_2[idx2]

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



    



if __name__ == '__main__':

    VIDEO_SET = {
        '2d_v1': {'cam_id1': 21, 'cam_id2': 27, 'range_': range(0, 16)},
        '2d_v2': {'cam_id1': 21, 'cam_id2': 27, 'range_': range(19, 25)},
        '2d_v3': {'cam_id1': 121, 'cam_id2': 127, 'range_': range(1, 13)},
        #'2d_v4': {'cam_id1': 41, 'cam_id2': 42, 'range_': range(1, 13)},
        '2d_v4': {'cam_id1': 42, 'cam_id2': 43, 'range_': range(1, 13)},
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
        'YOLOv7pose_pretrained-640-ByteTrack'
    ]
    

    video_set = '2d_v3'
    tracker_name = 'YOLOv8l_pretrained-640-ByteTrack'
    config_pred_option = 0

    
    video_set_dir = VIDEO_SET[video_set]['video_set_dir']
    cam1_id = VIDEO_SET[video_set]['cam_id1']
    cam2_id = VIDEO_SET[video_set]['cam_id2']
    range_ = VIDEO_SET[video_set]['range_']

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
        
        gt_txt1_path = str(Path(video_set_dir) / 'gt' / (vid1_basename + '.txt'))
        gt_txt2_path = str(Path(video_set_dir) / 'gt' / (vid2_basename + '.txt'))
        tracker_txt1_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid1_basename + '.txt'))
        tracker_txt2_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid2_basename + '.txt'))

        meta1_path = str(Path(video_set_dir) / 'meta' / (vid1_basename + '.yaml'))
        meta2_path = str(Path(video_set_dir) / 'meta' / (vid2_basename + '.yaml'))

        roi_path = str(Path(video_set_dir) / f'roi_{cam2_id}.txt')
        matches_path = str(Path(video_set_dir) / f'matches_{cam1_id}_to_{cam2_id}.txt')

        true_mct_gtgt_path = str(Path(video_set_dir) / 'true_mct_gtgt.txt')
        
        config_true_path = str(Path(video_set_dir) / tracker_name / 'config_pseudotrue_sct_gttracker.yaml')
        config_pred_path = str(Path(video_set_dir) / tracker_name / f'config_pred_mct_trackertracker_{config_pred_option}.yaml')
        ###############################################

        ################# make ground truth ##################
        # make pseudotrue sct gt-tracker
        out_pseudotrue_sct_gttracker1_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'sct_gttracker' / f'{cam1_id}_{video_id}.txt')
        main({
            'config': config_true_path,
        
            'meta_1': meta1_path,
            'meta_2': meta1_path,
            'camera_1': vid1_path,
            'camera_2': vid1_path,
            
            'sct_1': gt_txt1_path,
            'sct_2': tracker_txt1_path,

            'roi': roi_path,
            'matches': matches_path,

            'out_sta_txt': out_pseudotrue_sct_gttracker1_path,
            'out_sct_vid_1': None,
            'out_sct_vid_2': None,
            'out_sta_vid': None
        })

        out_pseudotrue_sct_gttracker2_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'sct_gttracker' / f'{cam2_id}_{video_id}.txt')
        main({
            'config': config_true_path,
        
            'meta_1': meta2_path,
            'meta_2': meta2_path,
            'camera_1': vid2_path,
            'camera_2': vid2_path,
            
            'sct_1': gt_txt2_path,
            'sct_2': tracker_txt2_path,

            'roi': roi_path,
            'matches': None,

            'out_sta_txt': out_pseudotrue_sct_gttracker2_path,
            'out_sct_vid_1': None,
            'out_sct_vid_2': None,
            'out_sta_vid': None
        })

        # make pseudotrue mct tracker-tracker
        out_pseudotrue_mct_trackertracker_path = str(Path(video_set_dir) / tracker_name / 'pseudotrue' / 'mct_trackertracker' / f'{cam1_id}_{cam2_id}_{video_id}.txt')
        make_pseudotrue_mct_trackertracker(
            true_mct_gtgt_path,
            meta1_path,
            meta2_path,
            out_pseudotrue_sct_gttracker1_path,
            out_pseudotrue_sct_gttracker2_path,
            out_pseudotrue_mct_trackertracker_path
        )

        # predict mct tracker-tracker
        out_pred_mct_trackertracker_path = str(Path(video_set_dir) / tracker_name / 'pred' / f'{config_pred_option}' / f'{cam1_id}_{cam2_id}_{video_id}.txt')
        main({
            'config': config_pred_path,
        
            'meta_1': meta1_path,
            'meta_2': meta2_path,
            'camera_1': vid1_path,
            'camera_2': vid2_path,
            
            'sct_1': tracker_txt1_path,
            'sct_2': tracker_txt2_path,

            'roi': roi_path,
            'matches': matches_path,

            'out_sta_txt': out_pred_mct_trackertracker_path,
            'out_sct_vid_1': None,
            'out_sct_vid_2': None,
            'out_sta_vid': None
        })



        ###############################################
