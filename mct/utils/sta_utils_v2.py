import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import time
import sys
import logging
from pipeline import main as main

sys.path.append(sys.path[0] + '/../..')

HERE = Path(__file__).parent


def 


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
    config_pred_option = 1

    
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

        true_mct_gtgt_path = str(Path(video_set_dir) / 'true_mct_gtgt_correspondences.txt')

        config_true_path = str(Path(video_set_dir) / tracker_name / 'config_true.yaml')
        config_pred_path = str(Path(video_set_dir) / tracker_name / f'config_pred_{config_pred_option}.yaml')
        ###############################################

        ################# make ground truth ##################
        out_true_sct_gttracker1_path = str(Path(video_set_dir) / tracker_name / 'true' / 'sct_gttracker' / f'{cam1_id}_{video_id}.txt')
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

            'out_sta_txt': out_true_sct_gttracker1_path,
            'out_sct_vid_1': None,
            'out_sct_vid_2': None,
            'out_sta_vid': None
        })

        out_true_sct_gttracker2_path = str(Path(video_set_dir) / tracker_name / 'true' / 'sct_gttracker' / f'{cam2_id}_{video_id}.txt')
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

            'out_sta_txt': out_true_sct_gttracker2_path,
            'out_sct_vid_1': None,
            'out_sct_vid_2': None,
            'out_sta_vid': None
        })







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
