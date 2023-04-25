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
import logging

sys.path.append(sys.path[0] + '/../..')

HERE = Path(__file__).parent


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t|%(levelname)s\t|%(funcName)s\t|%(lineno)d\t|%(message)s',
    handlers=[
        logging.FileHandler("~/Downloads/log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)


"""
# online
queue_1 = Queue() # input of SCT (frame_img, frame_id, frame_time)
queue_2 = Queue() # input of SCT
queue_5 = Queue() # input of Visualize
queue_6 = Queue() # input of Visualize

cap_1 = Camera(1, <meta,> [queue_1, queue_5])
cap_2 = Camera(2, <meta,> [queue_2, queue_6])

queue_3 = Queue() # input of STA (frame_id, frame_time, boxes)
queue_4 = Queue() # input of STA
queue_7 = Queue() # input of Visualize
queue_8 = Queue() # input of Visualize

sct_1 = SCT(queue_1, [queue_3, queue_7])
sct_2 = SCT(queue_2, [queue_4, queue_8])

queue_9 = Queue() # output of STA (frame_id_1, frame_time_1, frame_id_2, frame_time_2, maps)
sta = STA(queue_3, queue_4, queue_9)

visualizer = Visualizer([queue_5, queue_6], [queue_7, queue_8], queue_9)

# offline
cap_1 = Camera(1)
cap_2 = Camera(2)

sct_1 = LoadSCTResult(1)
sct_2 = LoadSCTResult(2)
sta = LoadSTAResult(1, 2)

visualizer = Visualizer()
"""

  




if __name__ == '__main__':


    VIDEO_SET = {
        '2d_v1': {'cam_id1': 21, 'cam_id2': 27, 'detection_mode': 'box', 'representation_mode': 'foot', 'range_': range(0, 16)},
        '2d_v2': {'cam_id1': 21, 'cam_id2': 27, 'detection_mode': 'box', 'representation_mode': 'foot', 'range_': range(19, 25)},
        '2d_v3': {'cam_id1': 121, 'cam_id2': 127, 'detection_mode': 'box', 'representation_mode': 'bottom', 'range_': range(6, 7)},
        #'2d_v4': {'cam_id1': 41, 'cam_id2': 42, 'detection_mode': 'pose', 'representation_mode': 'bottom', 'range_': range(1, 13)},
        '2d_v4': {'cam_id1': 42, 'cam_id2': 43, 'detection_mode': 'pose', 'representation_mode': 'bottom', 'range_': range(1, 13)},
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
    FILTER_CHOICE = [None, 'IQR', 'GMM']
    WINDOW_CHOICE = [(1, 0), (11, 5)]

    video_set = '2d_v4'
    tracker_name = 'YOLOv8l_pretrained-640-ByteTrack'
    filter_type = None  # None, 'GMM', 'IQR'
    max_filter_iters = 2
    IQR_lower = 30
    IQR_upper = 70
    window_size = 1
    window_boundary = 0
    gttracker_filter = None #IQRFilter(25, 75).run

    
    video_set_dir = VIDEO_SET[video_set]['video_set_dir']
    cam1_id = VIDEO_SET[video_set]['cam_id1']
    cam2_id = VIDEO_SET[video_set]['cam_id2']
    representation_mode = VIDEO_SET[video_set]['representation_mode']
    range_ = VIDEO_SET[video_set]['range_']

    # if filter_type == 'GMM':
    #     filter = GMMFilter(n_components=2, std_coef=3).run
    #     # gttracker_filter = GMMFilter(n_components=1, std_coef=3).run
    # elif filter_type == 'IQR':
    #     filter = IQRFilter(IQR_lower, IQR_upper).run
    #     # gttracker_filter = IQRFilter(25, 75).run
    # else:
    #     filter = None
    #     # gttracker_filter = None

    for video_id in tqdm(range_):

        ################# retrieve paths #############
        vid1_path = list((Path(video_set_dir) / 'videos').glob(f"{cam1_id}_{('00000' + str(video_id))[-5:]}_*_*.avi"))
        assert len(vid1_path) == 1
        vid1_path = str(vid1_path[0])
        vid1_name = os.path.split(vid1_path)[0]
        vid1_basename = os.path.splitext(vid1_name)[0]
        vid2_path = list((Path(video_set_dir) / 'videos').glob(f"{cam2_id}_{('00000' + str(video_id))[-5:]}_*_*.avi"))
        assert len(vid2_path) == 1
        vid2_path = str(vid2_path[0])
        vid2_name = os.path.split(vid2_path)[0]
        vid2_basename = os.path.splitext(vid2_name)[0]
        
        gt_txt1_path = str(Path(video_set_dir) / 'gt' / (vid1_basename + '.txt'))
        gt_txt2_path = str(Path(video_set_dir) / 'gt' / (vid2_basename + '.txt'))
        tracker_txt1_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid1_basename + '.txt'))
        tracker_txt2_path = str(Path(video_set_dir) / tracker_name / 'sct' / (vid2_basename + '.txt'))

        meta1_path = str(Path(video_set_dir) / 'meta' / (vid1_basename + '.yaml'))
        meta2_path = str(Path(video_set_dir) / 'meta' / (vid2_basename + '.yaml'))
        ###############################################

        ################# load files ##################
        logging.debug(f'Opening video at {vid1_path}')
        cap1 = cv2.VideoCapture(vid1_path)
        logging.debug(f'Opening video at {vid2_path}')
        cap2 = cv2.VideoCapture(vid2_path)


        ###############################################
