from pathlib import Path

import numpy as np
import cv2

import sys
HERE = Path(__file__).parent

sys.path.append(str(HERE.parent))

from mct.utils.draw import visualize_multi_camera
from mct.utils.general import create_global_id_mapper


def run(
    video_set,
    detection_mode,
    config_pred_option,
    cam_id1,
    cam_id2,
    cam_id3,
    video_id,
    tracker,
):  
    
    print('Processing video', video_id)

    ROOT_DIR = HERE / 'recordings' / video_set
    VID_DIR = HERE / 'recordings' / video_set / 'videos'
    TRACKER_DIR = HERE / 'recordings' / video_set / tracker / 'sct'
    GT_DIR = HERE / 'recordings' / video_set / f'gt{"_pose" if detection_mode == "pose" else ""}'

    vid_w_prefix = ('00000' + str(video_id))[-5:]
    vid_path1 = str(list(VID_DIR.glob(f'{cam_id1}_{vid_w_prefix}_*.avi'))[0])
    txt_path1 = list(TRACKER_DIR.glob(f'{cam_id1}_{vid_w_prefix}_*.txt'))[0]
    vid_path2 = str(list(VID_DIR.glob(f'{cam_id2}_{vid_w_prefix}_*.avi'))[0])
    txt_path2 = list(TRACKER_DIR.glob(f'{cam_id2}_{vid_w_prefix}_*.txt'))[0]
    vid_path3 = str(list(VID_DIR.glob(f'{cam_id3}_{vid_w_prefix}_*.avi'))[0])
    txt_path3 = list(TRACKER_DIR.glob(f'{cam_id3}_{vid_w_prefix}_*.txt'))[0]

    out_video_path = str(ROOT_DIR / tracker / f'pred/{config_pred_option}_val' / f'frame2track_{video_id}.avi')
    print(out_video_path)
    
    match12 = np.loadtxt(ROOT_DIR / tracker / 'pred' / f'{config_pred_option}_frame2track_{cam_id1}_{cam_id2}.txt', delimiter=',', dtype=int)
    match12 = match12[match12[:, 1] == video_id][:, [2, 5]]
    match23 = np.loadtxt(ROOT_DIR / tracker / 'pred' / f'{config_pred_option}_frame2track_{cam_id2}_{cam_id3}.txt', delimiter=',', dtype=int)
    match23 = match23[match23[:, 1] == video_id][:, [2, 5]]
    
    cap1 = cv2.VideoCapture(vid_path1)
    cap2 = cv2.VideoCapture(vid_path2)
    cap3 = cv2.VideoCapture(vid_path3)
    try:
        dets1 = np.loadtxt(txt_path1, delimiter=',')
    except:
        dets1 = np.loadtxt(txt_path1)
    try:
        dets2 = np.loadtxt(txt_path2, delimiter=',')
    except:
        dets2 = np.loadtxt(txt_path2)
    try:
        dets3 = np.loadtxt(txt_path3, delimiter=',')
    except:
        dets3 = np.loadtxt(txt_path3)

    global_ids_mapper = create_global_id_mapper([dets1, dets2, dets3], [match12, match23])

    visualize_multi_camera(
        caps=[cap1, cap2, cap3],
        scts=[dets1, dets2, dets3],
        grid_size=(2, 2),
        global_ids_mapper=global_ids_mapper,
        resize=(960, 480),
        out_video_path=out_video_path,
        show=False,
    )


if __name__ == '__main__':

    video_set = '2d_v4'
    detection_mode = 'pose' # 'box' or 'pose'
    config_pred_option = 18
    cam_id1 = 41
    cam_id2 = 42
    cam_id3 = 43
    tracker = f'YOLOv7{detection_mode}_pretrained-640-ByteTrack-IDfixed'

    for video_id in range(9, 10):
        run(
            video_set=video_set,
            detection_mode=detection_mode,
            config_pred_option=config_pred_option,  
            cam_id1=cam_id1,
            cam_id2=cam_id2,
            cam_id3=cam_id3,
            video_id=video_id,
            tracker=tracker,
        )

    
    
    