import os
import argparse
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from threading import Thread
from pathlib import Path

import numpy as np
import cv2

import sys
sys.path.append(sys.path[0] + '/..')

from mct.utils.draw import plot_box


HERE = Path(__file__).parent


def visualize_from_txt(vid_path, txt_path, **kwargs):

    out_dir = str(HERE / '../output')
    filename = os.path.split(vid_path)[1]

    cap = cv2.VideoCapture(vid_path)
    with open(txt_path, 'r') as f:
        det_seq = np.array([[eval(e) for e in l.strip().split(',' if ',' in l else None)[:7]] for l in f.readlines()])

    if 'vid_path2' in kwargs:
        cap2 = cv2.VideoCapture(kwargs['vid_path2'])
        with open(kwargs['txt_path2'], 'r') as f:
            det_seq2 = np.array([[eval(e) for e in l.strip().split(',' if ',' in l else None)[:7]] for l in f.readlines()])

    if kwargs.get('save_video', False):
        if 'vid_path2' in kwargs:
            filename2 = os.path.split(kwargs['vid_path2'])[1]
        writer = cv2.VideoWriter(os.path.join(out_dir, 'vis_' + (filename if 'vid_path2' not in kwargs else filename + '_' + filename2)),
                             cv2.VideoWriter_fourcc(*'XVID'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + (0 if 'vid_path2' not in kwargs else cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) ), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if 'vid_path2' in kwargs:
        n_frames = min(n_frames, int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    for frame_count in tqdm(range(n_frames)):
        dets = det_seq[det_seq[:, 0] == frame_count]
        success, frame = cap.read()
        n_uniques_1 = np.unique(det_seq[:, 1])
        if 'vid_path2' not in kwargs:
            vis_img = plot_box(frame, dets)
            cv2.putText(vis_img, str(n_uniques_1), (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=6)
            show_img = vis_img
        else:
            vid_id = int(filename.split('_')[1])

            dets2 = det_seq2[det_seq2[:, 0] == frame_count]
            success2, frame2 = cap2.read()
            n_uniques_2 = np.unique(det_seq2[:, 1])

            if 'correspondence' in kwargs:
                correspondence = kwargs['correspondence'][kwargs['correspondence'][:, 1] == vid_id]
                for id1, id2 in correspondence[:, [2, 5]]:
                    dets[dets[:, 1] == id1, 1] = 100 - id2
                    dets2[dets2[:, 1] == id2, 1] = 100 - id2

            vis_img = plot_box(frame, dets)
            cv2.putText(vis_img, str(n_uniques_1), (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=6)
            vis_img2 = plot_box(frame2, dets2)
            cv2.putText(vis_img2, str(n_uniques_2), (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=6)
            show_img = np.concatenate([vis_img, vis_img2], axis=1)

        if kwargs.get('save_video', False):
            writer.write(show_img)

        if kwargs.get('display', False):
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
            cv2.imshow(filename, show_img)
            key = cv2.waitKey(2)
            if key == 27:
                break
            elif key == ord('e'):
                exit(0)
            elif key == ord(' '):
                cv2.waitKey(0)


    cap.release()
    if kwargs.get('save_video', False):
        writer.release()
    cv2.destroyAllWindows()


def show(vid_path1, vid_path2):
    cap1 = cv2.VideoCapture(vid_path1)
    cap2 = cv2.VideoCapture(vid_path2)

    window_name = os.path.split(vid_path1)[1]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        _, frame1 = cap1.read()
        __, frame2 = cap2.read()
        
        if not (_ and __):
            break
        
        collage = np.concatenate([frame1, frame2], axis=1)

        cv2.imshow(os.path.split(vid_path1)[1], collage)

        key = cv2.waitKey(10)
        if key == 27 or not _:
            break
        elif key == ord('q'):
            exit(0)
        elif key == ord(' '):
            cv2.waitKey(0)

    cv2.destroyAllWindows()





if __name__ == '__main__':



    ROOT_DIR = os.path.join(HERE, 'recordings/2d_v4')
    VID_DIR = os.path.join(HERE, 'recordings/2d_v4/videos')
    TRACKER_DIR = os.path.join(HERE, 'recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct')
    GT_DIR = os.path.join(HERE, 'recordings/2d_v4/gt_pose')

    vid_list1 = sorted([str(path) for path in Path(VID_DIR).glob('42_00011*.avi')]) # ['21_00000_2022-11-03_14-56-57-643967.avi']
    txt_list1 = sorted([str(path) for path in Path(TRACKER_DIR).glob('42_00011*.txt')])
    vid_list2 = sorted([str(path) for path in Path(VID_DIR).glob('43_00011*.avi')])
    txt_list2 = sorted([str(path) for path in Path(TRACKER_DIR).glob('43_00011*.txt')])
    #
    # correspondence = np.loadtxt(f'{ROOT_DIR}/pred_mct_gtgt_correspondences.txt', delimiter=',', dtype=int)   # pred_mct_gtgt_correspondences.txt true_mct_gtgt_correspondences.txt
    correspondence = np.loadtxt(f'{ROOT_DIR}/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/pred/18_frame2track_42_43.txt', delimiter=',', dtype=int)
    #
    for vid_path1, txt_path1, vid_path2, txt_path2 in zip(vid_list1, txt_list1, vid_list2, txt_list2):
        visualize_from_txt(vid_path1, txt_path1, save_video=False, display=True, vid_path2=vid_path2, txt_path2=txt_path2, correspondence=correspondence) # , correspondence=correspondence

    # for vid_path1, txt_path1 in zip(vid_list1, txt_list1):
    #     visualize_from_txt(vid_path1, txt_path1, save_video=False, display=True) # , correspondence=correspondence

    # for vid_id in range(19, 25):
    #     vid_path1 = str(list(Path('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v2/videos').glob(f'21_000{vid_id}*.avi'))[0])
    #     txt_path1 = str(list(Path('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v2/gt').glob(f'21_000{vid_id}*.txt'))[0])
    #     vid_path2 = str(list(Path('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v2/videos').glob(f'27_000{vid_id}*.avi'))[0])
    #     txt_path2 = str(list(Path('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v2/gt').glob(f'27_000{vid_id}*.txt'))[0])
    #     visualize_from_txt(vid_path1, txt_path1, vid_path2=vid_path2,txt_path2=txt_path2)


    # for vid_path1, vid_path2 in zip(vid_list1, vid_list2):
    #     show(vid_path1, vid_path2)
