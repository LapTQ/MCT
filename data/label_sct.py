import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import cv2

import sys
sys.path.append(sys.path[0] + '/..')

from mct.utils.draw import plot_box


HERE = Path(__file__).parent


def visualize_multi_camera(
        caps, 
        grid_size,
        scts=None,
        global_ids_mapper=None,
        out_video_path=None,
        resize=None,
        show=False,
    ):

    if global_ids_mapper is None:
        global_ids_mapper = [None] * len(caps)

    if scts is None:
        scts = [None] * len(caps)

    writer_created = False
    H = grid_size[0] * int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = grid_size[1] * int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    n_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    for fid in tqdm(range(1, n_frames + 1)):

        grid = np.zeros((H, W, 3), dtype=np.uint8)

        frames = []
        for cap, sct, gID in zip(caps, scts, global_ids_mapper):
            _, img = cap.read()

            if sct is not None:
                dets = sct[sct[:, 0] == fid]

                if gID is not None:
                    for i, d in enumerate(dets):
                        dets[i, 1] = gID[int(dets[i, 1])]
                
                img = plot_box(img, dets, thickness=8)

            frames.append(img)
        
        for i, frame in enumerate(frames):
            row = i // grid_size[1]
            col = i % grid_size[1]
            grid[row * frame.shape[0]:(row + 1) * frame.shape[0], col * frame.shape[1]:(col + 1) * frame.shape[1]] = frame

        if out_video_path is not None:
            if not writer_created:
                writer = cv2.VideoWriter(
                    out_video_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    caps[0].get(cv2.CAP_PROP_FPS),
                    (W, H) if resize is None else resize
                )
                writer_created = True
            
            if resize is not None:
                grid = cv2.resize(grid, resize)
            
            writer.write(grid)
        
        if show:
            cv2.namedWindow('grid', cv2.WINDOW_NORMAL)
            cv2.imshow('grid', grid)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('e'):
                exit(0)
            elif key == ord(' '):
                cv2.waitKey(0)
    
    if out_video_path is not None:
        writer.release()
    cv2.destroyAllWindows()


def create_global_id_mapper(scts, matches):
    """assuming [[[c1, c2], ...], [[c2, c3], ...], [[c3, c4], ...], ...]"""

    global_ids_mapper = [{} for _ in range(len(matches) + 1)]
    global_id_count = 0
    
    table = np.full((sum([len(m) for m in matches]), len(matches) + 1), -1, dtype=int)
    i = 0
    for j, m in enumerate(matches):
        for id1, id2 in m:
            table[i, j] = id1
            table[i, j + 1] = id2
            i += 1
    
    for i_out in range(table.shape[0]):
        rows = []
        k = 0
        if table[i_out, 0] != -1:
            rows.append((i_out, table[i_out].copy()))
            table[i_out] = -1
        while k < len(rows):
            i, row = rows[k]
            for j, id in enumerate(row):
                if id != -1:
                    for new_i in np.where(table[:, j] == id)[0]:
                        if i != new_i:
                            rows.append((new_i, table[new_i].copy()))
                        table[new_i] = -1
            k += 1

        if len(rows) > 0:
            global_id_count += 1
            for row in rows:
                for j, id in enumerate(row[1]):
                    if id != -1:
                        global_ids_mapper[j][id] = global_id_count

    for i, sct in enumerate(scts):
        for id in np.unique(sct[:, 1]).astype('int32'):
            if id not in global_ids_mapper[i]:
                global_id_count += 1
                global_ids_mapper[i][id] = global_id_count
    
    print(global_ids_mapper)
    
    return global_ids_mapper


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

    for video_id in [4, 9, 11]:
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

    
    
    