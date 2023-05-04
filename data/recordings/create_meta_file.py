import os
import cv2
from pathlib import Path


HERE = Path(__file__).parent.resolve()
DIR = '2d_v3'

def run(vid_path):
    vid_path = str(vid_path)
    parent, vid_name = os.path.split(vid_path)
    vid_basename, _ = os.path.splitext(vid_name)
    cap = cv2.VideoCapture(vid_path)
    info = {
        'name': vid_name,
        'cam_id': vid_basename.split('_')[0],
        'video_id': int(vid_basename.split('_')[1]),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'start_time': '_'.join(vid_basename.split('_')[2:]),
        'start_frame_id': 1,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    out_dir = os.path.join(parent, '../meta')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, vid_basename + '.yaml'), 'w') as f:
        f.write('\n'.join([f'{k}: {v}' for k, v in info.items()]))



if __name__ == '__main__':
    vid_paths = list((HERE / DIR / 'videos').glob('*.avi'))
    for vid_path in vid_paths:
        run(vid_path)