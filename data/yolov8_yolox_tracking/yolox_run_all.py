import os
from pathlib import Path


HERE = Path(__file__).parent


def run(vid_path):

    params = {
        'name': 'yolox-l',
        'ckpt': HERE / '../../mct/weights/yolox_l.pth',
        'path': vid_path,
        'tsize': 640,
        'conf': 0.5,
        'tracking_method': 'strongsort',
        'reid-weights': HERE / 'osnet_x1_0.pt',
        'classes': 0,
        'save-txt': '',
        'save-vid': ''
    }

    command = f"python3 {HERE}/yolox_track.py video {' '.join([f'--{k} {v}' for k, v in params.items()])}"
    print('\n\n[INFO]\t Running command:', command)
    os.system(command)


def main():
    DIR = str(HERE / '../recordings/2d_v2/videos')
    for vid_name in sorted(os.listdir(DIR)):
        vid_path = os.path.join(DIR, vid_name)
        run(vid_path)
        # break


if __name__ == '__main__':
    main()