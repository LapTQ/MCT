import os
from pathlib import Path


HERE = Path(__file__).parent

DIR = str(HERE / '../recordings/PETS09/videos')


def run(vid_path):

    params = {
        'name': 'yolox-s', # don't care
        'ckpt': HERE / '../../mct/weights/yolov7-w6-pose.pt',
        'path': vid_path,
        'tsize': 640,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'tracking_method': 'bytetrack',
        'reid-weights': HERE / 'osnet_x1_0.pt',
        'device': 'cuda', # 'cuda' or 'cpu'
        'save-txt': '',
        'save-vid': ''
    }

    command = f"python3 {HERE}/yolov7pose_track.py video {' '.join([f'--{k} {v}' for k, v in params.items()])}"
    print('\n\n[INFO]\t Running command:', command)
    os.system(command)


def main():
    for vid_name in sorted(os.listdir(DIR)):
        vid_path = os.path.join(DIR, vid_name)
        run(vid_path)
        #break


if __name__ == '__main__':
    main()