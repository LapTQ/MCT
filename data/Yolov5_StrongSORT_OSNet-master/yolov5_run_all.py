from  multiprocessing import Pool
from pathlib import Path
import os

HERE = Path(__file__).parent

DIR = str(HERE / '../recordings/2d_v1/videos')  # video set

def run(vid_path):
    params = {
        'source': vid_path,
        'yolo-weights': str(HERE / '../../mct/weights/yolov5l.pt'),
        'img': 640,
        'tracking-method': 'strongsort',
        'reid-weights': str(HERE / 'osnet_x1_0.pt'),
        'classes': 0,
        'device': "''", # 'cpu' or "''"
        'save-txt': '',
        'save-vid': ''
    }

    command = f"python3 {HERE}/yolov5_track.py {' '.join([f'--{k} {v}' for k, v in params.items()])}"
    print('\n\n[INFO]\t Running command:', command)
    os.system(command)


def main():
    for vid_name in os.listdir(DIR):
        vid_path = os.path.join(DIR, vid_name)
        run(vid_path)


if __name__ == '__main__':
    main()



