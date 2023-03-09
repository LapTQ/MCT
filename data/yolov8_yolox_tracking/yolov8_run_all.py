from pathlib import Path
import os

HERE = Path(__file__).parent

def run(vid_path):
    params = {
        'source': vid_path,
        'yolo-weights': str(HERE / '../../mct/weights/yolov8l.pt'),
        'img': 1280,
        'tracking-method': 'strongsort',
        'reid-weights': str(HERE / 'osnet_x1_0.pt'),
        'classes': 0,
        'save-txt': '',
        'save-vid': ''
    }

    command = f"python3 {HERE}/yolov8_track.py {' '.join([f'--{k} {v}' for k, v in params.items()])}"
    print('\n\n[INFO]\t Running command:', command)
    os.system(command)


def main():
    DIR = str(HERE / '../recordings/2d_v3/videos')
    for vid_name in os.listdir(DIR):
        vid_path = os.path.join(DIR, vid_name)
        run(vid_path)


if __name__ == '__main__':
    main()