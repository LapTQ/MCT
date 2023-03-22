from pathlib import Path
import os

HERE = Path(__file__).parent

DIRS = ['2d_v1', '2d_v2', '2d_v3']
TRACKER_PREFIX = 'YOLO'

tracker_names = [name for name in os.listdir(str(HERE / 'recordings' / DIRS[0])) if name.startswith(TRACKER_PREFIX)]

for tracker_name in tracker_names:
    detA = 0
    AssA = 0
    HOTA = 0
    for dir in DIRS:
        with open(HERE / 'recordings' / dir / tracker_name / 'HOTA_eval.txt', 'r') as f:
            while True:
                line = f.readline()
                if line.startswith('COMBINED'):
                    break
            line = line.split()
            HOTA += float(line[1])
            detA += float(line[2])
            AssA += float(line[3])
    detA /= len(DIRS)
    AssA /= len(DIRS)
    HOTA /= len(DIRS)
    print(f'{tracker_name}:\tDetA = {detA:.2f}\tAssA = {AssA:.2f}\tHOTA = {HOTA:.2f}')
