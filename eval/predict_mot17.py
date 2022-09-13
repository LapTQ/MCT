import os
from pathlib import Path

root = '../data/MOT17/train'

for dir in os.listdir(root):
    input_path = os.path.join(root, dir, 'img1')
    output_path = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/eval/TrackEval/data/trackers/mot_challenge/MOT17-train/SCT/data'
    os.system(f'python3 ../mct/main.py --input {input_path} --output {output_path} --save_txt')
    os.rename(str(list(Path(output_path).glob('*img1.txt'))[0]), os.path.join(output_path, dir + '.txt'))

'''
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL SCT --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1
'''