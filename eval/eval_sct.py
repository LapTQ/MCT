import os
from pathlib import Path

DATASET_NAME = '2d_v3'
GT_FOLDER = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/gt'
TRACKER_FOLDER = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/YOLOv8l_pretrained-640-StrongSORT/sct'
TRACKERS_TO_EVAL = 'YOLOv8l_pretrained-640-StrongSORT'


EXTENSION = '.txt'
METRICS = 'HOTA'
USE_PARALLEL = True
NUM_PARALLEL_CORES = 4


HERE = str(Path(__file__).parent)


def prepare_for_eval():

    vid_name_list = [name for name in os.listdir(GT_FOLDER) if name.endswith(EXTENSION)]

    # ====================== prepare ground truth =======================
    gt_competition_dir = os.path.join(HERE, f'TrackEval/data/gt/mot_challenge')
    gt_dataset_dir = os.path.join(gt_competition_dir, f'{DATASET_NAME}-train')
    print('[INFO]\t Creating directory', gt_dataset_dir)
    os.system(f'rm -r {gt_dataset_dir}')
    os.makedirs(gt_dataset_dir)

    for vid_name in vid_name_list:
        vid_dir = os.path.join(gt_dataset_dir, os.path.splitext(vid_name)[0])
        print(f'[INFO]\t Creating directory {vid_dir}/gt')
        os.makedirs(f'{vid_dir}/gt')
        print(f'[INFO]\t Copying from {GT_FOLDER}/{vid_name} to {vid_dir}/gt/gt.txt')
        os.system(f'cp {GT_FOLDER}/{vid_name} {vid_dir}/gt/gt.txt')
        print(f'[INFO]\t Creating {vid_dir}/seqinfo.ini')
        with open(f'{vid_dir}/seqinfo.ini', 'w') as f:
            msg = f"""[Sequence]
name={os.path.splitext(vid_name)[0]}
imDir=img1
frameRate=30
seqLength=1000
imWidth=1920
imHeight=1080
imExt=.jpg"""
            f.write(msg)

    for split in ['train', 'test', 'all']:
        with open(f'{gt_competition_dir}/seqmaps/{DATASET_NAME}-{split}.txt', 'w') as f:
            msg = '\n'.join(['name'] + [os.path.splitext(name)[0] for name in vid_name_list])
            f.write(msg)

    # ================== prepare tracker =======================
    tracker_competition_dir = os.path.join(HERE, f'TrackEval/data/trackers/mot_challenge')
    tracker_dataset_dir = os.path.join(tracker_competition_dir, f'{DATASET_NAME}-train/{TRACKERS_TO_EVAL}')
    print(f'[INFO]\t Creating directory {tracker_dataset_dir}/data')
    os.system(f'rm -r {tracker_dataset_dir}')
    os.makedirs(f'{tracker_dataset_dir}/data')

    for vid_name in vid_name_list:
        print(f'[INFO]\t Copying from {TRACKER_FOLDER}/{vid_name} to {tracker_dataset_dir}/data/{vid_name}')
        os.system(f'cp {TRACKER_FOLDER}/{vid_name} {tracker_dataset_dir}/data/{vid_name}')

    params = {
        'GT_FOLDER': gt_competition_dir,
        'TRACKERS_FOLDER': tracker_competition_dir,
        'BENCHMARK': DATASET_NAME,
        'SPLIT_TO_EVAL': 'train',
        'TRACKERS_TO_EVAL': TRACKERS_TO_EVAL
    }

    return params


def eval(params):
    command = f"python3 {HERE}/TrackEval/scripts/run_mot_challenge.py {' '.join([f'--{k} {v}' for k, v in params.items()])} --METRICS {METRICS} --USE_PARALLEL {USE_PARALLEL} --NUM_PARALLEL_CORES {NUM_PARALLEL_CORES}"
    print('[INFO]\t Running command $', command)
    os.system(command)


if __name__ == '__main__':
    params = prepare_for_eval()
    eval(params)
