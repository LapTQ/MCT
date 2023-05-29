import os
from pathlib import Path
import numpy as np

GT_DIR = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/gt_splited'
TRACKER_DIR = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct'

def run(filename):

    print('[INFO] filename =', filename)

    gt_path = os.path.join(GT_DIR, filename)
    tracker_path = os.path.join(TRACKER_DIR, filename)

    with open(gt_path, 'r') as f:
        gt = f.read().strip().split('\n')
        gt = [l.split(',') for l in gt]
        gt = [[eval(i) for i in l] for l in gt]
        gt = np.array(gt)
    with open(tracker_path, 'r') as f:
        tracker = f.read().strip().split('\n')
        tracker = [l.split() for l in tracker]
        tracker = [[eval(i) for i in l] for l in tracker]
        types = [type(i) for i in tracker[0]]
        tracker = np.array(tracker)
    
    # assert len(gt) == len(tracker), f'gt has {len(gt)} rows, tracker has {len(tracker)} rows'
    
    gt_max_fid = np.max(np.int32(gt[:, 0]))
    tracker_max_fid = np.max(np.int32(tracker[:, 0]))
    assert gt_max_fid == tracker_max_fid, f'gt has {gt_max_fid} frames, tracker has {tracker_max_fid} frames'

    ret = []

    for fid in np.unique(gt[:, 0]):
        assert fid in np.int32(tracker[:, 0]).tolist()
        gt_dets = gt[gt[:, 0] == fid]
        tracker_dets = tracker[tracker[:, 0] == fid]
        assert len(gt_dets) == len(tracker_dets), f'at frame {fid}, gt has {len(gt_dets)} rows, tracker has {len(tracker_dets)} rows'
        
        gt_dets = gt_dets[np.argsort(gt_dets[:, 2])]
        tracker_dets = tracker_dets[np.argsort(tracker_dets[:, 2])]
        tracker_dets[:, 1] = gt_dets[:, 1]
        for d in tracker_dets:
            ret.append([types[i](d[i]) for i in range(len(types))])
    
    ret = '\n'.join([' '.join([str(i) for i in l]) for l in ret])
    
    with open(tracker_path, 'w') as f:
        f.write(ret)



if __name__ == '__main__':
    for filename in sorted(os.listdir(GT_DIR)):
        run(filename)