from pathlib import Path
import os
import math

HERE = Path(__file__).parent

FILE_PATH = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack/pred/18_eval.txt'
VID = [1, 2, 3, 4, 5, 6, 9, 10, 7, 8, 11, 12]


TP = [0, 0, 0]  # easy, medium, hard
FP = [0, 0, 0]
FN = [0, 0, 0]


with open(FILE_PATH, 'r') as f:
    ct = f.read().strip()

ct = [l.strip() for l in ct.split('\n')]

for l in ct:
    if l.startswith('======'):
        video_id = int(l.split(',')[2].strip().split()[-1])
        idx = VID.index(video_id) // 4
    elif l.startswith('TP:'):
        TP[idx] += int(l.split()[-1])
    elif l.startswith('FP:'):
        FP[idx] += int(l.split()[-1])
    elif l.startswith('FN:'):
        FN[idx] += int(l.split()[-1])

        if video_id == 12:
            break
    
for i, name in enumerate(['EASY', 'MEDI', 'HARD']):
    pre = TP[i] / (TP[i] + FP[i])
    rec = TP[i] / (TP[i] + FN[i])
    f1 = 2 * pre * rec / (pre + rec)
    print(f'{name}: {round(f1, 3)} ({TP[i]} - {FP[i]} - {FN[i]})')

sTP = sum(TP)
sFP = sum(FP)
sFN = sum(FN)
spre = sTP / (sTP + sFP)
srec = sTP / (sTP + sFN)
sf1 = 2 * spre * srec / (spre + srec)
print(f'ALLL: {round(sf1, 3)} ({sTP} - {sFP} - {sFN})')



