from pathlib import Path
import os

HERE = Path(__file__).parent

DIRS = ['2d_v2']
COUNTS = [6]
TRACKER_PREFIX = 'YOLOv8l_pretrained-640-StrongSORT'
FILENAME = 'log_error_analysis_pred_mct_trackertracker_correspondences_v2_noFilter_windowsize11_windowboundary5.txt'

tracker_names = [name for name in os.listdir(str(HERE / 'recordings' / DIRS[0])) if name.startswith(TRACKER_PREFIX)]

c_Pre = 0
c_Rec = 0
c_F1 = 0
for tracker_name in tracker_names:
    Pre = 0
    Rec = 0
    F1 = 0
    for dir in DIRS:
        with open(HERE / 'recordings' / dir / tracker_name / FILENAME, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                if line.startswith('Pre'):
                    line = line.split()
                    Pre += float(line[1])
                elif line.startswith('Rec'):
                    line = line.split()
                    Rec += float(line[1])
                elif line.startswith('F1'):
                    line = line.split()
                    F1 += float(line[1])

    Pre /= sum(COUNTS)
    Rec /= sum(COUNTS)
    F1 /= sum(COUNTS)

    print(f'{tracker_name}:\tP = {Pre:.3f}\tR = {Rec:.3f}\tF1 = {F1:.3f}')

    c_Pre += Pre
    c_Rec += Rec
    c_F1 += F1

c_Pre /= len(tracker_names)
c_Rec /= len(tracker_names)
c_F1 /= len(tracker_names)
print(f'Combined:\tP = {c_Pre:.3f}\tR = {c_Rec:.3f}\tF1 = {c_F1:.3f}')
