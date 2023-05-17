import os
from pathlib import Path

DIR = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/gt_splited'
EXTENSION = '.zip'

filenames = [name for name in os.listdir(DIR) if name.endswith(EXTENSION)]

for filename in filenames:
    path = os.path.join(DIR, filename)
    basename = os.path.splitext(filename)[0]

    print(f'[INFO]\t Archiving {DIR}/{filename} to directory {DIR}')
    os.system(f'unzip {DIR}/{filename} -d {DIR}' )
    print(f"[INFO]\t Moving {os.path.join(DIR, 'gt', 'gt.txt')} to {os.path.join(DIR, basename + '.txt')}")
    os.system('mv ' + os.path.join(DIR, 'gt', 'gt.txt') + ' ' + os.path.join(DIR, basename + '.txt'))
    print(f"[INFO]\t Removing {os.path.join(DIR, 'gt')}")
    os.system('rm -r ' + os.path.join(DIR, 'gt'))

