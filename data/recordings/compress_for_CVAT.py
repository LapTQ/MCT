import os
from pathlib import Path

SRC_DIR = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv8l_pretrained-640-ByteTrack/sct'
DST_DIR = '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv8l_pretrained-640-ByteTrack/sct'
EXTENSION = '.txt'

filenames = [name for name in os.listdir(SRC_DIR) if name.endswith(EXTENSION)]

temp_dir = os.path.join(DST_DIR, 'gt')
assert not os.path.isdir(temp_dir), f'Directory {temp_dir} already exists'
os.mkdir(temp_dir)

for filename in filenames:
    basename = os.path.splitext(filename)[0]
    filepath = os.path.join(SRC_DIR, filename)

    print(f'[INFO] Reading source file at {filepath}')
    with open(filepath, 'r') as f:
        ct = f.read()
        ct = ct.replace(' ', ',').replace('-1,-1,-1,0', '1,1,1.0')

    temp_dst_path = os.path.join(DST_DIR, 'gt', 'gt.txt')
    print(f'[INFO] Writing temporary file at {temp_dst_path}')
    with open(temp_dst_path, 'w') as f:
        f.write(ct)

    dst_path = os.path.join(DST_DIR, basename + '.zip')
    if os.path.isfile(dst_path):
        print(f"[INFO] Removing previous {dst_path}")
        os.system(f"rm {dst_path}")
    print(f'[INFO] Archiving {dst_path}')
    os.system(f"cd {DST_DIR} && zip -r {basename + '.zip'} {'gt'}")

if os.path.isdir(temp_dir):
    print(f"[INFO] Removing temporary directory {temp_dir}")
    os.system(f"rm -r {temp_dir}")

