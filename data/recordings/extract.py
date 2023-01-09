import os
from pathlib import Path

HERE = str(Path(__file__).parent / '2d_v2/gt')

files = [str(path) for path in Path(HERE).glob('*000*.zip')]

for path in files:
    filename = os.path.split(path)[1]
    basename = os.path.splitext(filename)[0]

    os.system(f'unzip {HERE}/{filename} -d {HERE}' )
    os.system('mv ' + os.path.join(HERE, 'gt', 'gt.txt') + ' ' + os.path.join(HERE, basename + '.txt'))
    os.system('rm -r ' + os.path.join(HERE, 'gt'))

