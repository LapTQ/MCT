import os
from pathlib import Path

HERE = str(Path(__file__).parent)

files = [str(path) for path in Path(HERE).glob('*000*.zip')]

for filename in files:
    basename = os.path.splitext(filename)[0]

    os.system('unzip ' + filename + ' -d .')
    os.system('mv ' + os.path.join(HERE, 'gt', 'gt.txt') + ' ' + os.path.join(HERE, basename + '.txt'))
    os.system('rm -r ' + os.path.join(HERE, 'gt'))

