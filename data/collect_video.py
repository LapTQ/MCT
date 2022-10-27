import cv2
from datetime import datetime
from tqdm import tqdm
import os
from pathlib import Path
from multiprocessing import Pool
from threading import Thread
import queue
import time

CAMERAS = [
    'rtsp://admin:123456a@@192.168.3.63/cam1',
    'rtsp://admin:123456a@@192.168.3.64/cam2',
]
DURATIONS = [
    1,
    1,
]  # minutes

OUT_DIR = Path(__file__).parent / 'recordings'
DATETIME_FORMAT = '%Y-%m-%d_%H-%M-%S-%f'
VID_EXT = '.avi'

os.makedirs(OUT_DIR, exist_ok=True)


def record(cam_address, duration):

    cam_id = os.path.split(cam_address)[-1]
    cap = cv2.VideoCapture(cam_address)

    if not cap.isOpened():
        print('Cannot open RTSP stream from', cam_address)
        exit(-1)

    exists = list(Path(OUT_DIR).glob(cam_id + '*.avi'))
    vid_id = 0 if len(exists) == 0 else int(sorted(exists)[-1].stem.split('_')[1]) + 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    n_frames = int(duration * 60 * fps)

    q = queue.Queue()

    now = datetime.now().strftime(DATETIME_FORMAT)
    name = cam_id + '_' + ('00000' + str(vid_id))[-5:] + '_' + now + VID_EXT

    out_video_path = os.path.join(OUT_DIR, name)
    video_writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),    # critical, much smaller than MJPG
        fps,
        (W, H)
    )

    def _read():
        pbar = tqdm(range(n_frames), ascii=True, desc=name)
        n_errors = 0
        for _ in pbar:
            success, frame = cap.read()

            if not success:
                n_errors += 1
            else:
                q.put(frame)
                time.sleep(0.08)    # critical @@
            pbar.set_postfix(n_errors=n_errors, queue_size=q.qsize())

    def _write():
        n_written = 0
        while True:
            if not q.empty():
                frame = q.get()
                video_writer.write(frame)
                n_written += 1
            if n_written == n_frames:
                break

    p1 = Thread(target=_read)
    p2 = Thread(target=_write)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    cap.release()
    video_writer.release()
    print(f'[INFO] Stream from {cam_address} is saved in {out_video_path}')




if __name__ == '__main__':

    start_time = time.time()
    pool = Pool(len(CAMERAS))
    pool.starmap(record, zip(CAMERAS, DURATIONS))
    print('[INFO] Running time:', time.time() - start_time)