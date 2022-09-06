import os
import argparse
import cv2
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import pymongo
from pymongo import MongoClient

import sys
sys.path.append(sys.path[0] + '/..')

from mct.sct import SCT
from mct.utils.vis_utils import plot_box


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    # ap.add_argument('--hardware', type=str, required=True)
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--output', type=str, default='../output/')
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--save_db', action='store_true')
    ap.add_argument('--port', type=int)
    ap.add_argument('--save_txt', action='store_true')

    opt = ap.parse_args()

    return opt


def main(opt):

    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)
    opt.output = Path(opt.output)

    if opt.input == '0':
        opt.input = 0
    elif not os.path.exists(opt.input):
        print('[INFO] Video %s not exists' % opt.input)
        return

    t0 = time.time()

    # TODO different options here: if opt.hardware == 'weak':
    sct = SCT()
    detector = sct.create_detector()
    tracker = sct.create_tracker()

    # TODO video loader:
    video_loader = cv2.VideoCapture(opt.input)
    FPS = video_loader.get(cv2.CAP_PROP_FPS)
    filename = os.path.basename(str(opt.input))
    name_root, _ = os.path.splitext(filename)
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    if opt.save_db:
        print("[INFO] Connecting to database at mongodb://localhost:27017")
        mongo_client = MongoClient('localhost', opt.port)
        sct_db = mongo_client['sct_db']
        sct_collections = sct_db[now + '_' + name_root]

    if opt.display:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

    if opt.save_txt:
        txt_buffer = []
        out_txt = open(opt.output/(now + '_' + name_root + '.txt'), 'w')

    print('[TIME] Loading models:', time.time() - t0)

    frame_count = 0
    while True:
        ret, frame = video_loader.read()

        if not ret or frame is None or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

        frame_count += 1
        print('[INFO] Frame:', frame_count)

        t0 = time.time()

        dets = detector.predict(frame, BGR=True)    # [[x1, y1, x2, y2, conf], ...]

        print('[INFO] Detect %d people' % dets.shape[0])
        print('[TIME] Detection:', time.time() - t0)

        t0 = time.time()
        # TODO ret = [[frame, id, x1, y1, w, h], ...]
        # TODO add frame num to dets?
        # TODO refactor: adapter
        # TODO tại sao trong code của kalman không dùng tới conf của dets? kiểm tra lại thông số của kalman xem có liên quan không
        ret = tracker.update(dets)  # [id, x1, y1, x2, y2, conf]

        print('[TIME] Tracking:', time.time() - t0)

        if opt.save_db:
            t0 = time.time()
            for obj in ret:
                sct_collections.update_one(
                    {'trackid': np.int32(obj[0])},
                    {'$push': {'detections': {'frameid': frame_count,
                                              'box': obj[1:5].tolist(), # xyxy
                                              'score': obj[5]
                                              }
                               }
                     },
                    upsert=True
                )
            print('[TIME] Save to database:', time.time() - t0)

        if opt.save_txt:
            t0 = time.time()
            for obj in ret:
                # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
                txt_buffer.append(
                    f'{frame_count}, {np.int32(obj[0])}, {obj[1]:.2f}, {obj[2]:.2f}, {(obj[3] - obj[1]):.2f}, {(obj[4] - obj[2]):.2f}, {obj[5]:.6f}, -1, -1, -1')
            print('[TIME] Save to .txt:', time.time() - t0)

        if opt.display:
            t0 = time.time()
            # TODO visualizer
            info = np.concatenate([np.array([[frame_count]] * len(ret)), ret], axis=1)
            show_img = plot_box(frame, info)
            cv2.imshow(filename, show_img)
            print('[TIME] Visualization:', time.time() - t0)

    video_loader.release()
    cv2.destroyAllWindows()

    if opt.save_txt:
        print('\n'.join(txt_buffer), file=out_txt)
        print('[INFO] Result saved in', opt.output/(now + '_' + name_root + '.txt'))
        out_txt.close()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)