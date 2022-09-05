import os
import argparse
import cv2
import time

import numpy as np

from mct.sct import SCT
from mct.utils.vis_utils import plot_box


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--hardware', type=str, required=True)
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--output', type=str, default='output')
    ap.add_argument('--display', action='store_true')

    opt = ap.parse_args()

    return opt


def main(opt):

    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    if opt.input == '0':
        opt.input = 0
    elif not os.path.exists(opt.input):
        print('[INFO] Video %s not exists' % opt.input)
        return

    # TODO different options here: if opt.hardware == 'weak':
    sct = SCT()
    detector = sct.create_detector()
    tracker = sct.create_tracker()

    # TODO video loader:
    video_loader = cv2.VideoCapture(opt.input)
    FPS = video_loader.get(cv2.CAP_PROP_FPS)
    filename = os.path.basename(str(opt.input))

    if opt.display:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

    t0 = time.time()

    print('[TIME] Loading models:', time.time() - t0)

    frame_count = 0

    while True:
        ret, frame = video_loader.read()

        if not ret or frame is None or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

        frame_count += 1

        t0 = time.time()

        dets = detector.predict(frame, BGR=True)    # [[x1, y1, x2, y2, conf], ...]

        print('[INFO] Detect %d people' % dets.shape[0])
        print('[TIME] Detection:', time.time() - t0)

        t0 = time.time()
        # TODO ret = [[frame, id, x1, y1, w, h], ...]
        # TODO add frame num to dets
        # TODO refactor: adapter
        # TODO tại sao trong code của tracker không dùng tới conf của dets? kiểm tra lại thông số của kalman xem có liên quan không
        ret = tracker.update(dets)  # [id, x1, y1, x2, y2, conf]

        print('[TIME] Tracking:', time.time() - t0)

        t0 = time.time()
        # TODO visualizer
        if opt.display:
            info = np.concatenate([np.array([[frame_count]] * len(ret)), ret], axis=1)
            show_img = plot_box(frame, info)
            cv2.imshow(filename, show_img)

        print('[TIME] Visualization:', time.time() - t0)

    video_loader.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    opt = parse_opt()
    main(opt)