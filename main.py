import argparse
import os
import cv2
import time

from detection.detector import YOLOv5
from tracking.tracker import SORT
from visualization.visualizer import plot_box

def parse_opt():
    ap = argparse.ArgumentParser()

    ap.add_argument('--input', type=str, default='0')
    ap.add_argument('--output', type=str, default='output')
    ap.add_argument('--display', action='store_true')

    opt = ap.parse_args()

    return opt


def main(opt):

    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    if not os.path.exists(opt.input):
        print('[INFO] Video %s not exists' % opt.input)
        return

    # TODO loader module
    video_loader = cv2.VideoCapture(opt.input)
    FPS = video_loader.get(cv2.CAP_PROP_FPS)
    filename = os.path.basename(opt.input)

    if opt.display:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

    t0 = time.time()

    # TODO tham so
    detector = YOLOv5()
    tracker = SORT()

    print('[TIME] Loading models:', time.time() - t0)

    frame_count = 0

    while True:
        ret, frame = video_loader.read()

        if not ret or frame is None or cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
            break

        frame_count += 1

        t0 = time.time()

        dets = detector.predict(frame, BGR=True)    # [[x1, y1, x2, y2, conf], ...]

        print('[INFO] Detect %d people' % dets.shape[0])
        print('[TIME] Detection:', time.time() - t0)

        t0 = time.time()
        # TODO ret = [[frame, id, x1, y1, w, h], ...]
        # TODO add frame num
        ret = tracker.update(dets)  # [id, x1, y1, x2, y2]

        print('[TIME] Tracking:', time.time() - t0)

        t0 = time.time()
        # TODO visualizer
        if opt.display:

            show_img = plot_box(frame, ret)    # dets[:, :4]
            cv2.imshow(filename, show_img)

        print('[TIME] Visualization:', time.time() - t0)

    video_loader.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)

