import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2

import sys
sys.path.append(sys.path[0] + '/..')

from mct.sct import SCT
from mct.utils.vis_utils import plot_box
from mct.utils.db_utils import DBDirector, PymongoBuilder, MongoEngineBuilder
from mct.utils.vid_utils import LoaderDirector, VideoLoaderBuilder, ImageFolderLoaderBuilder


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    # ap.add_argument('--hardware', type=str, required=True)
    ap.add_argument('--input', type=str, required=True, help='path to a video/webcam or an image folder')
    ap.add_argument('--imdir_ini', type=str, default=None, help='path to image folder metadata')
    ap.add_argument('--output', type=str, default=None, help='path to output folder')
    ap.add_argument('--display', action='store_true', help='visualize tracking result')
    ap.add_argument('--save_db', action='store_true', help='save result to database')
    ap.add_argument('--db_framework', type=str, default='pymongo') # pymongo mongoengine
    ap.add_argument('--port', type=int, help='port of localhost to save to database')
    ap.add_argument('--save_txt', action='store_true', help='save to .txt in MOT challenge format')

    opt = ap.parse_args()

    return opt


def main(opt):

    if opt.input == '0':
        opt.input = 0
    elif not os.path.exists(opt.input):
        print('[INFO] Video %s not exists' % opt.input)
        return

    if opt.output is None:
        opt.output = HERE/'../output'
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)
    opt.output = Path(opt.output) # TODO

    t0 = time.time()

    # TODO different options here: if opt.hardware == 'weak':
    sct = SCT()
    detector = sct.create_detector()
    tracker = sct.create_tracker()

    # video loader
    loader_director = LoaderDirector()
    if os.path.isdir(opt.input):    # if input a folder of images
        loader_builder = ImageFolderLoaderBuilder()
        loader_director.set_builder(loader_builder)
        loader_director.build_imagefolderloader(opt.input, opt.imdir_ini)
        loader = loader_builder.get_product()
    else:   # if input a video/cam
        loader_builder = VideoLoaderBuilder()
        loader_director.set_builder(loader_builder)
        loader_director.build_videoloader(opt.input)
        loader = loader_builder.get_product()

    FPS = loader.get_fps()

    filename = os.path.basename(str(opt.input))
    name_root, _ = os.path.splitext(filename)
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    # database
    if opt.save_db:
        print(f"[INFO] Connecting to database using port {opt.port}")
        db_director = DBDirector()
        if opt.db_framework == 'pymongo':
            db_builder = PymongoBuilder()
            db_director.set_builder(db_builder)
            db_director.build_pymongo('localhost', opt.port, 'sct_db', now + '_' + name_root)
            mongo = db_builder.get_product()
        elif opt.db_framework == 'mongoengine':
            db_builder = MongoEngineBuilder()
            db_director.set_builder(db_builder)
            db_director.build_mongoengine('localhost', opt.port, 'sct_db', now + '_' + name_root)
            mongo = db_builder.get_product()
        else:
            raise ValueError(f'{opt.db_framework} not supported')

    if opt.display:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

    if opt.save_txt:
        txt_buffer = []
        out_txt = open(opt.output/(now + '_' + name_root + '.txt'), 'w')

    print('[TIME] Loading models:', time.time() - t0)

    print('[INFO] Processing', opt.input)
    frame_count = 0
    for _ in tqdm(range(len(loader))):
        ret, frame = loader.read()

        if not ret or frame is None or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

        frame_count += 1

        t0 = time.time()

        dets = detector.predict(frame, BGR=True)    # [[x1, y1, x2, y2, conf], ...]

        # print('[INFO] Detect %d people' % dets.shape[0])
        # print('[TIME] Detection:', time.time() - t0)

        t0 = time.time()
        # TODO refactor: adapter
        # TODO tại sao trong code của kalman không dùng tới conf của dets? kiểm tra lại thông số của kalman xem có liên quan không
        tracklets = tracker.update(dets)  # [[id, x1, y1, x2, y2, conf]...]

        ret = np.concatenate([np.array([frame_count] * len(tracklets)).reshape(-1, 1), tracklets], axis=1) # [[frame, id, x1, y1, x2, y2, conf]...]

        # print('[TIME] Tracking:', time.time() - t0)

        if opt.save_db:
            t0 = time.time()
            mongo.update(ret)
            # print('[TIME] Save to database:', time.time() - t0)

        if opt.save_txt:
            t0 = time.time()
            for obj in ret:
                # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
                txt_buffer.append(
                    f'{int(obj[0])}, {int(obj[1])}, {obj[2]:.2f}, {obj[3]:.2f}, {(obj[4] - obj[2]):.2f}, {(obj[5] - obj[3]):.2f}, {obj[6]:.6f}, -1, -1, -1')
            # print('[TIME] Save to .txt:', time.time() - t0)

        if opt.display:
            t0 = time.time()
            # TODO visualizer
            show_img = plot_box(frame, ret) # ret      [np.array([[frame_count]] * len(dets)), np.array([[-1]] * len(dets))
            cv2.imshow(filename, show_img)
            # print('[TIME] Visualization:', time.time() - t0)

    loader.release()
    cv2.destroyAllWindows()

    if opt.save_db:
        mongo.close()

    if opt.save_txt:
        print('\n'.join(txt_buffer), file=out_txt)
        print('[INFO] Result saved in', opt.output/(now + '_' + name_root + '.txt'))
        out_txt.close()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

