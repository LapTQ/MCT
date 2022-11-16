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

from mct.sct import SimpleSCT
from mct.utils.vis_utils import plot_box
from mct.utils.db_utils import Pymongo, MongoEngine
from mct.utils.vid_utils import VideoLoader, ImageFolderLoader


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    # ap.add_argument('--hardware', type=str, required=True)
    ap.add_argument('--input', type=str, required=True, help='path to a video/webcam or an image folder')
    ap.add_argument('--imdir_ini', type=str, default=None, help='path to image folder metadata')
    ap.add_argument('--save_db', action='store_true', help='save result to database')
    ap.add_argument('--db_framework', type=str, default='pymongo') # pymongo mongoengine
    ap.add_argument('--db_host', type=str, default=None, help='address of db container') # from host: 'localhost', from docker env: 'mct_mongodb'
    ap.add_argument('--db_port', type=int, default=None, help='port of db container') # from host: 1111, from docker env: 27017
    ap.add_argument('--db_name', type=str, default='sct_db', help='name of database')
    ap.add_argument('--output', type=str, default=None, help='path to output folder')
    ap.add_argument('--save_txt', action='store_true', help='save to .txt in MOT challenge format')
    ap.add_argument('--export_video', action='store_true', help='save visualization as video')
    ap.add_argument('--display', action='store_true', help='visualize tracking result')

    opt = ap.parse_args()

    return opt


def main(opt):

    # process input path
    if opt.input == '0':
        opt.input = 0
    elif not os.path.exists(opt.input):
        print('[INFO] Video %s not exists' % opt.input)
        return

    # process output path
    if opt.output is None:
        opt.output = str(HERE/'../output')
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)
    opt.output = Path(opt.output)

    t0 = time.time()

    # video loader
    if os.path.isdir(opt.input):    # if input a folder of images
        loader = ImageFolderLoader.Builder(opt.input, opt.imdir_ini).get_product()
    else:                           # if input a video/cam
        loader = VideoLoader.Builder(opt.input).get_product()

    # create detector and tracker
    # TODO different options here: if opt.hardware == 'weak':
    sct = SimpleSCT.Builder(loader).get_product()

    FPS = loader.get_fps()

    # process output name
    filename = os.path.basename(str(opt.input))
    name_root, _ = os.path.splitext(filename)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name_root = now + '_' + name_root

    # database
    if opt.save_db:
        assert opt.db_host is not None, "--db_host must be specified if --save_db is set"
        assert opt.db_port is not None, "--db_port must be specified if --save_db is set"
        print(f"[INFO] Connecting to {opt.db_host} using port {opt.db_port}")
        if opt.db_framework == 'pymongo':
            mongo = Pymongo.Builder(opt.db_host, opt.db_port
                                    ).set_database(opt.db_name
                                                   ).set_collection(out_name_root
                                                                    ).get_product()
        elif opt.db_framework == 'mongoengine':
            mongo = MongoEngine.Builder(opt.db_host, opt.db_port
                                        ).set_databse(opt.db_name
                                                      ).set_collection(out_name_root
                                                                    ).get_product()
        else:
            raise ValueError(f'{opt.db_framework} not supported')

        print(f"[INFO] DB tool: {opt.db_framework}")
        print(f"[INFO] DB database name: {opt.db_name}")
        print(f"[INFO] DB collection name: {out_name_root}")

    if opt.display:
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

    if opt.save_txt:
        txt_buffer = []
        out_txt = open(opt.output/(out_name_root + '.txt'), 'w')

    if opt.export_video:
        H = loader.get_height()
        W = loader.get_width()
        out_video = cv2.VideoWriter(str(opt.output/(out_name_root + '.avi')),
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    FPS,
                                    (W, H)
        )

    print('[TIME] Loading models:', time.time() - t0)

    print('[INFO] Processing', opt.input)
    pbar = tqdm(range(len(loader)))
    for _ in pbar:

        ret, frame = loader.read()

        # terminal condition
        condition = not ret or frame is None
        if opt.display:
            condition = condition or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q')
        if condition:
            break

        start_time = time.time()

        ret = sct.predict(frame, BGR=True)  # [[frame, id, x1, y1, x2, y2, conf]...], frame from 0

        end_time = time.time()
        pbar.set_postfix({'FPS': int(1/(end_time - start_time))})

        # print('[TIME] Tracking:', time.time() - t0)

        if opt.save_db:
            # t0 = time.time()
            mongo.update(ret)
            # print('[TIME] Save to database:', time.time() - t0)

        if opt.save_txt:
            # t0 = time.time()
            for obj in ret:
                # [frame, id, x1, y1, w, h, conf, -1, -1, -1], frame from 1
                txt_buffer.append(
                    f'{int(obj[0])}, {int(obj[1])}, {obj[2]:.2f}, {obj[3]:.2f}, {(obj[4] - obj[2]):.2f}, {(obj[5] - obj[3]):.2f}, {obj[6]:.6f}, -1, -1, -1')
            # print('[TIME] Save to .txt:', time.time() - t0)

        if opt.display or opt.export_video:
            t0 = time.time()
            show_img = plot_box(frame, ret)     # ret      [np.array([[frame_count]] * len(dets)), np.array([[-1]] * len(dets))

        if opt.display:
            cv2.imshow(filename, show_img)

        if opt.export_video:
            out_video.write(show_img)

    loader.release()

    if opt.save_db:
        mongo.close()
        print('[INFO] DB closed')

    if opt.save_txt:
        print('\n'.join(txt_buffer), file=out_txt)
        print('[INFO] MOT17-format .txt saved in', opt.output/(out_name_root + '.txt'))
        out_txt.close()

    if opt.display:
        cv2.destroyAllWindows()

    if opt.export_video:
        out_video.release()
        print('[INFO] Video demo saved in', str(opt.output/(out_name_root + '.avi')))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

