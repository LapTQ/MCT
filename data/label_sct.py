import os
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from pathlib import Path

import numpy as np
import cv2

import sys
sys.path.append(sys.path[0] + '/..')

from mct.sct import SimpleSCT
from mct.utils.vis_utils import plot_box
from mct.utils.db_utils import Pymongo, MongoEngine
from mct.utils.vid_utils import VideoLoader


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--input', action='append', nargs='+', required=True, help='list of paths to videos or path to a directory of videos')
    ap.add_argument('--from_txt', action='store_true', help='get detection result from txt instead of a tracker')
    ap.add_argument('--save_db', action='store_true', help='save result to database')
    ap.add_argument('--db_framework', type=str, default='pymongo') # pymongo mongoengine
    ap.add_argument('--db_host', type=str, default=None, help='address of db container') # from host: 'localhost', from docker env: 'mct_mongodb'
    ap.add_argument('--db_port', type=int, default=None, help='port of db container') # from host: 1111, from docker env: 27017
    ap.add_argument('--db_name', type=str, default='sct_db', help='name of database')
    ap.add_argument('--output', type=str, default=None, help='path to output folder')
    ap.add_argument('--save_txt', action='store_true', help='save to .txt')
    ap.add_argument('--export_video', action='store_true', help='save visualization as video')
    ap.add_argument('--display', action='store_true', help='visualize tracking result')

    opt = ap.parse_args()

    return opt


def main(opt):

    ########## process input path ###########
    assert len(opt.input) == 1, 'Support <--input vid1 vid2 vid3> or <--input vid_dir> only'
    opt.input = opt.input[0] # [[...]] to [...]
    assert not (os.path.isdir(opt.input[0]) and len(opt.input) != 1), 'Support only 1 directory of videos'
    if os.path.isdir(opt.input[0]):
        # TODO glob video extensions only
        opt.input = [os.path.join(opt.input[0], filename) for filename in os.listdir(opt.input[0])
                     if os.path.splitext(filename)[1] in ['.mp4', '.avi']]
    for i in range(len(opt.input)):
        if not os.path.exists(opt.input[i]):
            print('[ERROR] Video %s not exists' % opt.input[i])
            return
        elif opt.input[i] == '0':
            opt.input[i] = 0
    # opt.input is [path_vid1, path_vid2, path_vid3, ...]

    ########### process output path ############
    if opt.output is None:
        opt.output = str(HERE/'../output')
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)
    opt.output = Path(opt.output)

    # database
    if opt.save_db:
        assert opt.db_host is not None, "--db_host must be specified if --save_db is set"
        assert opt.db_port is not None, "--db_port must be specified if --save_db is set"
        print(f"[INFO] Connecting to {opt.db_host} using port {opt.db_port}")
        sct_db_name = datetime.now().strftime("%Y%m%d%H%M%S") + '_sct'
        if opt.db_framework == 'pymongo':
            mongo = Pymongo.Builder(opt.db_host, opt.db_port
                                    ).set_database(opt.db_name
                                                   ).set_collection(sct_db_name
                                                                    ).get_product()
        elif opt.db_framework == 'mongoengine':
            mongo = MongoEngine.Builder(opt.db_host, opt.db_port
                                        ).set_databse(opt.db_name
                                                      ).set_collection(sct_db_name
                                                                       ).get_product()
        else:
            raise ValueError(f'{opt.db_framework} not supported')

        print(f"[INFO] DB tool: {opt.db_framework}")
        print(f"[INFO] DB database name: {opt.db_name}")
        print(f"[INFO] DB collection name: {sct_db_name}")

    def _do_sct(_input):

        cam_id, vid_id, *record_time = os.path.splitext(os.path.split(_input)[-1])[0].split('_')
        cam_id = int(cam_id)
        vid_id = int(vid_id)
        record_time = datetime.strptime('_'.join(record_time), '%Y-%m-%d_%H-%M-%S-%f')

        t0 = time.time()

        loader = VideoLoader.Builder(_input).get_product()

        if not opt.from_txt:
            sct = SimpleSCT.Builder(loader).get_product()
        else:
            with open(_input[:-3] + 'txt', 'r') as f:
                det_seq = np.array([[eval(e) for e in l.strip().split(',')[:7]] for l in f.readlines()])

        # process output name
        filename = os.path.basename(str(_input))
        stem, _ = os.path.splitext(filename)
        out_stem = stem # datetime.now().strftime("%Y%m%d%H%M%S") + '_' + stem

        FPS = loader.get_fps()

        if opt.display:
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)

        if not opt.from_txt and opt.save_txt:
            txt_buffer = []
            out_txt = open(opt.output / (out_stem + '.txt'), 'w')

        if opt.export_video:
            H = loader.get_height()
            W = loader.get_width()
            out_video = cv2.VideoWriter(str(opt.output / (out_stem + '.avi')),
                                        cv2.VideoWriter_fourcc(*'XVID'),
                                        FPS,
                                        (W, H)
                                        )

        print('[TIME] Loading models:', time.time() - t0)

        print('[INFO] Processing', _input)
        pbar = tqdm(range(len(loader)))
        for _ in pbar:
            # TODO add capture time from loader
            ret, frame = loader.read()

            # terminal condition
            condition = not ret or frame is None
            if opt.display:
                condition = condition or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q')
            if condition:
                break

            start_time = time.time()

            if not opt.from_txt:
                ret = sct.predict(frame, BGR=True)  # [[frame, id, x1, y1, x2, y2, conf]...]
            else:
                ret = det_seq[det_seq[:, 0] == _ + 1]
                ret[:, 4:6] += ret[:, 2:4]


            end_time = time.time()
            pbar.set_postfix({'FPS': int(1 / (end_time - start_time))})

            # print('[TIME] Tracking:', time.time() - t0)

            if opt.save_db:
                if len(ret) > 0:
                    mongo.update(np.concatenate(
                        [np.repeat([[cam_id, vid_id, (record_time + timedelta(seconds=ret[0, 0] / FPS)).timestamp()]],
                                   len(ret), axis=0),
                         ret],
                        axis=1)
                    )  # [[cam, vid, time, frame, id, x1,...],...])

            if not opt.from_txt and opt.save_txt:
                for obj in ret:
                    # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
                    txt_buffer.append(
                        f'{int(obj[0])}, {int(obj[1])}, {obj[2]:.2f}, {obj[3]:.2f}, {(obj[4] - obj[2]):.2f}, {(obj[5] - obj[3]):.2f}, {obj[6]:.6f}, -1, -1, -1')

            if opt.display or opt.export_video:
                show_img = plot_box(frame, ret)  # ret  [np.array([[frame_count]] * len(dets)), np.array([[-1]] * len(dets))

            if opt.display:
                cv2.imshow(filename, show_img)

            if opt.export_video:
                out_video.write(show_img)

        loader.release()

        if not opt.from_txt and opt.save_txt:
            print('\n'.join(txt_buffer), file=out_txt)
            print('[INFO] MOT17-format .txt saved in', opt.output / (out_stem + '.txt'))
            out_txt.close()

        if opt.display:
            cv2.destroyAllWindows()

        if opt.export_video:
            out_video.release()
            print('[INFO] Video demo saved in', str(opt.output / (out_stem + '.avi')))

    threads = [Thread(target=_do_sct, args=(_input,)) for _input in opt.input]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if opt.save_db:
        mongo.close()
        print('[INFO] DB closed')


VID_DIR = os.path.join(HERE, 'recordings')
TXT_DIR = os.path.join(HERE, 'recordings')

def visualize_from_txt(vid_path, txt_path, **kwargs):

    parent, filename = os.path.split(vid_path)

    cap = cv2.VideoCapture(vid_path)
    with open(txt_path, 'r') as f:
        det_seq = np.array([[eval(e) for e in l.strip().split(',')[:7]] for l in f.readlines()])

    if 'vid_path2' in kwargs:
        cap2 = cv2.VideoCapture(kwargs['vid_path2'])
        with open(kwargs['txt_path2'], 'r') as f:
            det_seq2 = np.array([[eval(e) for e in l.strip().split(',')[:7]] for l in f.readlines()])

    if kwargs.get('save_video', False):
        writer = cv2.VideoWriter(os.path.join(parent, 'vis_' + (filename if 'vid_path2' not in kwargs else filename[3:])),
                             cv2.VideoWriter_fourcc(*'XVID'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    for frame_count in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        dets = det_seq[det_seq[:, 0] == frame_count]
        dets[:, 4:6] += dets[:, 2:4]
        success, frame = cap.read()
        if 'vid_path2' not in kwargs:
            vis_img = plot_box(frame, dets)
            show_img = vis_img
        else:
            vid_id = int(filename.split('_')[1])

            dets2 = det_seq2[det_seq2[:, 0] == frame_count]
            dets2[:, 4:6] += dets2[:, 2:4]
            success2, frame2 = cap2.read()

            correspondence = kwargs['correspondence'][kwargs['correspondence'][:, 1] == vid_id]
            for id1, id2 in correspondence[:, [2, 5]]:
                dets[dets[:, 1] == id1, 1] = 100 - id1
                dets2[dets2[:, 1] == id2, 1] = 100 - id1

            vis_img = plot_box(frame, dets)
            vis_img2 = plot_box(frame2, dets2)
            show_img = np.concatenate([vis_img, vis_img2], axis=1)

        if kwargs.get('save_video', False):
            writer.write(show_img)

        cv2.imshow(filename, show_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('e'):
            exit(0)
        elif key == ord(' '):
            cv2.waitKey(0)


    cap.release()
    if kwargs.get('save_video', False):
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)

    vid_list1 = sorted([str(path) for path in Path(VID_DIR).glob('21*.avi')]) # ['21_00000_2022-11-03_14-56-57-643967.avi']
    txt_list1 = sorted([str(path) for path in Path(TXT_DIR).glob('21*.txt')])
    vid_list2 = sorted([str(path) for path in Path(VID_DIR).glob('27*.avi')])
    txt_list2 = sorted([str(path) for path in Path(TXT_DIR).glob('27*.txt')])

    correspondence = np.loadtxt('recordings/correspondences.txt', delimiter=',', dtype=int)

    for vid_path1, txt_path1, vid_path2, txt_path2 in tqdm(zip(vid_list1, txt_list1, vid_list2, txt_list2)):
        visualize_from_txt(vid_path1, txt_path1, save_video=False, vid_path2=vid_path2, txt_path2=txt_path2, correspondence=correspondence)

