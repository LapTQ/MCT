import csv
import os
from pathlib import Path
import json
import cv2
import numpy as np
from xml.dom import minidom
from collections import namedtuple
from scipy.spatial.transform import Rotation
import logging
import sys
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(funcName)20s |%(lineno)d\t|%(levelname)8s |%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


HERE = Path(__file__).parent

sys.path.append(str(HERE.parent))
from mct.utils.find_homo import select_ROI


Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'view_id'])
Bbox = namedtuple('Bbox', ['xc', 'yc', 'w', 'h'])


def load_mapview_projection(file_path):

    with open(file_path, 'r') as f:
        matching = json.load(f)
    
    map = np.array(matching['map'])
    cam = np.array(matching['camera'])

    return cam, map


def calc_H(src, dst):
    H, mask = cv2.findHomography(src, dst)
    return H


def load_PETS_single_calibration(path):

    Geometric = namedtuple('Geometric', ['w', 'h', 'ncx', 'nfx', 'dx', 'dy', 'dpx', 'dpy'])
    Intrinsic = namedtuple('Intrinsic', ['f', 'kappa1', 'cx', 'cy', 'sx'])
    Extrinsic = namedtuple('Exctrinsic', ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])

    gt_xml = minidom.parse(path)

    camera = gt_xml.getElementsByTagName('Camera')

    assert len(camera) == 1

    geom = gt_xml.getElementsByTagName('Geometry')[0]
    intr = gt_xml.getElementsByTagName('Intrinsic')[0]
    extr = gt_xml.getElementsByTagName('Extrinsic')[0]

    geom = Geometric(w=int(geom.attributes["width"].value), h=int(geom.attributes["height"].value), ncx=float(geom.attributes["ncx"].value), nfx=float(geom.attributes["nfx"].value), dx=float(geom.attributes["dx"].value), dy=float(geom.attributes["dy"].value), dpx=float(geom.attributes["dpx"].value), dpy=float(geom.attributes["dpy"].value))
    intr = Intrinsic(f=float(intr.attributes["focal"].value), kappa1=float(intr.attributes["kappa1"].value), cx=float(intr.attributes["cx"].value), cy=float(intr.attributes["cy"].value), sx=float(intr.attributes["sx"].value))
    extr = Extrinsic(tx=float(extr.attributes["tx"].value), ty=float(extr.attributes["ty"].value), tz=float(extr.attributes["tz"].value), rx=float(extr.attributes["rx"].value), ry=float(extr.attributes["ry"].value), rz=float(extr.attributes["rz"].value))

    K = np.array([[(intr.sx/geom.dpx)*intr.f, 0, intr.cx], [0, (1/geom.dpy)*intr.f, intr.cy], [0, 0, 1]])
    try:
        R = Rotation.from_euler('xyz', [extr.rx, extr.ry, extr.rz], degrees=False).as_matrix()
    except:
        R = Rotation.from_euler('xyz', [extr.rx, extr.ry, extr.rz], degrees=False).as_dcm()

    T = np.array([[extr.tx, extr.ty, extr.tz]]).T

    return K, R, T


def load_PETS_multi_calibration(root_path, view_list=[1,5,6,7,8]):
    
    calib_list = list()
    for view_id in view_list:
        camera_matrix_path = os.path.join(root_path, 'View_{0:03d}.xml'.format(view_id))
        K, R, T = load_PETS_single_calibration(str(camera_matrix_path))
        calib_list.append(Calibration(K=K, R=R, T=T, view_id=view_id))
        
    return calib_list


def calc_projection_matrix(K, R, T):
    M_int = np.hstack([K, [[0], [0], [0]]])
    M_ext = np.vstack([np.hstack([R, T.reshape(3, 1)]), [0, 0, 0, 1]])
    P = M_int @ M_ext
    return P


def get_matching_points(P, n=140, world_size=100000, seed=42):

    # sample world points
    np.random.seed(seed)
    W = np.vstack([np.random.randint(world_size, size=2*n).reshape(2, -1), np.tile([0, 1], (n, 1)).T])

    # get image points
    I = P @ W
    I = I / I[2, :]
    return I[:2, :].T


def load_gt(file_path, save_path=None):

    logging.info('Loading ground truth annotation from {}'.format(file_path))

    ret = []
    f = open(file_path, 'r')
    reader = csv.reader(f, delimiter=' ')

    for row in reader:
        # filter out lost tracks
        if row[6] == '1':
            logging.warning('Skipping lost track, please double check')
            continue

        tid, x1, y1, x2, y2, fid = map(int, row[:6])

        # convert to [[frame, id, x1, y1, w, h, conf, ...],...] (MOT format)
        fid += 1
        w = x2 - x1
        h = y2 - y1

        ret.append([fid, tid, x1, y1, w, h, 1, 1, 1])
    
    f.close()
    
    if save_path is not None:

        logging.info('Saving ground truth annotation to {}'.format(save_path))

        parent, _ = os.path.split(save_path)
        os.makedirs(parent, exist_ok=True)

        ret_str = []
        for line in ret:
            ret_str.append('{},{},{:.1f},{:.1f},{:.1f},{:.1f},{},{},{:.1f}'.format(*line))
        ret_str = '\n'.join(ret_str)
        with open(save_path, 'w') as f:
            f.write(ret_str)
    
    else:
        logging.warning('No save path specified, skipping saving annotation')
                
    return ret


def check_calibration():

    DATA_DIR = HERE / 'PETS09'
    BASE_CAM_ID = 1
    CAM_LIST = [1, 7]

    FRAME_ID = 755

    calibs = load_PETS_multi_calibration(str(DATA_DIR), CAM_LIST)
    matching = {}
    for idx, cam_id in enumerate(CAM_LIST):
        K,R,T = calibs[idx].K, calibs[idx].R, calibs[idx].T
        P = calc_projection_matrix(K, R, T)
        matching[cam_id] = get_matching_points(P)
    
    H_list = {
        cam_id: calc_H(matching[cam_id], matching[BASE_CAM_ID])  for cam_id in CAM_LIST if cam_id != BASE_CAM_ID
    }

    for frame_id in range(795):
        
        show_imgs = {}
        for cam_id in CAM_LIST:    
            img = cv2.imread(str(DATA_DIR / f'View_{cam_id:03d}' / f'frame_{frame_id:04d}.jpg'))
            show_imgs[cam_id] = img

        show = show_imgs[BASE_CAM_ID].astype('int32')
        for cam_id in CAM_LIST:
            if cam_id != BASE_CAM_ID:
                show += cv2.warpPerspective(show_imgs[cam_id], H_list[cam_id], show_imgs[BASE_CAM_ID].shape[:2][::-1], flags=cv2.INTER_LINEAR)
                
        show = np.clip(show / len(CAM_LIST), 0, 255).astype('uint8')
        
        cv2.imshow(str(cam_id), show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)







if __name__ == "__main__":
    
    DATA_DIR = HERE / 'PETS09'
    TARGET_DIR = HERE / 'recordings/PETS09/'
    CAMERA_LIST = [1, 5, 6, 7]
    SRC2DST_CAM_LIST = [(5, 1), (6, 1), (7, 1), (6, 5), (7, 5), (7, 6)]
    FPS = 7.0
    FAKE_START_TIME = '2023-10-24_04-00-00-000000'
    FAKE_VID_ID = '00000'

    
    os.makedirs(str(TARGET_DIR), exist_ok=True)
    NAME_POSTFIX = FAKE_VID_ID + '_' + FAKE_START_TIME

    # create annotations files
    GT_DIR = TARGET_DIR / 'gt'
    os.makedirs(str(GT_DIR), exist_ok=True)

    gt_annots = {}

    for cam_id in CAMERA_LIST:
        annot = load_gt(
            file_path=str(DATA_DIR / 'View_{:03d}.txt'.format(cam_id)), 
            save_path=str(GT_DIR / '{}_{}.txt').format(cam_id, NAME_POSTFIX)
        )

        gt_annots[cam_id] = annot
    
    
    # create meta files
    META_DIR = TARGET_DIR / 'meta'
    os.makedirs(str(META_DIR), exist_ok=True)
    for cam_id in CAMERA_LIST:

        meta_path = str(META_DIR / '{}_{}.yaml').format(cam_id, NAME_POSTFIX)

        logging.info('Creating meta file {}'.format(meta_path))

        sample_img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(cam_id) / 'frame_0000.jpg'))
        with open(meta_path, 'w') as f:
            f.write('\n'.join([
                'name: {}_{}.avi'.format(cam_id, NAME_POSTFIX),
                'cam_id: {}'.format(cam_id),
                'video_id: 0',
                'fps: {}'.format(FPS),
                'width: {}'.format(sample_img.shape[1]),
                'height: {}'.format(sample_img.shape[0]),
                'start_time: {}'.format(FAKE_START_TIME),
                'start_frame_id: 1',
                'frame_count: 795'
            ]))

    
    # create Re-ID result dir
    os.makedirs(str(TARGET_DIR / 'Re-ID' / NAME_POSTFIX), exist_ok=True)

    
    # create true mct gt-gt file
    true_mct_gtgt_path = TARGET_DIR / 'true_mct_gtgt.txt'
    logging.warning('Creating true mct gt-gt file at {}'.format(str(true_mct_gtgt_path)))
    true_mct_gtgt = open(str(true_mct_gtgt_path), 'w')
    id_list = {
        cam_id: np.unique(np.array(gt_annots[cam_id], dtype='int32')[:, 1]) for cam_id in CAMERA_LIST
    }
    pair_matchings = []
    for cid1, tid1_list in id_list.items():
        for cid2, tid2_list in id_list.items():
            if cid1 < cid2:
                for tid1 in tid1_list:
                    for tid2 in tid2_list:
                        if tid1 == tid2:
                            pair_matchings.append('{},{},{},{},{},{}'.format(cid1, 0, tid1, cid2, 0, tid2))
    true_mct_gtgt.write('\n'.join(pair_matchings))
    true_mct_gtgt.close()

    
    # # create videos
    # VIDEO_DIR = TARGET_DIR / 'videos'
    # os.makedirs(str(VIDEO_DIR), exist_ok=True)
    # for cam_id in CAMERA_LIST:
    #     video_path = str(VIDEO_DIR / '{}_{}.avi').format(cam_id, NAME_POSTFIX)
    #     sample_img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(cam_id) / 'frame_0000.jpg'))

    #     logging.info('Creating video file {}'.format(video_path))

    #     video_writer = cv2.VideoWriter(
    #         video_path,
    #         cv2.VideoWriter_fourcc(*'XVID'),
    #         FPS,
    #         (sample_img.shape[1], sample_img.shape[0])
    #     )

    #     for frame_id in tqdm(range(795)):
    #         img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(cam_id) / 'frame_{:04d}.jpg'.format(frame_id)))
    #         video_writer.write(img)
    #     video_writer.release()


    # create fake SCT tracker result from GT
    logging.warning('Creating fake SCT tracker result from GT')

    FAKE_TRACKER_DIR = TARGET_DIR / 'GTTrackerbox' / 'sct'
    os.makedirs(str(FAKE_TRACKER_DIR), exist_ok=True)

    for cam_id in CAMERA_LIST:
        annot = load_gt(
            file_path=str(DATA_DIR / 'View_{:03d}.txt'.format(cam_id)), 
            save_path=str(FAKE_TRACKER_DIR / '{}_{}.txt').format(cam_id, NAME_POSTFIX)
        )

    
    # create matching points between cameras
    calibs = load_PETS_multi_calibration(str(DATA_DIR), CAMERA_LIST)
    matching = {}
    
    for idx, cam_id in enumerate(CAMERA_LIST):
        K,R,T = calibs[idx].K, calibs[idx].R, calibs[idx].T
        P = calc_projection_matrix(K, R, T)
        matching[cam_id] = get_matching_points(P)
    
    for src_cam_id, dst_cam_id in SRC2DST_CAM_LIST:

        cat = np.concatenate([matching[src_cam_id], matching[dst_cam_id]], axis=1)
        matches_path = str(TARGET_DIR / 'matches_{}_to_{}.txt'.format(src_cam_id, dst_cam_id))
        logging.warning('Createing matching points between camera {} (source) and {} (target) at {}'.format(src_cam_id, dst_cam_id, matches_path))
        np.savetxt(matches_path, cat)

        sample_src_img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(src_cam_id) / 'frame_0000.jpg'))
        sample_dst_img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(dst_cam_id) / 'frame_0000.jpg'))
        H = calc_H(matching[src_cam_id], matching[dst_cam_id])
        contour = select_ROI(sample_src_img, sample_dst_img, H)
        dst_roi_path = str(TARGET_DIR / 'roi_{}_wrt_{}.txt'.format(dst_cam_id, src_cam_id))
        logging.warning('Saving overlapping ROI on camera {} (w.r.t camera {}) at {}'.format(dst_cam_id, src_cam_id, dst_roi_path))
        np.savetxt(dst_roi_path, contour.reshape(-1, 2))

                            


