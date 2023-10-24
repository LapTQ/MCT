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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(funcName)20s |%(lineno)d\t|%(levelname)8s |%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


HERE = Path(__file__).parent

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
  

def sample_world_points(n=140, world_size=100000, seed=42):
    """Return n points in the world coordinates arranged in a matrix.
    Each column is a point (xw, yw, 0, 1) (i.e., zw must be 0)"""
    np.random.seed(seed)
    return np.vstack([np.random.randint(world_size, size=2*n).reshape(2, -1), np.tile([0, 1], (n, 1)).T])


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

        for row in reader:
            tid, x1, y1, x2, y2, fid = map(int, row[:6])

            # convert to [[frame, id, x1, y1, w, h, conf, ...],...] (MOT format)
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


if __name__ == '__main__':

    DATA_DIR = HERE / 'PETS09'
    BASE_CAM_ID = 1
    CAM_LIST = [1, 7]

    FRAME_ID = 755

    calibs = load_PETS_multi_calibration(str(DATA_DIR), CAM_LIST)
    matching = {}
    for idx, cam_id in enumerate(CAM_LIST):
        K,R,T = calibs[idx].K, calibs[idx].R, calibs[idx].T
        M_int = np.hstack([K, [[0], [0], [0]]])
        M_ext = np.vstack([np.hstack([R, T.reshape(3, 1)]), [0, 0, 0, 1]])
        P = M_int @ M_ext
        
        W = sample_world_points()
        I = P @ W
        I = I / I[2, :]

        matching[cam_id] = I[:2, :].T
    
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







# if __name__ == "__main__":
    
#     DATA_DIR = HERE / 'PETS09'
#     TARGET_DIR = HERE / 'recordings/PETS09/'
#     CAMERA_LIST = [1, 5, 6, 7, 8]
#     FAKE_START_TIME = '2023-10-24_04-00-00-000000'
#     FAKE_VID_ID = '00000'
#     FPS = 7.0

    
#     os.makedirs(str(TARGET_DIR), exist_ok=True)
#     NAME_POSTFIX = FAKE_VID_ID + '_' + FAKE_START_TIME

#     # create annotations files
#     GT_DIR = TARGET_DIR / 'gt'
#     os.makedirs(str(GT_DIR), exist_ok=True)
#     for cam_id in CAMERA_LIST:
#         load_gt(
#             file_path=str(DATA_DIR / 'View_{:03d}.txt'.format(cam_id)), 
#             save_path=str(GT_DIR / '{}_{}.txt').format(cam_id, NAME_POSTFIX)
#         )
    
#     # create meta files
#     META_DIR = TARGET_DIR / 'meta'
#     os.makedirs(str(META_DIR), exist_ok=True)
#     for cam_id in CAMERA_LIST:

#         meta_path = str(META_DIR / '{}_{}.yaml').format(cam_id, NAME_POSTFIX)

#         logging.info('Creating meta file {}'.format(meta_path))

#         sample_img = cv2.imread(str(DATA_DIR / 'View_{:03d}'.format(cam_id) / 'frame_0000.jpg'))
#         with open(meta_path, 'w') as f:
#             f.write('\n'.join([
#                 'name: {}_{}.avi'.format(cam_id, NAME_POSTFIX),
#                 'cam_id: {}'.format(cam_id),
#                 'video_id: 0',
#                 'fps: {}'.format(FPS),
#                 'width: {}'.format(sample_img.shape[1]),
#                 'height: {}'.format(sample_img.shape[0]),
#                 'start_time: {}'.format(FAKE_START_TIME),
#                 'start_frame_id: 0',
#                 'frame_count: 795'
#             ]))


