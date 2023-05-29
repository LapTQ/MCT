from threading import Thread
from flask import current_app
import time
from queue import Queue
import cv2

from app import db
from app.models import Camera, Region, CameraMatchingPoint
import json

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import MyQueue, Config, Camera as PLCamera, Tracker, SCT as PLSCT, Visualize as PLVisualize, Scene, STA as PLSTA, SyncFrame as PLSync, StaffMap as PLStaffMap

mylist = Queue()


# def async_startup(app, seconds):
#     with app.app_context():
#         print('Task start')
#         i = 0
#         while True:
#             time.sleep(1)
#             mylist.put(f'counting {i}')
#             print(i)
#             i += 1
            
#         print('Task complete')

config = Config('data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/config_pred_mct_trackertracker_18.yaml')

cams = {}


def async_startup(app):
    with app.app_context():

        for c in Camera.query.all():
            cams[c.id] = {'num': c.num, 
                          'address': c.address}
            # break           
            
        sm_iq_sct = {}
        sm_iq_sta = {}

        import yaml
        for cid, cv in cams.items():
            meta = yaml.safe_load(open(f'data/recordings/2d_v4/meta/{cv["num"]}_00011_2023-04-15_08-30-00-000000.yaml', 'r'))
            iq_sct = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SCT<{cid}>')
            iq_vis_sct_annot = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-Vis_SCT-Annot<{cid}>')
            iq_vis_sct_video = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-Vis_SCT-Video<{cid}>')
            
            sm_iq_sct[cid] = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SM_SCT<{cid}>')
            
            pl_camera_noretimg = PLCamera(config, cv['address'], meta, [iq_sct], name=f'CameraNoRet<{cid}>')   # Needed to run mock checkin
            pl_camera_retimg = PLCamera(config, cv['address'], meta, [iq_vis_sct_video], name=f'CameraRet<{cid}>')
            
            tracker = Tracker(config.get('DETECTION_MODE'), config.get('TRACKING_MODE'), f'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/{cv["num"]}_00011_2023-04-15_08-30-00-000000.txt')
            pl_sct = PLSCT(config, tracker, iq_sct, [iq_vis_sct_annot, sm_iq_sct[cid]], name=f'SCT<{cid}>')
            
            
            pl_display = PLVisualize(config, 'SCT', iq_vis_sct_annot, iq_vis_sct_video, name=f'Visualize<{cid}>')
            cv['pl_camera_noretimg'] = pl_camera_noretimg
            cv['pl_camera_retimg'] = pl_camera_retimg
            cv['pl_sct'] = pl_sct
            cv['pl_display'] = pl_display
            

        stas = {}
        for r in Region.query.filter_by(type='overlap').all():
            cpid = r.primary_cam_id
            pw = cams[cpid]['pl_camera_noretimg'].width
            ph = cams[cpid]['pl_camera_noretimg'].height
            proi = np.array(json.loads(r.points))
            proi[:, 0] *= pw
            proi[:, 1] *= ph
            proi = proi.reshape(-1, 1, 2)
            ps = Scene(pw, ph, proi, config.get('ROI_TEST_OFFSET'))

            csid = r.secondary_cam_id
            sw = cams[csid]['pl_camera_noretimg'].width
            sh = cams[csid]['pl_camera_noretimg'].height
            m = np.array(json.loads(CameraMatchingPoint.query.filter_by(primary_cam_id=csid, secondary_cam_id=cpid).first().points))
            m = m.astype('int32')
            H, _ = cv2.findHomography(m[:, :2], m[:, 2:])
            sroi = cv2.perspectiveTransform(proi, np.linalg.inv(H))
            
            ps = Scene(pw, ph, proi, config.get('ROI_TEST_OFFSET'))
            ss = Scene(sw, sh, sroi, config.get('ROI_TEST_OFFSET'))

            iq_sync_s = MyQueue(config.get('QUEUE_MAXSIZE'))
            iq_sync_p = MyQueue(config.get('QUEUE_MAXSIZE'))
            cams[csid]['pl_camera_noretimg'].add_output_queue(iq_sync_s, f'IQ-Sync<(*{csid}, {cpid})>')
            cams[cpid]['pl_camera_noretimg'].add_output_queue(iq_sync_p, f'IQ-Sync<({csid}, *{cpid})>')

            iq_sta_sct_s = MyQueue(config.get('QUEUE_MAXSIZE'))
            iq_sta_sct_p = MyQueue(config.get('QUEUE_MAXSIZE'))
            iq_sta_sync = MyQueue(config.get('QUEUE_MAXSIZE'))
            cams[csid]['pl_sct'].add_output_queue(iq_sta_sct_s, f'IQ-STA_SCT<(*{csid}, {cpid})>')
            cams[cpid]['pl_sct'].add_output_queue(iq_sta_sct_p, f'IQ-STA_SCT<({csid}, *{cpid})>')
            
            sm_iq_sta[(csid, cpid)] = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SM_STA<({csid}, {cpid})>')
            
            stas[(csid, cpid)] = {
                'pl_sync': PLSync(config, [iq_sync_s, iq_sync_p], iq_sta_sync, name=f'Sync<({csid}, {cpid})>'),
                'pl_sta': PLSTA(config, [ss, ps], H, [iq_sta_sct_s, iq_sta_sct_p], iq_sta_sync, [sm_iq_sta[(csid, cpid)]], name=f'STA<({csid}, {cpid})>')
            }

        ckr = Region.query.filter_by(type='checkin').first()
        ckw = cams[ckr.primary_cam_id]['pl_camera_noretimg'].width
        ckh = cams[ckr.primary_cam_id]['pl_camera_noretimg'].height
        ckroi = np.array(json.loads(ckr.points))
        ckroi[:, 0] *= ckw
        ckroi[:, 1] *= ckh
        ckroi = ckroi.reshape(-1, 1, 2)
        cks = Scene(ckw, ckh, ckroi, config.get('ROI_TEST_OFFSET'))

        pl_sm = PLStaffMap(config, sm_iq_sct, sm_iq_sta, cks, ckr.primary_cam_id)

        for cid, cv in cams.items():
            cv['pl_camera_noretimg'].start()
            # cv['pl_camera_retimg'].start()
            cv['pl_sct'].start()
            # cv['pl_display'].start()

        for cv in stas.values():
            cv['pl_sync'].start()
            cv['pl_sta'].start()

        pl_sm.start()






def startup():
    Thread(target=async_startup, args=(current_app._get_current_object(),)).start() # type: ignore







