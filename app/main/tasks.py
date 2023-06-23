from threading import Thread
from flask import current_app
import time
from queue import Queue
import cv2

from app.extensions import db
from app.models import Camera, Region, CameraMatchingPoint, Scene
import json

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import CameraPipeline as PLCamera, SCTPipeline as PLSCT, VisualizePipeline as PLVisualize, STAPipeline as PLSTA, SyncPipeline as PLSync, MCMapPipeline as PLStaffManager, Tracker, MyQueue


cams = {}
sm_oq = {}


def async_startup(app):
    with app.app_context():

        config = app.config['PIPELINE']

        for c in Camera.query.all():
            cams[c.id] = {'num': c.num, 
                          'address': c.address}
            # break           
            
        sm_iq_sct = {}
        sm_iq_sta = {}
        

        import yaml
        for cid, cv in cams.items():
            meta = yaml.safe_load(open(f'data/recordings/2d_v4/meta/{cv["num"]}_00011_2023-04-15_08-30-00-000000.yaml', 'r'))
            iq_sct = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SCTPipeline<{cid}>')
            
            sm_iq_sct[cid] = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SM_SCT<{cid}>')
            
            pl_camera = PLCamera(config, cv['address'], meta, [iq_sct], ret_img=False, online_put_sleep=config.get('CAMERA_SLEEP_MUL_FACTOR') / meta['fps'], name=f'CameraPipeline<{cid}>')   # Needed to run mock checkin
            cv['pl_camera'] = pl_camera
            
            tracker = Tracker(config.get('DETECTION_MODE'), config.get('TRACKING_MODE'), f'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/{cv["num"]}_00011_2023-04-15_08-30-00-000000.txt')
            cv['pl_sct'] = PLSCT(config, tracker, iq_sct, [sm_iq_sct[cid]], online_put_sleep=pl_camera.online_put_sleep, name=f'SCTPipeline<{cid}>')
            

        stas = {}
        for r in Region.query.filter_by(type='overlap').all():
            cpid = r.primary_cam_id
            pw = cams[cpid]['pl_camera'].width
            ph = cams[cpid]['pl_camera'].height
            proi = np.array(json.loads(r.points))
            proi[:, 0] *= pw
            proi[:, 1] *= ph
            proi = proi.reshape(-1, 1, 2)

            csid = r.secondary_cam_id
            sw = cams[csid]['pl_camera'].width
            sh = cams[csid]['pl_camera'].height
            m = np.array(json.loads(CameraMatchingPoint.query.filter_by(primary_cam_id=csid, secondary_cam_id=cpid).first().points))
            m = m.astype('int32')
            H, _ = cv2.findHomography(m[:, :2], m[:, 2:])
            sroi = cv2.perspectiveTransform(proi, np.linalg.inv(H))
            
            ps = Scene(pw, ph, proi, config.get('ROI_TEST_OFFSET'))
            ss = Scene(sw, sh, sroi, config.get('ROI_TEST_OFFSET'))

            iq_sync_s = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-Sync<(*{csid}, {cpid})>')
            iq_sync_p = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-Sync<({csid}, *{cpid})>')
            cams[csid]['pl_camera'].add_output_queue(iq_sync_s, iq_sync_s.name)
            cams[cpid]['pl_camera'].add_output_queue(iq_sync_p, iq_sync_p.name)

            iq_sta_sct_s = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-STA_SCT<(*{csid}, {cpid})>')
            iq_sta_sct_p = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-STA_SCT<({csid}, *{cpid})>')
            iq_sta_sync = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-STA_Sync<({csid}, {cpid})>')
            cams[csid]['pl_sct'].add_output_queue(iq_sta_sct_s, iq_sta_sct_s.name)
            cams[cpid]['pl_sct'].add_output_queue(iq_sta_sct_p, iq_sta_sct_p.name)
            
            sm_iq_sta[(csid, cpid)] = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'IQ-SM_STA<({csid}, {cpid})>')

            online_put_sleep = min(cams[cpid]['pl_camera'].online_put_sleep, cams[csid]['pl_camera'].online_put_sleep)
            
            stas[(csid, cpid)] = {
                'pl_sync': PLSync(
                    config, 
                    [iq_sync_s, iq_sync_p], 
                    [iq_sta_sync], 
                    online_put_sleep=online_put_sleep, 
                    name=f'Sync<({csid}, {cpid})>'),
                'pl_sta': PLSTA(
                    config, 
                    [ss, ps], 
                    H, 
                    [iq_sta_sct_s, iq_sta_sct_p], 
                    iq_sta_sync, 
                    [sm_iq_sta[(csid, cpid)]], 
                    online_put_sleep=online_put_sleep, 
                    name=f'STAPipeline<({csid}, {cpid})>')
            }

        ckr = Region.query.filter_by(type='checkin').first()
        ckw = cams[ckr.primary_cam_id]['pl_camera'].width
        ckh = cams[ckr.primary_cam_id]['pl_camera'].height
        ckroi = np.array(json.loads(ckr.points))
        ckroi[:, 0] *= ckw
        ckroi[:, 1] *= ckh
        ckroi = ckroi.reshape(-1, 1, 2)
        cks = Scene(ckw, ckh, ckroi, config.get('ROI_TEST_OFFSET'))

        pl_sm = PLStaffManager(
            current_app._get_current_object(), # type: ignore
            config, 
            sm_iq_sct, 
            sm_iq_sta, 
            cks, 
            ckr.primary_cam_id, 
            sm_oq, 
            online_put_sleep=min(cv['pl_camera'].online_put_sleep for cv in cams.values())
        )

        for cid, cv in cams.items():
            cv['pl_camera'].start()

        if config.get('RUNNING_MODE') == 'offline':
            for cid, cv in cams.items():
                cv['pl_camera'].join()
        
        for cid, cv in cams.items():
            cv['pl_sct'].start()

        for cv in stas.values():
            cv['pl_sync'].start()
        
        if config.get('RUNNING_MODE') == 'offline':
            for cid, cv in cams.items():            ##############
                cv['pl_sct'].join()
            
            for cv in stas.values():                ##############
                cv['pl_sync'].join()

        for cv in stas.values():
            cv['pl_sta'].start()
        
        if config.get('RUNNING_MODE') == 'offline':
            for cv in stas.values():
                cv['pl_sta'].join()

        pl_sm.start()






def startup():
    Thread(target=async_startup, args=(current_app._get_current_object(),)).start() # type: ignore







