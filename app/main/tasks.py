from threading import Thread
from flask import current_app
import time
from queue import Queue

from app import db
from app.models import Camera, Region, CameraMatchingPoint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import MyQueue, Config, Camera as PLCamera, Tracker, SCT as PLSCT, Visualize as PLVisualize, Scene

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

config = Config('mct/utils/config.yaml')

cams = {}


def async_startup(app):
    with app.app_context():

        for c in Camera.query.all():
            cams[c.id] = {'num': c.num, 
                          'address': c.address}
            # break


        import yaml
        for cid, cv in cams.items():
            meta = yaml.safe_load(open(f'data/recordings/2d_v4/meta/{cv["num"]}_00011_2023-04-15_08-30-00-000000.yaml', 'r'))
            iq_sct = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'Input-Queue-SCT<CAMID={cv["num"]}>')
            iq_vis_sct_annot = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'Input-Queue-Vis_SCT-Annot<CAMID={cv["num"]}>')
            iq_vis_sct_video = MyQueue(config.get('QUEUE_MAXSIZE'), name=f'Input-Queue-Vis_SCT-Video<CAMID={cv["num"]}>')
            pl_camera_noretimg = PLCamera(config, cv['address'], meta, [iq_sct])
            pl_camera_retimg = PLCamera(config, cv['address'], meta, [iq_vis_sct_video])
            tracker = Tracker(config.get('DETECTION_MODE'), config.get('TRACKING_MODE'), f'data/recordings/2d_v4/gt_splited/{cv["num"]}_00011_2023-04-15_08-30-00-000000.txt')
            pl_sct = PLSCT(config, tracker, iq_sct, [iq_vis_sct_annot])
            pl_display = PLVisualize(config, 'SCT', iq_vis_sct_annot, iq_vis_sct_video)
            cv['pl_display'] = pl_display
            pl_camera_noretimg.start()
            pl_sct.start()
            pl_camera_retimg.start()
            pl_display.start()






def startup():
    Thread(target=async_startup, args=(current_app._get_current_object(),)).start() # type: ignore







