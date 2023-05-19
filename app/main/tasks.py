from threading import Thread
from flask import current_app
import time
from queue import Queue

from app import db
from app.models import Camera, Region, CameraMatchingPoint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import MyQueue, Config, Camera as PLCamera

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

config = Config('/media/tran/003D94E1B568C6D11/Workingspace/MCT/mct/utils/config.yaml')

cams = {}


def async_startup(app):
    with app.app_context():

        for c in Camera.query.all():
            print(c.id)
            cams[c.id] = {'num': c.num, 
                          'address': c.address} 


        import yaml
        for cid, cv in cams.items():
            meta = yaml.safe_load(open(f'/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/meta/{cv["num"]}_00011_2023-04-15_08-30-00-000000.yaml', 'r'))
            pl_camera = PLCamera(config, cv['address'], meta)
            cv['pl_camera'] = pl_camera
            pl_camera.start()






def startup():
    Thread(target=async_startup, args=(current_app._get_current_object(),)).start() # type: ignore







