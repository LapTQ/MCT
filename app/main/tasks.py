from threading import Thread
from flask import current_app
import cv2

from app.extensions import monitor, db
from app.entities import User, Camera, Region, CameraMatchingPoint, Message, Notification, Productivity, Detection, STA
import json
import datetime

import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mct.sta.engines import Scene


def async_startup(app):
    with app.app_context():

        ########### MOCK TEST ###########
        for entity in [Message, Notification, Productivity, Detection, STA]:
            for record in entity.query.all():
                db.session.delete(record)
        for u in User.query.all():
            u.last_message_read_time = datetime.datetime(2001, 5, 24)
        db.session.commit()
        #################################

        config = app.config['PIPELINE']
        W = config.get('CAMERA_FRAME_WIDTH')
        H = config.get('CAMERA_FRAME_HEIGHT')

        assert W is not None
        assert H is not None

        # register cameras
        for camera in Camera.query.all():
            
            video_name = os.path.split(camera.address)[1][:-4]
            meta_path = f'data/recordings/2d_v4/meta/{video_name}.yaml'

            ######### MOCK TEST #########
            txt_path = f'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/{video_name}.txt'
            #############################

            monitor.register_camera(
                cam_id=camera.id,
                address=camera.address,
                meta_path=meta_path,
                txt_path=txt_path,
            )

        # register overlapping regions
        for region in Region.query.filter_by(type='overlap').all():
            
            # create scene object for primary camera
            roi_primary = np.array(json.loads(region.points))
            roi_primary[:, 0] *= W
            roi_primary[:, 1] *= H
            roi_primary = roi_primary.reshape(-1, 1, 2)
            scene_primary = Scene(W, H, roi_primary, config.get('ROI_TEST_OFFSET'))

            # create scene object for secondary camera
            # note that in camera matching points, the primary (source) and secondary (destination) camera are reversed
            matches = np.array(json.loads(
                CameraMatchingPoint.query.filter_by(
                    primary_cam_id=region.secondary_cam_id, 
                    secondary_cam_id=region.primary_cam_id,
                ).first().points
            )).astype('int32')
            homo, _ = cv2.findHomography(matches[:, :2], matches[:, 2:])
            roi_secondary = cv2.perspectiveTransform(roi_primary, np.linalg.inv(homo))
            scene_secondary = Scene(W, H, roi_secondary, config.get('ROI_TEST_OFFSET'))

            # register overlap
            monitor.register_overlap(
                cam_id_primary=region.primary_cam_id,
                cam_id_secondary=region.secondary_cam_id,
                scene_primary=scene_primary,
                scene_secondary=scene_secondary,
                homo=homo
            )

        # register check-in region
        region = Region.query.filter_by(type='checkin').first()
        roi = np.array(json.loads(region.points))
        roi[:, 0] *= W
        roi[:, 1] *= H
        roi = roi.reshape(-1, 1, 2)
        scene = Scene(W, W, roi, config.get('ROI_TEST_OFFSET'))
        monitor.register_checkin(
            cam_id=region.primary_cam_id,
            scene=scene
        )

        # start
        monitor.start()


def startup():
    Thread(target=async_startup, args=(current_app._get_current_object(),)).start() # type: ignore







