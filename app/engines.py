import logging
import yaml
from threading import Lock
from app.entities import CameraMatchingPoint, Region, User
from mct.sta.base import ConfigPipeline, Pipeline
from mct.sta.engines import MyQueue, Scene, Tracker
from mct.sta.pipeline import CameraPipeline, SCTPipeline, STAPipeline, SyncPipeline
from mct.utils.draw import plot_box, plot_loc, plot_roi
from mct.utils.general import calc_loc

import cv2
import numpy as np

import json
import time
from typing import Union


logger = logging.getLogger(__name__)


class VisualizePipeline(Pipeline):

    def __init__(
            self,
            app,
            db,
            config: ConfigPipeline,
            cam_id: int,
            annot_queue: MyQueue,
            video_queue: MyQueue,
            online_put_sleep: Union[int, float] = 0,
            name='Visualizer'
    ) -> None:
        super().__init__(config, online_put_sleep, name)

        self.app = app
        self.db = db

        self.cam_id = cam_id
        self.annot_queue = annot_queue
        self.video_queue = video_queue
        self._check_video_queue()

        self._load_scenes()

        self.wait_list = self._new_wait_list()

        logger.info(f'{self.name}:\t initialized')


    def _load_scenes(self):

        self.SCENE_COLORS = {
            'overlap': (0, 255, 0),
            'checkin': (255, 0, 0),
            'workarea': (0, 0, 255),
        }

        self.scenes = {k: [] for k in self.SCENE_COLORS}
        W = self.config.get('CAMERA_FRAME_WIDTH')
        H = self.config.get('CAMERA_FRAME_HEIGHT')
        assert W is not None
        assert H is not None
        with self.app.app_context():
            for r in Region.query.filter_by(primary_cam_id=self.cam_id).all():
                roi = np.array(json.loads(r.points)) # type: ignore
                roi[:, 0] *= W
                roi[:, 1] *= H
                roi = roi.reshape(-1, 1, 2)
                scene = Scene(W, H, roi, self.config.get('ROI_TEST_OFFSET'))
                self.scenes[r.type].append(scene)

            for r in Region.query.filter_by(secondary_cam_id=self.cam_id, type='overlap').all():
                roi = np.array(json.loads(r.points))
                roi[:, 0] *= W
                roi[:, 1] *= H
                roi = roi.reshape(-1, 1, 2)
                matches = np.array(json.loads(
                    CameraMatchingPoint.query.filter_by(
                        primary_cam_id=r.secondary_cam_id,
                        secondary_cam_id=r.primary_cam_id,
                    ).first().points
                )).astype('int32')
                homo, _ = cv2.findHomography(matches[:, :2], matches[:, 2:])
                roi = cv2.perspectiveTransform(roi, np.linalg.inv(homo))
                scene = Scene(W, H, roi, self.config.get('ROI_TEST_OFFSET'))
                self.scenes[r.type].append(scene)


    def _start(self):

        assert self.config.get('RUNNING_MODE') == 'online'

        start_time = time.time()
        pop_signin_count = 0
        pop_signin_id = None

        still_wait = [True, True]

        while not self.is_stopped():

            self.trigger_pause()

            for i, (wl, iq) in enumerate(
                zip(self.wait_list, [self.annot_queue,
                                     self.video_queue])
            ):
                if still_wait[i]:
                    item = iq.get(block=True)
                    if item == '<EOS>':
                        still_wait[i] = False
                    else:
                        wl.append(item)

            if True not in still_wait:
                # self._put_to_output_queues('<EOS>')
                logger.info(f'{self.name}:\t reached <EOS> token')
                self.wait_list = self._new_wait_list()  # release memory TODO very naive
                break

            active_list = self.wait_list
            self.wait_list = self._new_wait_list()

            adict = {}
            for i, al in enumerate(active_list):
                for item in al:
                    for k, v in item.items():
                        if k not in adict:
                            adict[k] = [[], []]
                        adict[k][i].append(v)

            if len(adict) == 0:
                continue

            c1_adict_matched, c2_adict_matched = self._match_index(*adict['frame_id'])

            if len(c1_adict_matched) == 0:  # critical because there might be no matches (e.g SCT result comes slower), wo we need to wait more
                self.wait_list = active_list
                continue

            # add un-processed items to wait list
            self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
            self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])

            # using continuous indexes from 0 -> T-1 rather than discrete indexes
            T = len(c1_adict_matched)
            logger.debug(f'{self.name}:\t processing {T} frames')
            for k in adict:
                if len(adict[k][0]) > 0:
                    adict[k][0] = [adict[k][0][idx] for idx in c1_adict_matched]
                if len(adict[k][1]) > 0:
                    adict[k][1] = [adict[k][1][idx] for idx in c2_adict_matched]

            mid_time = time.time()
            pre_time = mid_time - start_time

            for t in range(T):

                # handle out-of-memory for exclusively offline
                self.trigger_pause()

                frame_img = adict['frame_img'][1][t]
                dets = adict['sct_output'][0][t]

                for scene_type, scenes in self.scenes.items():
                    for s in scenes:
                        frame_img = plot_roi(
                            frame_img,
                            s.roi,
                            self.config.get('VIS_ROI_THICKNESS'),
                            color=self.SCENE_COLORS[scene_type],
                        )

                # expecting dets to be in the format of [uid, tid, x1, y1, x2, y2, score, ...]
                with self.app.app_context():
                    texts = [User.query.filter_by(id=user_id).first().name if user_id != -1 else '' for user_id in np.int32(dets[:, 0]).tolist()]   # type: ignore
                frame_img = plot_box(frame_img, dets, self.config.get('VIS_SCT_BOX_THICKNESS'), texts=texts)
                # if detection_mode == 'pose':
                    # for kpt in dets[:, 10:]:
                    #     frame_img = plot_skeleton_kpts(frame_img, kpt.T, 3)

                locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'))
                frame_img = plot_loc(frame_img, np.concatenate([dets[:, :2], locs], axis=1), self.config.get('VIS_SCT_LOC_RADIUS'))

                if 'signin_user_id' in adict:
                    signin_user_id = adict['signin_user_id'][0][t]
                    if signin_user_id is not None:
                        pop_signin_count = 1
                        pop_signin_id = signin_user_id
                    if pop_signin_count > 0:
                        cv2.putText(frame_img, f'user<{pop_signin_id}> signed in', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), thickness=7)
                        pop_signin_count += 1
                    if pop_signin_count >= 60:
                        pop_signin_count = 0

                out_item = {
                    'frame_img': frame_img,                   # type: ignore
                    'frame_id': adict['frame_id'][1][t]
                }

                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item)

                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')

            start_time = mid_time


    def _match_index(self, a, b):
        i, j = 0, 0
        li, lj = [], []

        while i < len(a) and j < len(b):

            if a[i] == b[j]:
                li.append(i)
                lj.append(j)
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1

        return li, lj


    def _new_wait_list(self):
        return [[], []]


    def _check_video_queue(self):
        assert isinstance(self.video_queue, MyQueue), f'visualizing SCT requires video input queue, got {type(self.video_queue)}'


class MCMapPipeline(Pipeline):

    def __init__(
            self,
            app,
            db,
            config: ConfigPipeline,
            sct_queues: dict,
            sta_queues: dict,
            checkin_scene: Scene,
            checkin_cid: int,
            online_put_sleep: Union[int, float] = 0,
            name='MCMapPipeline'
    ) -> None:
        super().__init__(config, online_put_sleep, name)

        self.app = app
        self.db = db

        self.sct_queues = sct_queues
        self.sta_queues = sta_queues

        self.checkin_scene = checkin_scene
        self.checkin_cid = checkin_cid
        self._check_checkin_cid()

        self.wait_list = self._new_wait_list()

        logger.info(f'{self.name}:\t initialized')


    def _load_users(self):
        self.users = {}
        users = User.query.filter(User.role.in_(['intern', 'engineer'])).all()
        for user in users:
            self.users[user.id] = user
            user.load_workareas()
            user.load_next_workshift()


    def _start(self) -> None:

        with self.app.app_context():
            self._load_users()

            start_time = time.time()
            still_wait = [{k: True for k in q} for q in [self.sct_queues, self.sta_queues]]

            while not self.is_stopped():

                self.trigger_pause()

                for i, q in enumerate([self.sct_queues, self.sta_queues]):
                    for k, v in q.items():
                        if k not in self.wait_list[i]:
                            self.wait_list[i][k] = []

                        items = []
                        if self.config.get('RUNNING_MODE') == 'online':
                            if still_wait[i][k]:
                                item = v.get(block=True)
                                if item == '<EOS>':
                                    still_wait[i][k] = False
                                else:
                                    items = [item]
                        else:
                            while still_wait[i][k]:
                                item = v.get()
                                if item == '<EOS>':
                                    still_wait[i][k] = False
                                else:
                                    items.append(item)

                        self.wait_list[i][k].extend(items)

                if True not in [q for w in still_wait for q in w.values()] and self.config.get('RUNNING_MODE') == 'online':
                    for cid, oq in self.output_queues.items():
                        oq.put('<EOS>')
                    logger.info(f'{self.name}:\t reached <EOS> token')
                    for user in self.users.values():
                        user.update_detection('<EOS>', '<EOS>', '<EOS>')
                    self.wait_list = self._new_wait_list()  # release memory TODO very naive
                    break

                active_list = self.wait_list
                self.wait_list = self._new_wait_list()

                sta_midxs = self._match_sta_index(active_list[1])
                sct_midxs, sta_midxs = self._match_sct_index(active_list[0], active_list[1], sta_midxs)

                if len(list(sta_midxs.values())[0]) == 0:
                    self.wait_list = active_list
                    continue

                T = len(list(sta_midxs.values())[0])
                logger.debug(f'{self.name}:\t processing {T} frames')

                for wl, al, idxs in zip(self.wait_list, active_list, [sct_midxs, sta_midxs]):   # type: ignore
                    for k in al:
                        if k not in wl:
                            wl[k] = []
                        wl[k].extend(al[k][idxs[k][-1] + 1:])
                        al[k] = [al[k][idx] for idx in idxs[k]]

                mid_time = time.time()
                pre_time = mid_time - start_time

                for t in range(T):

                    # process sign in
                    signin_user_id = active_list[0][self.checkin_cid][t]['signin_user_id']
                    if signin_user_id is not None:
                        dets = active_list[0][self.checkin_cid][t]['sct_output']
                        locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'), (self.checkin_scene.width / 2, self.checkin_scene.height)) # type: ignore
                        in_roi_idxs = self.checkin_scene.is_in_roi(locs)
                        dets_in_roi = dets[in_roi_idxs]
                        if len(dets_in_roi) > 1:
                            raise ValueError(f'Found {len(dets_in_roi)} detections in checkin area')
                        elif len(dets_in_roi) == 1:
                            tid = int(dets_in_roi[0][1])
                            self.users[signin_user_id].update_hint(self.checkin_cid, tid)

                    sct_outputs = {(cid, int(d[1])): d
                                for cid, cv in active_list[0].items()
                                for d in cv[t]['sct_output']}
                    matches = {}
                    for (cid1, cid2), cv in active_list[1].items():
                        for tid1, tid2 in cv[t]['matches']:
                            matches[(cid1, tid1)] = (cid2, tid2)
                            matches[(cid2, tid2)] = (cid1, tid1)

                    out_visualize = {cam_id: [] for cam_id in active_list[0]}

                    for user in self.users.values():
                        hint_cid, hint_tid = user.get_hint()
                        match = matches.get((hint_cid, hint_tid), (None, None))
                        user.update_hint(*match)
                        det = sct_outputs.get((hint_cid, hint_tid), None)
                        if det is None:
                            loc = None
                            dtime = min([cv[t]['frame_time'] for cv in active_list[0].values()])
                            cid = None
                        else:
                            loc = calc_loc(np.expand_dims(det, axis=0), self.config.get('LOC_INFER_MODE'))[0]
                            dtime = active_list[0][hint_cid][t]['frame_time']
                            cid = hint_cid

                            del sct_outputs[(hint_cid, hint_tid)]

                            det[0] = user.id    # replace trivial frame_id in sct_output with user id
                            out_visualize[cid].append(det)
                        user.update_detection(cid, dtime, loc)

                    for (cid, tid), d in sct_outputs.items():
                        d[0] = -1
                        out_visualize[cid].append(d)

                    for cid in out_visualize:
                        out_visualize[cid] = np.array(out_visualize[cid])   # type: ignore
                        if len(out_visualize[cid]) == 0:
                            out_visualize[cid] =  np.empty((0, 10))          # type: ignore , whatever, as long as dim2 >= 2


                    end_time = time.time()
                    sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                        else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time))
                    time.sleep(sleep)

                    # expecting key of the output queue is camera ID
                    self.lock.acquire()
                    for cid, oq in self.output_queues.items():
                        oq.put(
                            {
                                'frame_id': active_list[0][cid][t]['frame_id'],
                                'sct_output': out_visualize[cid],
                                'sct_detection_mode': active_list[0][cid][t]['sct_detection_mode'],
                                'signin_user_id': signin_user_id if cid == self.checkin_cid else None
                            },
                            block=True
                        )
                    self.lock.release()
                    mid_time = end_time
                    logger.debug(f'{self.name}:\t slept {sleep}')

                start_time = mid_time

                if self.config.get('RUNNING_MODE') == 'offline':
                    for cid, oq in self.output_queues.items():
                        oq.put('<EOS>')
                    break


    def _check_checkin_cid(self):
        assert self.checkin_cid in self.sct_queues


    def _match_sta_index(self, sta_dict):
        ks, vs = list(sta_dict.keys()), list(sta_dict.values())
        cids = {}
        for i, k in enumerate(ks):
            for j, cid in enumerate(k):
                if cid not in cids:
                    cids[cid] = []
                cids[cid].append((i, j))

        its = [0 for _ in range(len(ks))]
        idxs = [[] for _ in range(len(ks))]

        while True:
            cond = True
            for i, it in enumerate(its):
                cond = cond and (it < len(vs[i]))
            if not cond:
                break

            good = True

            for cid in cids:
                br1 = False
                fid_p = None
                for i, j in cids[cid]:
                    fid = vs[i][its[i]][f'frame_id_{j + 1}']
                    if fid_p is None:
                        fid_p = fid
                    else:
                        if fid != fid_p:
                            if fid < fid_p:
                                its[i] += 1
                            else:
                                for ib in range(i):
                                    its[ib] += 1
                            br1 = True
                            break
                if br1:
                    good = False
                    break

            if good:
                for i in range(len(ks)):
                    idxs[i].append(its[i])
                    its[i] += 1

        return {k: v for k, v in zip(ks, idxs)} # type: ignore


    def _match_sct_index(self, sct_dict, sta_dict, sta_idxs):
        sct_idxs = {k: [] for k in sct_dict.keys()}
        new_sta_idxs = {k: [] for k in sta_idxs.keys()}
        prev_j = {k: 0 for k in sct_dict.keys()}

        for idxs in zip(*sta_idxs.values()):
            temp_sct_idx = {}
            temp_sta_idx = {}
            for i, sta_k in enumerate(sta_idxs.keys()):
                temp_sta_idx[sta_k] = idxs[i]
                for k, cid in enumerate(sta_k):
                    if cid in temp_sct_idx:
                        continue

                    temp_sct_idx[cid] = None
                    while True:
                        j = prev_j[cid]
                        if j >= len(sct_dict[cid]):
                            break
                        sta_fid = sta_dict[sta_k][idxs[i]][f'frame_id_{k + 1}']
                        sct_fid = sct_dict[cid][j]['frame_id']
                        prev_j[cid] += 1
                        if sta_fid == sct_fid:
                            temp_sct_idx[cid] = j
                            break


            if None in temp_sct_idx.values():
                continue

            for k, v in temp_sct_idx.items():
                sct_idxs[k].append(v)
            for k, v in temp_sta_idx.items():
                new_sta_idxs[k].append(v)

        return sct_idxs, new_sta_idxs       # type: ignore


    def _new_wait_list(self):
        return [{}, {}] # sct and sta


class Monitor:

    def __init__(self, name='Monitor'):
        self.name = name


    def init_app(self, app, db, fake_clock):
        self.app = app
        self.db = db
        self.config = app.config['PIPELINE']
        self.fake_clock = fake_clock

        logger.info(f'{self.name}:\t initialized with app and db')


    def register_camera(
            self,
            cam_id: int,
            address: str,
            meta_path: str,
            txt_path: Union[str, None] = None,
    ) -> None:

        logger.info(f'{self.name}:\t registering camera {cam_id}')

        if not hasattr(self, 'pl_cameras'):
            self.pl_cameras = {}

        if not hasattr(self, 'pl_scts'):
            self.pl_scts = {}

        # create camera pipeline
        meta = yaml.safe_load(open(meta_path, 'r'))
        pl_camera = CameraPipeline(
            config=self.config,
            source=address,
            meta=meta,
            online_put_sleep=1.0 / meta['fps'],
            name=f'PL Camera-<cam_id={cam_id}>',
        )

        # create tracker
        with self.app.app_context():
            use_real_tracker = self.app.config['USE_REAL_TRACKER']
        tracker = Tracker(
            detection_mode=self.config.get('DETECTION_MODE'),
            tracking_mode=self.config.get('TRACKING_MODE'),
            detection_weight=self.config.get('DETECTION_WEIGHT'),
            detection_conf_thres=self.config.get('DETECTION_CONF_THRES'),
            detection_iou_thres=self.config.get('DETECTION_IOU_THRES'),
            detection_tsize=self.config.get('DETECTION_TSIZE'),
            tracking_config=self.config.get('TRACKING_CONFIG'),
            device=self.config.get('DEVICE'),
            txt_path=txt_path,
            use_real_tracker=use_real_tracker,
        )

        # commit output queue to camera pipeline
        queue = MyQueue(name=f'IQ-SCT-<cam_id={cam_id}>')
        pl_camera.add_output_queue(queue, queue.name)

        # create sct pipeline
        pl_sct = SCTPipeline(
            config=self.config,
            tracker=tracker,
            input_queue=queue,
            # online_put_sleep=pl_camera.online_put_sleep * 0.1,
            name=f'PL SCT-<cam_id={cam_id}>',
        )

        self.pl_cameras[cam_id] = pl_camera
        self.pl_scts[cam_id] = pl_sct


    def register_overlap(
            self,
            cam_id_primary: int,
            cam_id_secondary: int,
            scene_primary: Scene,
            scene_secondary: Scene,
            homo: np.ndarray,
    ) -> None:

        logger.info(f'{self.name}:\t registering overlapping area from sec_id={cam_id_secondary} to pri_id={cam_id_primary}')

        if not hasattr(self, 'pl_stas'):
            self.pl_stas = {}

        if not hasattr(self, 'pl_syncs'):
            self.pl_syncs = {}

        # create sync pipeline
        oq_cam_secondary = MyQueue(name=f'IQ-Sync-<*sec={cam_id_secondary}, pri={cam_id_primary}>')
        oq_cam_primary = MyQueue(name=f'IQ-Sync-<sec={cam_id_secondary}, *pri={cam_id_primary}>')
        self.pl_cameras[cam_id_secondary].add_output_queue(oq_cam_secondary, oq_cam_secondary.name)
        self.pl_cameras[cam_id_primary].add_output_queue(oq_cam_primary, oq_cam_primary.name)

        pl_sync = SyncPipeline(
            config=self.config,
            input_queues=[oq_cam_secondary, oq_cam_primary],
            # online_put_sleep=min(self.pl_cameras[cam_id_secondary].online_put_sleep, self.pl_cameras[cam_id_primary].online_put_sleep) * 0.1,
            name=f'PL Sync-<sec={cam_id_secondary}, pri={cam_id_primary}>',
        )

        # commit output queues to sct and sync
        oq_sct_secondary = MyQueue(name=f'IQ-STA_SCT-<*sec={cam_id_secondary}, pri={cam_id_primary}>')
        oq_sct_primary = MyQueue(name=f'IQ-STA_SCT-<sec={cam_id_secondary}, *pri={cam_id_primary}>')
        oq_sync = MyQueue(name=f'IQ-STA_SYNC-<sec={cam_id_secondary}, pri={cam_id_primary}>')
        self.pl_scts[cam_id_secondary].add_output_queue(oq_sct_secondary, oq_sct_secondary.name)
        self.pl_scts[cam_id_primary].add_output_queue(oq_sct_primary, oq_sct_primary.name)
        pl_sync.add_output_queue(oq_sync, oq_sync.name)

        # create sta pipeline
        pl_sta = STAPipeline(
            config=self.config,
            scenes=[scene_secondary, scene_primary],
            homo=homo,
            sct_queues=[oq_sct_secondary, oq_sct_primary],
            sync_queue=oq_sync,
            # online_put_sleep=pl_sync.online_put_sleep,
            name=f'PL STA-<sec={cam_id_secondary}, pri={cam_id_primary}>'
        )

        self.pl_syncs[(cam_id_secondary, cam_id_primary)] = pl_sync
        self.pl_stas[(cam_id_secondary, cam_id_primary)] = pl_sta


    def register_checkin(
            self,
            cam_id: int,
            scene: Scene,
    ) -> None:
        assert hasattr(self, 'pl_scts')
        assert hasattr(self, 'pl_cameras')
        assert cam_id in self.pl_scts

        self.checkin_cid = cam_id

        self.fake_clock.set_start_time(self.pl_cameras[cam_id].record_time)

        # commit output queues to sct and sta
        oq_scts = {}
        for cid, pl in self.pl_scts.items():
            queue = MyQueue(name=f'IQ-MCMap_SCT-<cam_id={cid}>')
            pl.add_output_queue(queue, queue.name)
            oq_scts[cid] = queue

        oq_stas = {}
        for (cid_sec, cid_pri), pl in self.pl_stas.items():
            queue = MyQueue(name=f'IQ-MCMap_STA-<sec={cid_sec}, pri={cid_pri}>')
            pl.add_output_queue(queue, queue.name)
            oq_stas[(cid_sec, cid_pri)] = queue

        # create MCMap pipeline
        self.pl_mcmap = MCMapPipeline(
            app=self.app,
            db=self.db,
            config=self.config,
            sct_queues=oq_scts,
            sta_queues=oq_stas,
            checkin_scene=scene,
            checkin_cid=cam_id,
            # online_put_sleep=min([pl.online_put_sleep for pl in self.pl_cameras.values()]) * 0.1,
        )


    def start(self):

        self.fake_clock.start()

        # start camera pipelines
        for pl in self.pl_cameras.values():
            pl.start()

        if self.config.get('RUNNING_MODE') == 'offline':
            for pl in self.pl_cameras.values():
                pl.join()

        # start SCT pipelines
        for pl in self.pl_scts.values():
            pl.start()

        # start Sync pipelines
        for pl in self.pl_syncs.values():
            pl.start()

        if self.config.get('RUNNING_MODE') == 'offline':
            for pl in self.pl_scts.values():
                pl.join()

            for pl in self.pl_syncs.values():
                pl.join()

        # start STA pipelines
        for pl in self.pl_stas.values():
            pl.start()

        if self.config.get('RUNNING_MODE') == 'offline':
            for pl in self.pl_stas.values():
                pl.join()

        # start MCMap pipeline
        self.pl_mcmap.start()


    def register_display(self, cam_id: int, key: str):
        """Register a display for a camera.

        Args:
            cam_id (int): Camera ID.
            key (str): Key for the display. The client can access to the display queue with this key.
        """

        lock = Lock()

        # create visualize pipeline if not exists
        if not hasattr(self, 'pl_visualizes'):
            self.pl_visualizes = {}

        lock.acquire()
        if cam_id not in self.pl_visualizes:
            pl_visualize = self._create_visualize(cam_id)
            pl_visualize.start()
        else:
            pl_visualize = self.pl_visualizes[cam_id]
        lock.release()

        # commit output queue to visualize
        if not hasattr(self, 'display_queues'):
            self.display_queues = {}

        lock.acquire()
        if cam_id not in self.display_queues:
            self.display_queues[cam_id] = {}

        if key not in self.display_queues[cam_id]:
            oq_visualize = MyQueue(name=key)
            pl_visualize.add_output_queue(oq_visualize, oq_visualize.name)
            self.display_queues[cam_id][key] = oq_visualize
        lock.release()


    def _create_visualize(self, cam_id: int) -> VisualizePipeline:

        while not hasattr(self, 'pl_cameras') or cam_id not in self.pl_cameras or not hasattr(self, 'pl_mcmap'):
            logger.info(f'{self.name}:\t waiting for camera {cam_id} to be registered...')
            time.sleep(1)

        oq_video = MyQueue(name=f'IQ-Visualize_Video-<cam_id={cam_id}>')
        oq_annot = MyQueue(name=cam_id)     # must be exactly cam_id
        self.pl_cameras[cam_id].add_output_queue(oq_video, oq_video.name)
        self.pl_mcmap.add_output_queue(oq_annot, oq_annot.name)

        pl_visualize = VisualizePipeline(
            app=self.app,
            db=self.db,
            config=self.config,
            cam_id=cam_id,
            annot_queue=oq_annot,
            video_queue=oq_video,
            online_put_sleep=self.pl_cameras[cam_id].online_put_sleep,
            name=f'PL Visualize-<cam_id={cam_id}>'
        )
        self.pl_visualizes[cam_id] = pl_visualize

        return pl_visualize


    def get_display_queue(self, cam_id: int, key: str):
        """Return display queue of a camera to a client session."""
        if not hasattr(self, 'display_queues'):
            return None
        elif cam_id not in self.display_queues:
            return None
        else:
            return self.display_queues[cam_id].get(key, None)


    def withdraw_display(self, cam_id: int, key: str):

        if not hasattr(self, 'display_queues'):
            return

        if cam_id not in self.pl_visualizes:
            return

        if cam_id not in self.display_queues:
            return

        if key not in self.display_queues[cam_id]:
            return

        # remove output queue from visualize
        pl_visualize = self.pl_visualizes[cam_id]
        display_queue = pl_visualize.remove_output_queue(key)
        self._release_queue(display_queue)

        lock = Lock()
        lock.acquire()
        del self.display_queues[cam_id][key]
        del display_queue

        # stop visualize if no display queue
        if len(self.display_queues[cam_id]) == 0:

            logger.info(f'Destructing VisualizePipeline for camera {cam_id}...')

            oq_video = pl_visualize.video_queue
            oq_annot = pl_visualize.annot_queue
            del self.pl_visualizes[cam_id]
            del pl_visualize
            self.pl_cameras[cam_id].remove_output_queue(oq_video.name)
            self.pl_mcmap.remove_output_queue(oq_annot.name)
            del oq_video
            del oq_annot
            del self.display_queues[cam_id]

        lock.release()


    def signal_signin(self, user_id):
        self.pl_cameras[self.checkin_cid].signal_signin(user_id)


    def _release_queue(self, queue):
        while not queue.empty():
            queue.get()