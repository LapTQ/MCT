from typing import Union, List
from datetime import datetime
import os
import time
import cv2
import logging
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

sys.path.append(str(HERE))

from mct.utils.general import calc_loc, hungarian, map_mono                   # type: ignore
from mct.sta.engines import FilterBase, IQRFilter, GMMFilter, MyQueue, Scene, Tracker
from .base import ConfigPipeline, Pipeline

logger = logging.getLogger(__name__)

# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s\t|%(funcName)30s\t|%(lineno)d\t|%(levelname)s\t|%(message)s')

# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(formatter)
# logger.addHandler(handler)

#logging.FileHandler("~/Downloads/log.txt"),


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(funcName)20s |%(lineno)d\t|%(levelname)8s |%(message)s',
    handlers=[
        #logging.FileHandler("~/Downloads/log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)


class CameraPipeline(Pipeline):

    def __init__(
            self,
            config: ConfigPipeline, 
            source: Union[int, str],
            meta: Union[dict, None] = None, 
            online_put_sleep: Union[int, float] = 0,
            ret_img: bool = True,
            name='CameraPipeline'
    ) -> None:
        super().__init__(config, online_put_sleep, name)
        
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        self.meta = meta
        self._check_meta()

        self.ret_img = ret_img

        self.signin_user_id = None
        
        logger.info(f'{self.name}:\t initilized')

    
    def _start(self) -> None:

        assert not (self.config.get('RUNNING_MODE') == 'offline' and self.ret_img)

        if self.meta is None:
            frame_id = 0
        else:   # if reading from video on disk
            frame_id = self.meta['start_frame_id'] - 1

        start_time = time.time()

        while not self.is_stopped():

            self.trigger_pause()

            if not self.cap.isOpened():
                logger.info(f'{self.name}:\t problem connecting to {self.source}')
                self.stop()
                break

            if self.ret_img:
                ret, frame = self.cap.read()
            else:
                # running offline without visualizing
                ret, frame = True, None
                if frame_id == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    ret = False
            
            frame_id += 1

            if self.meta is None:
                frame_time = datetime.now().timestamp()
            else:   # if reading from video on disk
                frame_time = self.record_time + (frame_id - self.meta['start_frame_id']) / self.fps

            ########### MOCK TEST ###########
            if '<cam_id=1>' in self.name:
                if frame_id == 217:
                    self.signal_signin(4)
                elif frame_id == 614:
                    self.signal_signin(3)
                elif frame_id == 889:
                    self.signal_signin(5)
            #################################

            signin_user_id = self._observe_signin()
            if signin_user_id is not None:
                logger.info(f'{self.name}: get sign-in signal with user_id={signin_user_id} at {datetime.fromtimestamp(frame_time)}')

            # send None as the end-of-stream signal
            out_item = '<EOS>' if not ret else {
                    'frame_img': frame,
                    'frame_id': frame_id,
                    'frame_time': frame_time,
                    'signin_user_id': signin_user_id
                }
            
            end_time = time.time()
            # if reading from video on disk, then sleep according to fps to sync time.
            sleep = 0 if self.meta is None or self.config.get('RUNNING_MODE') == 'offline' \
                else max(0, self.online_put_sleep - (end_time - start_time))
            time.sleep(sleep)
            
            self._put_to_output_queues(out_item)

            start_time = end_time
            logger.debug(f"{self.name}:\t slept {sleep}")

            if out_item == '<EOS>':
                logger.info(f'{self.name}:\t disconnected from {self.source}')
                break

        self.cap.release()

    
    def _check_meta(self):
        if self.config.get('CAMERA_FRAME_WIDTH') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('CAMERA_FRAME_WIDTH'))
        if self.config.get('CAMERA_FRAME_HEIGHT') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('CAMERA_FRAME_HEIGHT'))
        if self.config.get('CAMERA_FPS') is not None:
            assert self.meta is None, 'camera_fps must not be set when capturing videos from disk'
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('CAMERA_FPS'))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
        if self.meta is not None:
            self.record_time = datetime.strptime(
                self.meta['start_time'], 
                '%Y-%m-%d_%H-%M-%S-%f'
            ).timestamp()

    
    def _observe_signin(self):
        signin_user_id, self.signin_user_id = self.signin_user_id, None
        return signin_user_id
        

    def signal_signin(self, sid):
        self.signin_user_id = sid
        

class SCTPipeline(Pipeline):

    def __init__(
            self, 
            config: ConfigPipeline, 
            tracker: Tracker,
            input_queue: MyQueue,
            online_put_sleep: Union[int, float] = 0,
            name='SCTPipeline'
    ) -> None:
        super().__init__(config, online_put_sleep, name)

        self.tracker = tracker
        
        self.input_queue = input_queue

        logger.info(f'{self.name}:\t initialized')

    
    def _start(self) -> None:

        start_time = time.time()
        still_wait = True
        
        while not self.is_stopped():

            self.trigger_pause()

            items = []
            if self.config.get('RUNNING_MODE') == 'online':
                if still_wait:
                    item = self.input_queue.get(block=True)
                    if item == '<EOS>':
                        still_wait = False
                    else:
                        items = [item]
            else:
                while still_wait:
                    item = self.input_queue.get()
                    if item == '<EOS>':
                        still_wait = False
                    else:
                        items.append(item)

            if not still_wait and self.config.get('RUNNING_MODE') == 'online':
                self._put_to_output_queues('<EOS>')
                logger.info(f'{self.name}:\t reached <EOS> token')
                break

            T = len(items)
            logger.debug(f'{self.name}:\t processing {T} frames')

            mid_time = time.time()
            pre_time = mid_time - start_time
            
            for item in items:

                if item is None:
                    out_item = None
                else:
                    dets = self.tracker.infer(item['frame_img'], item['frame_id'])
                    
                    out_item = {
                        'frame_id': item['frame_id'],
                        'frame_time': item['frame_time'],
                        'sct_output': dets,
                        'sct_detection_mode': self.tracker.detection_mode,
                        'sct_tracking_mode': self.tracker.tracking_mode,
                        'signin_user_id': item['signin_user_id']
                    }

                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item)

                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')

            start_time = mid_time

            if self.config.get('RUNNING_MODE') == 'offline':
                self._put_to_output_queues('<EOS>')
                break


class SyncPipeline(Pipeline):

    def __init__(
            self, 
            config: ConfigPipeline, 
            input_queues: List[MyQueue],
            online_put_sleep: Union[int, float] = 0,
            name='SyncPipeline'
    ) -> None:
        super().__init__(config, online_put_sleep, name)

        self.input_queues = input_queues
        self._check_input_queues()

        self.wait_list = self._new_wait_list()

        logger.debug(f'{self.name}:\t initialized')

    
    def _start(self) -> None:

        start_time = time.time()
        still_wait = [True, True]

        while not self.is_stopped():

            self.trigger_pause()
            
            # load both the coming and waiting list
            for i, (iq, wl) in enumerate(zip(self.input_queues, self.wait_list)):

                items = []
                if self.config.get('RUNNING_MODE') == 'online':
                    if still_wait[i]:
                        item = iq.get(block=True)
                        if item == '<EOS>':
                            still_wait[i] = False
                        else:
                            items = [item]
                else:
                    while still_wait[i]:
                        item = iq.get()
                        if item == '<EOS>':
                            still_wait[i] = False
                        else:
                            items.append(item)

                wl.extend(items)
            
            if True not in still_wait and self.config.get('RUNNING_MODE') == 'online':
                self._put_to_output_queues('<EOS>')
                logger.info(f'{self.name}:\t reached <EOS> token')
                self.wait_list = self._new_wait_list()  # release memory TODO very naive
                break
            
            # do not map if the number of pairs are so small
            if min(len(self.wait_list[0]), len(self.wait_list[1])) < self.config.get('MIN_TIME_CORRESPONDENCES'):
                logger.debug(f'{self.name}:\t wait list not enough, waiting...')
                continue
            
            active_list = self.wait_list
            self.wait_list = self._new_wait_list()

            # reorganize items
            adict = {
                'frame_id': [],         # [[c1_f1, c1_f2,...], [c2_f1, c2_f2, ...]]
                'frame_time': [],
            }
            for citems in active_list:
                for k in adict:
                    adict[k].append([])
                for k in adict:
                    for item in citems:
                        adict[k][-1].append(item[k])
            # each array in frame_time, frame_id should already been in increasing order of time            

            # match timestamps
            c1_adict_matched, c2_adict_matched = map_mono(*adict['frame_time'], diff_thresh=self.config.get('TIME_DIFF_THRESH'))
            T = len(c1_adict_matched)

            # add un-processed items to wait list
            if T == 0:
                continue
                
            self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
            self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])
            
            logger.debug(f'{self.name}:\t processing {T} frames')

            mid_time = time.time()
            pre_time = mid_time - start_time
            
            for i, j in zip(c1_adict_matched, c2_adict_matched):

                logger.debug(f"{self.name}:\t sync frame_id={adict['frame_id'][0][i]} ({datetime.fromtimestamp(adict['frame_time'][0][i]).strftime('%M-%S-%f')}) with frame_id={adict['frame_id'][1][j]} ({datetime.fromtimestamp(adict['frame_time'][1][j]).strftime('%M-%S-%f')})")

                out_item = {
                    'frame_id_match': (adict['frame_id'][0][i], adict['frame_id'][1][j])
                }
                
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item)

                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')
            
            start_time = mid_time

            if self.config.get('RUNNING_MODE') == 'offline':
                self._put_to_output_queues('<EOS>')
                break
    

    def _check_input_queues(self):
        assert len(self.input_queues) == 2, f'currently support only bipartite mapping, got {len(self.input_queues)}'


    def _new_wait_list(self):
        return [[], []]


class STAPipeline(Pipeline):

    def __init__(
            self, 
            config: ConfigPipeline, 
            scenes: List[Scene],
            homo: Union[np.ndarray, None],
            sct_queues: List[MyQueue],
            sync_queue: MyQueue,
            online_put_sleep: Union[int, float] = 0,
            name='STAPipeline'
    ) -> None:
        super().__init__(config, online_put_sleep, name)
        """
        homo: tranform the view of scene[0] to scene[1]
        """

        self.sct_queues = sct_queues
        self._check_sct_queues()

        self.sync_queue = sync_queue

        self.scenes = scenes
        self._check_scenes()

        self.homo = homo
        self.lim = self.config.get('LOC_INFER_MODE')
        self.filter = self._create_fp_filter()
        self.ws = self.config.get('DIST_WINDOW_SIZE')
        self.wb = self.config.get('DIST_WINDOW_BOUNDARY')

        self.history = []       # [(frame_id_1, frame_id_2, ({id_1: loc_1, ...}, {id_2: loc_2, ...}), [(mid_1, mid_2), ...]), ...]
        self.distances = []     # record distances for FP elimination
        self.wait_list = self._new_wait_list()

        logger.info(f'{self.name}:\t initialized')


    def _start(self) -> None:

        start_time = time.time()
        still_wait = [True, True, True]
        
        while not self.is_stopped():

            self.trigger_pause()
            
            for i, (wl, iq) in enumerate(zip(self.wait_list, self.sct_queues + [self.sync_queue])):
                
                items = []
                if self.config.get('RUNNING_MODE') == 'online':
                    if still_wait[i]:
                        item = iq.get(block=True)
                        if item == '<EOS>':
                            still_wait[i] = False
                        else:
                            items = [item]
                else:
                    while still_wait[i]:
                        item = iq.get()
                        if item == '<EOS>':
                            still_wait[i] = False
                        else:
                            items.append(item)
                
                wl.extend(items)

            if True not in still_wait and self.config.get('RUNNING_MODE') == 'online':
                self._put_to_output_queues('<EOS>')
                logger.info(f'{self.name}:\t reached <EOS> token')
                self.wait_list = self._new_wait_list()  # release memory TODO very naive
                break
            
            active_list = self.wait_list
            self.wait_list = self._new_wait_list()

            # reorganize items
            adict = {
                'frame_id': [],             # [[c1_f1, c1_f2,...], [c2_f1, c2_f2, ...]]
                'sct_output': [],
            }
            for citems in active_list[:2]:   # process items of sct_queue
                for k in adict:
                    adict[k].append([])
                for k in adict:
                    for item in citems:
                        adict[k][-1].append(item[k])
            adict['frame_id_match'] = []
            for item in active_list[2]:     # process items of sync_queue
                adict['frame_id_match'].append(item['frame_id_match'])
            # each array in frame_id and frame_id_match should already been in increasing order of time
            
            c1_adict_matched, c2_adict_matched, sync_adict_matched = self._match_index(*adict['frame_id'], adict['frame_id_match'])  # type: ignore
            
            if len(c1_adict_matched) <= self.wb:  # critical because there might be no matches (e.g Sync result comes slower), so we need to wait more
                self.wait_list = active_list
                continue

            # add un-processed items to wait list, postponse the last BOUNDARY frames for the next iteration
            self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1 - self.wb:])
            self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1 - self.wb:])
            self.wait_list[2].extend(active_list[2][sync_adict_matched[-1] + 1 - self.wb:])
        
            # using continuous indexes from 0 -> T-1 rather than discrete indexes
            T = len(c1_adict_matched)
            logger.debug(f'{self.name}:\t processing {T} pairs of frames')
            del adict['frame_id_match']
            for k in adict:
                adict[k][0] = [adict[k][0][idx] for idx in c1_adict_matched]
                adict[k][1] = [adict[k][1][idx] for idx in c2_adict_matched]
            
            logger.debug(f"{self.name}:\t infer location from detection with LOC_INFER_MODE={self.lim}")
            adict['in_roi'] = [[], []]

            for t in range(T):
                self.history.append(
                    (
                        adict['frame_id'][0][t],
                        adict['frame_id'][1][t],
                        ({}, {}),   # locs
                        ({}, {}),   # loc in ROI
                        []
                    )
                )

                for c in range(2):

                    scene = self.scenes[c]
                    dets = adict['sct_output'][c][t]
                    locs = calc_loc(dets, self.lim, (scene.width / 2, scene.height))   # type: ignore

                    # filter in ROI
                    if self.config.get('STA_PSEUDO_SAME_CAMERA') and self.config.get('STA_PSEUDO_IN_ROI_ACCORD_CAM1'):
                        # handle for pseudo true mapping between gt box vs tracker pose, because differen detection mode might give different in-roi result
                        if c == 0:
                            dets_1 = adict['sct_output'][1][t]
                            in_roi_idxs_1 = self.scenes[1].is_in_roi(calc_loc(dets_1, self.config.get('STA_PSEUDO_IN_ROI_LOC_INFER_MODE_CAM1'), (self.scenes[1].width / 2, self.scenes[1].height)))   # type: ignore
                            dets_1 = dets_1[in_roi_idxs_1]
                            # assumming dets = [[_, _, x1(cx,...), y1(cy,...), w(x2,...), h(y2,...), ...], ...]
                            in_roi_idxs = np.where(np.all(abs(np.subtract(dets[:, 2:6].reshape(-1, 1, 4), dets_1[:, 2:6].reshape(1, -1, 4))) < 1, axis=-1))[0]
                        else:
                            in_roi_idxs = scene.is_in_roi(calc_loc(dets, self.config.get('STA_PSEUDO_IN_ROI_LOC_INFER_MODE_CAM1'), (scene.width / 2, scene.height)))        # type: ignore
                    else:
                        in_roi_idxs = scene.is_in_roi(locs)
                                        
                    dets_in_roi = dets[in_roi_idxs]
                    locs_in_roi = locs[in_roi_idxs]
                    if (c == 0 and (not self.config.get('STA_PSEUDO_SAME_CAMERA') or self.homo is not None)) \
                        or (c == 1 and self.config.get('STA_PSEUDO_SAME_CAMERA') and self.homo is not None) :  # if cam 1, then transform to view of cam 2
                        locs = cv2.perspectiveTransform(locs.reshape(-1, 1, 2), self.homo)
                        locs = locs.reshape(-1, 2) if locs is not None else np.empty((0, 2))
                        
                        locs_in_roi = cv2.perspectiveTransform(locs_in_roi.reshape(-1, 1, 2), self.homo)
                        locs_in_roi = locs_in_roi.reshape(-1, 2) if locs_in_roi  is not None else np.empty((0, 2))
                    adict['in_roi'][c].append(dets_in_roi)
                    logger.debug(f'{self.name}:\t camera {c + 1} frame {adict["frame_id"][c][t]} found {len(dets_in_roi)}/{len(dets)} objects in ROI')
                    
                    # store location history, both inside-only and inide-outside
                    assert dets_in_roi.shape[1] in (10, 61, 9), 'expect track_id is of index 1' # ATTENTION: 9 is for output of CVAT only @@
                    self.history[-1][2][c].update({id: loc for id, loc in zip(np.int32(dets[:, 1]), locs.tolist())})                     # type: ignore
                    self.history[-1][3][c].update({id: loc for id, loc in zip(np.int32(dets_in_roi[:, 1]), locs_in_roi.tolist())})       # type: ignore
            
            ub = 2e9
            logger.debug(f"{self.name}:\t calculating cost with WINDOW_SIZE={self.ws}, WINDOW_BOUNDARY={self.wb}")
            for iter_n in range(2 if self.filter and self.config.get('FP_REMAP') else 1):
                matches = []
                for t in range(T - self.wb):    # postponse the last BOUNDARY frames for the next iteration
                    
                    h1_ids = adict['in_roi'][0][t][:, 1]
                    h2_ids = adict['in_roi'][1][t][:, 1]

                    cost = np.empty((len(h1_ids), len(h2_ids)), dtype='float32')
                    gate = np.ones((len(h1_ids), len(h2_ids)), dtype='int32')
                    for i1, c1_id in enumerate(h1_ids):
                        for i2, c2_id in enumerate(h2_ids):
                            c1_locs, c2_locs = self._retrieve_window_locations(
                                c1_id, 
                                c2_id,
                                t - T
                            )

                            cost[i1, i2] = np.mean(
                                np.sqrt(np.sum(
                                    np.square(c1_locs - c2_locs),
                                    axis=1,
                                    keepdims=False
                                ))
                            )

                            if cost[i1, i2] > ub:
                                gate[i1, i2] = 0                    
                    
                    mi1s, mi2s = hungarian(cost, gate=gate)
                    if len(mi1s) > 0:
                        logger.debug(f'{self.name}:\t pair frame_id ({adict["frame_id"][0][t]}, {adict["frame_id"][1][t]}) found {len(mi1s)} matches')
                    for i1, i2 in zip(mi1s, mi2s):  # type: ignore
                        h1 = h1_ids[i1]
                        h2 = h2_ids[i2]
                        dist = cost[i1, i2]
                        matches.append([t, h1, h2, dist])
                
                matches = np.array(matches).reshape(-1, 4)
                
                # FP filter
                d = self.distances + matches[:, 3].tolist()
                if self.filter and len(d) >= self.config.get('MIN_SAMPLE_SIZE_TO_FP_FILTER'):
                    ub = self.filter(np.array(d).reshape(-1, 1))             # type: ignore
                    matches = matches[matches[:, 3] <= ub]                  
                    
                    logger.debug(f"{self.name}:\t applied FP_FLITER={self.config.get('FP_FILTER')}")
            
            self.distances.extend(matches[:, 3].tolist())       # type: ignore
            
            for m in matches:               # type: ignore
                t = int(m[0].item())
                self.history[t - T][4].append(np.int32(m[1:3]).tolist())    # correct only because t in [0, T-1]

            self.history = self.history[:len(self.history) - self.wb]   # postponse the last BOUNDARY frames for the next iteration
            
            mid_time = time.time()
            pre_time = mid_time - start_time
            
            for his in self.history[-max(T - self.wb, 0):]:     # postponse the last BOUNDARY frames for the next iteration
                out_item = {
                    'frame_id_1': his[0],
                    'frame_id_2': his[1],
                    'locs': his[2],
                    'locs_in_roi': his[3],
                    'matches': his[4]
                }

                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / (T - self.wb) + end_time - mid_time))
                time.sleep(sleep)    
                
                self._put_to_output_queues(out_item)

                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')
            
            start_time = mid_time
            
            if self.config.get('RUNNING_MODE') == 'offline':
                self._put_to_output_queues('<EOS>')
                break


    def _match_index(self, a, b, c):
        i, j, k = 0, 0, 0
        li, lj, lk = [], [], []

        while i < len(a) and j < len(b) and k < len(c):
            
            if a[i] == c[k][0] and b[j] == c[k][1]:
                li.append(i)
                lj.append(j)
                lk.append(k)
                i += 1
                j += 1
                k += 1
            else:
                if a[i] > c[k][0] or b[j] > c[k][1]:
                    k += 1

                else:
                    if a[i] < c[k][0]:
                        i += 1
                    if b[j] < c[k][1]:
                        j += 1
        
        return li, lj, lk


    def _create_fp_filter(self) -> Union[FilterBase, None]:

        filter_type = self.config.get('FP_FILTER')

        if not filter_type:
            return None
        
        if filter_type == 'iqr':
            return IQRFilter(
                self.config.get('FP_IQR_LOWER'),
                self.config.get('FP_IQR_UPPER')
            )
        
        if filter_type == 'gmm':
            return GMMFilter(
                self.config.get('FP_GMM_N_COMPONENTS'),
                self.config.get('FP_GMM_STD_COEF')
            )
        

    def _retrieve_window_locations(self, c1_id, c2_id, t) -> tuple:
        """Return a tuple of 2 np arrays for locations
        
        Arguments:
            t: index in the self.history (e.g -1 or -5)
        """
        assert self.ws % 2 == 1, 'window size must be an odd number.'
        
        locs = [self.history[t][3][0][c1_id]], [self.history[t][3][1][c2_id]]
        
        ns = [0, 0]     # counter for left, right
        slices = [
            [self.history[j][3] for j in range(t - 1, t - 1 - self.wb, -1)], 
            [self.history[j][3] for j in range(t + 1, t + self.wb + 1)]
        ]

        for i, slice in enumerate(slices):
            for j in range(self.wb):

                if j >= len(slice) or ns[i] >= (self.ws - 1) // 2:
                    break

                if c1_id in slice[j][0] and c2_id in slice[j][1]:
                    locs[0].append(slice[j][0][c1_id])
                    locs[1].append(slice[j][1][c2_id])
                    ns[i] += 1
        logger.debug(f'{self.name}:\t window size for ID pairs {c1_id, c2_id}: nl, nr = {ns}')

        return np.array(locs[0]), np.array(locs[1])

                
    def _check_sct_queues(self):
        assert len(self.sct_queues) == 2, f'currently support only bipartite mapping, got {len(self.sct_queues)}'


    def _check_scenes(self):
        assert len(self.scenes) == len(self.sct_queues), 'the number of scenes must == number of SCT queues'


    def _new_wait_list(self):
        assert len(self.sct_queues) == 2, f'currently support only bipartite mapping, got {len(self.sct_queues)}'
        return [[], [], []]     # the last list is for matches


class DisplayPipeline(Pipeline):

    def __init__(
            self,
            config: ConfigPipeline,
            input_queue: MyQueue,
            width: Union[int, None] = None,
            height: Union[int, None] = None,
            fps: Union[int, None] = None,
            path: Union[str, None] = None,
            name='DisplayPipeline'
    ) -> None:
        super().__init__(config, 0, name)
        
        self.input_queue = input_queue
        
        self.mode = self.config.get('DISPLAY_MODE')
        self.path = path
        self._check_mode()
        
        self.width = width
        self.height = height
        self.fps = fps
        self._check_fps()
        
        logger.info(f'{self.name}:\t initilized')
    
    def _start(self) -> None:

        assert self.config.get('RUNNING_MODE') == 'online'

        if self.mode == 'show':
            self._setup_window()
        elif self.mode == 'save':
            writer_created = False
        
        while not self.is_stopped():

            self.trigger_pause()

            item = self.input_queue.get(block=True)
            if item == '<EOS>':
                break

            frame_img = item['frame_img']
            frame_id = item['frame_id']

            if self.height is None:
                self.height = frame_img.shape[0]
            if self.width is None:
                self.width = frame_img.shape[1]

            if self.mode == 'save' and not writer_created:  # type: ignore
                writer = cv2.VideoWriter(
                    self.path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    (self.width, self.height)
                )
                writer_created = True

            frame_img = cv2.resize(frame_img, (self.width, self.height))

            logger.debug(f'{self.name}:\t {self.mode} from {self.input_queue.name}\t frame_id={frame_id}')
            
            if self.mode == 'show':
                cv2.imshow(self.name, frame_img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord(' '):                    
                    self.switch_pausing()
                    cv2.waitKey(0)
                    self.switch_pausing()
            elif self.mode == 'save':
                writer.write(frame_img)     # type: ignore
            
        if self.mode == 'show':
            cv2.destroyAllWindows()
        elif self.mode == 'save':
            writer.release()      # type: ignore

    
    def _setup_window(self) -> None:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    
    def _check_fps(self):
        assert self.fps is not None
        self.fps = float(self.fps)

    
    def _check_mode(self):
        if self.mode == 'save':
            assert self.path is not None


class ExportPipeline(Pipeline):

    def __init__(
            self,
            config: ConfigPipeline,
            input_queue: MyQueue,
            path: str, 
            name='ExportPipeline'
    ) -> None:
        super().__init__(config, 0, name)  # type: ignore

        self.input_queue = input_queue

        self.path = path
        self._check_path()

        logger.info(f'{self.name}:\t initialized')


    def _start(self) -> None:

        f = open(self.path, 'w')
        still_wait = True
        
        while not self.is_stopped():

            items = []
            if self.config.get('RUNNING_MODE') == 'online':
                if still_wait:
                    item = self.input_queue.get(block=True)
                    if item == '<EOS>':
                        still_wait = False
                    else:
                        items = [item]
            else:
                while still_wait:
                    item = self.input_queue.get()
                    if item == '<EOS>':
                        still_wait = False
                    else:
                        items.append(item)
            
            if not still_wait and self.config.get('RUNNING_MODE') == 'online':
                f.close()
                break

            logger.debug(f'{self.name}:\t processing {len(items)} frames')

            for item in items:
                print(str(item), file=f)
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break
        
        f.close()

    
    def _check_path(self):
        parent, filename = os.path.split(self.path)
        if not os.path.exists(parent) and parent != '':
            logger.debug(f'{self.name}:\t create directory {parent}')
            os.makedirs(parent)        