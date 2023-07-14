from typing import Union, List, Dict, Any
import queue
from threading import Lock, Thread
from datetime import datetime
import os
import time
import cv2
import logging
import sys
import yaml
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import shutil
from copy import deepcopy

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

sys.path.append(str(HERE))

from general import load_roi, load_homo, calc_loc                   # type: ignore
from map_utils import hungarian, map_mono
from vis_utils import plot_box, plot_skeleton_kpts, plot_loc, plot_roi
from filter import FilterBase, IQRFilter, GMMFilter

logger = logging.getLogger(__name__)

# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s\t|%(funcName)30s\t|%(lineno)d\t|%(levelname)s\t|%(message)s')

# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(formatter)
# logger.addHandler(handler)

#logging.FileHandler("~/Downloads/log.txt"),


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(funcName)20s\t|%(lineno)d\t|%(levelname)s\t|%(message)s',
    handlers=[
        #logging.FileHandler("~/Downloads/log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)


class Scene:

    def __init__(
            self,
            width: Union[int, float, None] = None,
            height: Union[int, float, None] = None,
            roi: Union[np.ndarray, None] = None,
            roi_test_offset: Union[int, float] = 0,
            name='Scene'
    ) -> None:

        self.width = width
        self.height = height
        
        self.roi = roi   
        self._check_roi()
        self.roi_test_offset = roi_test_offset

        self.name = name
    

    def is_in_roi(self, x: Union[tuple, list, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if point(s) (x, y) is in the scene's roi.
        
        If x is not a numpy array represent only 1 point, then return bool object.
        """
        is_numpy = True
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            is_numpy = False
        
        x = np.float32(x.reshape(-1, 2))    # type: ignore
        ret = [cv2.pointPolygonTest(self.roi, p, True) >= self.roi_test_offset for p in x]      # type: ignore

        if not is_numpy and len(ret) == 1:
            return ret[0]
        else:
            return np.array(ret, dtype=bool)
        
    
    def has_roi(self) -> bool:
        return self.roi is not None
    

    def _check_roi(self):
        if self.roi is not None:
            assert isinstance(self.roi, np.ndarray)
            self.roi = np.int32(self.roi)   # type: ignore


class ConfigPipeline:

    def __init__(
            self, 
            config_path: str, 
            name='ConfigPipeline'
    ) -> None:
        
        self.name = name

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            logger.info(f'{self.name}:\t loading config at {config_path}')
        self._check_config()

        self.stopped = False
        self.pausing = False

        self.lock = Lock()
        
        logger.info(f'{self.name}:\t initilized')

    
    def get(self, attr:str):
        return self.config[attr]


    def stop(self) -> None:
        self.lock.acquire()
        if not self.stopped:
            self.stopped = True
            logger.info(f'{self.name}:\t stop locked')
        self.lock.release()


    def is_stopped(self) -> bool:
        return self.stopped

    
    def switch_pausing(self) -> None:
        self.lock.acquire()
        self.pausing = not self.pausing
        logger.info(f'{self.name}:\t pausing switched')
        self.lock.release()


    def is_pausing(self) -> bool:
        return self.pausing

    
    def _check_config(self):
        assert self.config.get('RUNNING_MODE') in ['online', 'offline']

        # camera
        
        # SCTPipeline
        assert self.config.get('DETECTION_MODE') in ['pose', 'box']
        assert self.config.get('TRACKING_MODE') in ['bytetrack']
        
        # STAPipeline
        assert self.config.get('LOC_INFER_MODE') in [1, 2, 3]
        assert self.config.get('FP_FILTER') in [None, 'gmm', 'iqr']
        assert isinstance(self.config.get('FP_REMAP'), bool)
        assert self.config.get('MIN_SAMPLE_SIZE_TO_FP_FILTER') >= 1
        assert self.config.get('MIN_TIME_CORRESPONDENCES') >= 1
        assert isinstance(self.config.get('TIME_DIFF_THRESH'), (float, int)) or self.config.get('TIME_DIFF_THRESH') is None
        assert self.config.get('DIST_WINDOW_SIZE') >= 1 and self.config.get('DIST_WINDOW_SIZE') % 2 == 1
        assert self.config.get('DIST_WINDOW_BOUNDARY') >= 0

        # display
        assert self.config.get('DISPLAY_MODE') in ['show', 'save']


class MyQueue(queue.Queue):

    def __init__(self, maxsize: int = 0, name: Any = 'MyQueue') -> None:
        super().__init__(maxsize)
        self.name = name

        logger.debug(f'{self.name}:\t initilized')


class Pipeline(ABC):

    @abstractmethod
    def __init__(
            self, 
            config: ConfigPipeline,
            online_put_sleep: Union[int, float] = 0, 
            name='Pipeline'
    ) -> None:
        
        self.config = config
        self.name = name
        self.thread = Thread(target=self._start, args=(), name=self.name)

        self.output_queues = {}  
        self.online_put_sleep = online_put_sleep

        self.lock = Lock()
            

    @abstractmethod
    def _start(self) -> None:
        pass
    

    def start(self) -> None:
        self.thread.start()

        logger.info(f'{self.name}:\t started')
    

    def join(self) -> None:
        self.thread.join()

        logger.info(f'{self.name}:\t joined')


    def stop(self) -> None:
        if not self.config.is_stopped():
            logger.info(f'{self.name}:\t triggering config stop...')
            self.config.stop()
            self.output_queues = {}


    def is_stopped(self) -> bool:
        return self.config.is_stopped() is True


    def trigger_pause(self):
        while self.config.is_pausing():
            pass


    def switch_pausing(self):
        self.config.switch_pausing()

    
    def _put_to_output_queues(self, item, msg=None):
        for k, oq in self.output_queues.items():
            oq.put(
                item,
                block=self.config.get('QUEUE_PUT_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )
        
            logger.debug(f"{self.name}:\t put to {oq.name} with message: {msg}")
        

    def add_output_queue(self, queue, key):
        self.lock.acquire()
        assert key not in self.output_queues, f'{key} already exists in output_queues'
        self.output_queues[key] = queue
        self.lock.release()
        
        logger.info(f'{self.name}:\t added output queue with key {key}. Having {len(self.output_queues)} output queues.')   # type: ignore

    
    def remove_output_queue(self, key):
        self.lock.acquire()
        del self.output_queues[key]
        self.lock.release()

        logger.info(f'{self.name}:\t removed output queue with key {key}. Remaining {len(self.output_queues)} output queues.')  # type: ignore


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

        ##### START HERE #####
        self.signin_user_id = None
        ##### END HERE #####
        
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

            if not ret:
                logger.info(f'{self.name}:\t disconnected from {self.source}')
                if self.config.get('RUNNING_MODE') == 'online' or self.ret_img:
                    sleep = self.config.get('ONLINE_SLEEP_BEFORE_STOP')
                    logger.info(f'{self.name}:\t sleeping {sleep} second before stop all')
                    time.sleep(sleep)
                    self.stop()
                break
            
            frame_id += 1

            if self.meta is None:
                frame_time = datetime.now().timestamp()
            else:   # if reading from video on disk
                frame_time = datetime.strptime(self.meta['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (frame_id - self.meta['start_frame_id']) / self.fps

            ##### START HERE #####
            # mock check-in
            if '<cam_id=1>' in self.name:
                if frame_id == 184:
                    self.signal_signin(4)
                elif frame_id == 479:
                    self.signal_signin(3)
                elif frame_id == 878:
                    self.signal_signin(5)

            signin_user_id = self._observe_signin()
            if signin_user_id is not None:
                logger.info(f'{self.name}: get sign-in signal')
            ##### END HERE #####
            
            out_item = {
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

    
    def _observe_signin(self):
        signin_user_id, self.signin_user_id = self.signin_user_id, None
        return signin_user_id
        

    def signal_signin(self, sid):
        self.signin_user_id = sid
        

class Tracker:

    def __init__(
            self,
            detection_mode: Union[str, None] = None,
            tracking_mode: Union[str, None] = None,
            txt_path: Union[str, None] = None,
            name='Tracker'
    ) -> None:
        
        self.detection_mode = detection_mode
        self.tracking_mode = tracking_mode
        self.txt_path = txt_path
        self.name = name
        
        # if using offline mock tracking result
        if txt_path is not None:
            try:
                self.seq = np.loadtxt(txt_path)
            except:
                self.seq = np.loadtxt(txt_path, delimiter=',')
            logger.info(f'{self.name}:\t load tracking result from .txt at {txt_path}')
        
        self.name = name

        logger.info(f'{self.name}:\t initialized with DETECTION_MODE={self.detection_mode}, TRACKING_MODE={self.tracking_mode}')
        
    
    def infer(self, img: Union[np.ndarray, None], frame_id: Union[int, None] = None) -> np.ndarray:

        if self.txt_path is not None:
            assert  isinstance(frame_id, int), 'frame_id must be provided in mock test'

            # if detection_mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
            # if detection_mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
            dets = self.seq[self.seq[:, 0] == frame_id]

            logger.debug(f'{self.name}:\t frame {frame_id} detected {len(dets)} people')

            return dets

        else:
            raise NotImplementedError()


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
        
        while not self.is_stopped():

            self.trigger_pause()

            if self.config.get('RUNNING_MODE') == 'online':
                items = [self.input_queue.get(block=True)]
            else:
                items = []
                while not self.input_queue.empty():
                    items.append(self.input_queue.get())

            T = len(items)

            logger.debug(f'{self.name}:\t processing {T} frames')

            mid_time = time.time()
            pre_time = mid_time - start_time
            
            for item in items:
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

        while not self.is_stopped():

            self.trigger_pause()
            
            # load both the coming and waiting list
            for iq, wl in zip(self.input_queues, self.wait_list):

                if self.config.get('RUNNING_MODE') == 'online':
                    items = [iq.get(block=True)]
                else:
                    items = []
                    while not iq.empty():
                        items.append(iq.get())

                wl.extend(items)
            
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
                break
    

    def _check_input_queues(self):
        assert len(self.input_queues) == 2, f'currently support only bipartite mapping, got {len(self.input_queues)}'


    def _new_wait_list(self):
        return [[] for _ in range(len(self.input_queues))]


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
        
        while not self.is_stopped():

            self.trigger_pause()
            
            for wl, iq in zip(self.wait_list, self.sct_queues + [self.sync_queue]):     # type: ignore
                if self.config.get('RUNNING_MODE') == 'online':
                    items = [iq.get(block=True)]
                else:
                    items = []
                    while not iq.empty():
                        items.append(iq.get())
                
                wl.extend(items)

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


class VisualizePipeline(Pipeline):

    def __init__(
            self, 
            config: ConfigPipeline, 
            annot_queue: MyQueue,
            video_queue: MyQueue,
            scene: Union[Scene, None] = None,
            online_put_sleep: Union[int, float] = 0,
            name='Visualizer'
    ) -> None:
        super().__init__(config, online_put_sleep, name)
        
        self.annot_queue = annot_queue
        
        self.video_queue = video_queue
        self._check_video_queue()

        self.scene = scene

        self.wait_list = self._new_wait_list()
        self._create_cache_dir()

        logger.info(f'{self.name}:\t initialized')

    
    def _start(self):

        assert self.config.get('RUNNING_MODE') == 'online'

        start_time = time.time()
        pop_signin_count = 0
        pop_signin_id = None

        while not self.is_stopped():

            self.trigger_pause()
            
            for i, (wl, iq) in enumerate(zip(self.wait_list, [self.annot_queue, self.video_queue])):    # type: ignore
                wl.append(iq.get(block=True))

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
                if self.is_stopped():
                    break

                frame_img = adict['frame_img'][1][t]
                dets = adict['sct_output'][0][t]
                detection_mode = adict['sct_detection_mode'][0][t]

                # if scene is provided then plot roi and location point
                if self.scene is not None and self.scene.has_roi():
                    frame_img = plot_roi(frame_img, self.scene.roi, self.config.get('VIS_ROI_THICKNESS'))

                # expecting dets to be in the format of [uid, tid, x1, y1, x2, y2, score, ...]
                if detection_mode == 'box':
                    frame_img = plot_box(frame_img, dets, self.config.get('VIS_SCT_BOX_THICKNESS'), texts=np.int32(dets[:, 0]), text_prefix='user_id=')
                elif detection_mode == 'pose':
                    frame_img = plot_box(frame_img, dets, self.config.get('VIS_SCT_BOX_THICKNESS'), texts=np.int32(dets[:, 0]), text_prefix='user_id=')
                    # for kpt in dets[:, 10:]:
                    #     frame_img = plot_skeleton_kpts(frame_img, kpt.T, 3)
                else:
                    raise NotImplementedError()
                
                locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'))
                frame_img = plot_loc(frame_img, np.concatenate([dets[:, :2], locs], axis=1), self.config.get('VIS_SCT_LOC_RADIUS'))

                ##### START HERE #####
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
                ##### END HERE #####
                
                out_item = {
                    'frame_img': frame_img,                   # type: ignore
                    'frame_id': adict['frame_id'][1][t]
                }

                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item, f"frame_id={adict['frame_id'][1][t]}")

                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')
            
            start_time = mid_time
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break

    
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

            assert self.config.get('RUNNING_MODE') == 'online'

            item = self.input_queue.get(block=True)

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
        
        while not self.is_stopped():

            if self.config.get('RUNNING_MODE') == 'online':
                items = [self.input_queue.get(block=True)]
            else:
                items = []
                while not self.input_queue.empty():
                    items.append(self.input_queue.get())

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



sys.path.append(str(Path(__file__).parent.parent.parent))           
from app.models import User


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

        self._load_users()

    
    def _load_users(self):
        self.users = {}
        with self.app.app_context():
            users = User.query.filter(User.role.in_(['intern', 'engineer'])).all()
            for user in users:
                self.users[user.id] = user
                user.load_workareas()
                user.load_next_workshift()
        
    
    def _start(self) -> None:

        start_time = time.time()
        
        while not self.is_stopped():

            self.trigger_pause()

            for i, q in enumerate([self.sct_queues, self.sta_queues]):
                for k, v in q.items():
                    if k not in self.wait_list[i]:
                        self.wait_list[i][k] = []
                    
                    if self.config.get('RUNNING_MODE') == 'online':
                        items = [v.get(block=True)]
                    else:
                        items = []
                        while not v.empty():
                            items.append(v.get())

                    self.wait_list[i][k].extend(items)
            
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
                        with self.app.app_context():
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
                
                with self.app.app_context():
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
                        # user.update_detection(cid, dtime, loc)
                
                for (cid, tid), d in sct_outputs.items():
                    d[0] = -1
                    out_visualize[cid].append(d)
                
                for cid in out_visualize:
                    out_visualize[cid] = np.array(out_visualize[cid])   # type: ignore
                    if len(out_visualize[cid]) == 0:    
                        out_visualize[cid] =  np.empty((0, 10))          # type: ignore , whatever, as long as dim2 >= 2

     
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.online_put_sleep - (pre_time / T + end_time - mid_time)) # TODO fix this
                time.sleep(sleep)
                # expecting key of the output queue is camera ID
                for cid, oq in self.output_queues.items():  # type: ignore
                    oq.put(                                
                        {
                            'frame_id': active_list[0][cid][t]['frame_id'],
                            'sct_output': out_visualize[cid],
                            'sct_detection_mode': active_list[0][cid][t]['sct_detection_mode'],
                            'signin_user_id': signin_user_id if cid == self.checkin_cid else None
                        },
                        block=self.config.get('QUEUE_PUT_BLOCK'),
                        timeout=self.config.get('QUEUE_TIMEOUT')
                    )
                    
                mid_time = end_time
                logger.debug(f'{self.name}:\t slept {sleep}')

            start_time = mid_time


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


    def init_app(self, app, db):
        self.app = app
        self.db = db
        self.config = app.config['PIPELINE']

        logger.info(f'{self.name}:\t initialized with app and db')


    def register_camera(
            self,
            cam_id: int,
            address: str,
            meta_path: str,
            txt_path: str,
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
            online_put_sleep=self.config.get('CAMERA_SLEEP_MUL_FACTOR') / meta['fps'],
            name=f'PL Camera-<cam_id={cam_id}>',
        )

        # create tracker
        tracker = Tracker(
            detection_mode=self.config.get('DETECTION_MODE'), 
            tracking_mode=self.config.get('TRACKING_MODE'), 
            txt_path=txt_path,
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

        assert hasattr(self, 'pl_cameras')
        assert cam_id in self.pl_cameras

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
        
        assert cam_id in self.pl_cameras
        oq_video = MyQueue(name=f'IQ-Visualize_Video-<cam_id={cam_id}>')
        oq_annot = MyQueue(name=cam_id)     # must be exactly cam_id
        self.pl_cameras[cam_id].add_output_queue(oq_video, oq_video.name)
        self.pl_mcmap.add_output_queue(oq_annot, oq_annot.name)

        pl_visualize = VisualizePipeline(
            config=self.config,
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
        
        lock = Lock()

        lock.acquire()
        if cam_id not in self.pl_visualizes:
            lock.release()
            return

        if cam_id not in self.display_queues:
            lock.release()
            return
        
        if key not in self.display_queues[cam_id]:
            lock.release()
            return
        
        # remove output queue from visualize
        pl_visualize = self.pl_visualizes[cam_id]
        pl_visualize.remove_output_queue(key)
        display_queue = self.display_queues[cam_id][key]
        del self.display_queues[cam_id][key]
        del display_queue
        
        # stop visualize if no display queue
        if len(self.display_queues[cam_id]) == 0:

            logger.info(f'Destructing VisualizePipeline for camera {cam_id}...')

            # pl_visualize.stop()
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

        


def main(kwargs):

    config = ConfigPipeline(kwargs['config'])

    # input queues
    iq_sct_1 = MyQueue(name='SCTPipeline-1-Input-Queue')
    iq_sct_2 = MyQueue(name='SCTPipeline-2-Input-Queue')
    
    iq_sync_1 = MyQueue(name='Sync-1-Input-Queue')
    iq_sync_2 = MyQueue(name='Sync-2-Input-Queue')
    
    iq_sta_sct_1 = MyQueue(name='STAPipeline-1-InputSCT-Queue')
    iq_sta_sct_2 = MyQueue(name='STAPipeline-2-InputSCT-Queue')
    iq_sta_sync = MyQueue(name='STAPipeline-InputSync-Queue')

    iq_exp_sta = MyQueue(name='ExportSTA-Input-Queue')

    # ONLY USE META IF CAPTURING VIDEOS
    meta_1 = yaml.safe_load(open(kwargs['meta_1'], 'r'))
    meta_2 = yaml.safe_load(open(kwargs['meta_2'], 'r'))
    pl_camera_1_noretimg = CameraPipeline(config, kwargs['camera_1'], meta=meta_1, ret_img=False, name='CameraPipeline-1-NoRetImg')
    pl_camera_1_noretimg.add_output_queue(iq_sct_1, iq_sct_1.name)
    pl_camera_1_noretimg.add_output_queue(iq_sync_1, iq_sync_1.name)
    pl_camera_2_noretimg = CameraPipeline(config, kwargs['camera_2'], meta=meta_2, ret_img=False, name='CameraPipeline-2-NoRetImg')
    pl_camera_2_noretimg.add_output_queue(iq_sct_2, iq_sct_2.name)
    pl_camera_2_noretimg.add_output_queue(iq_sync_2, iq_sync_2.name)
    
    tracker1 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_1'], name='Tracker-1')
    tracker2 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_2'], name='Tracker-2')
    pl_sct_1 = SCTPipeline(config, tracker=tracker1, input_queue=iq_sct_1, name='SCTPipeline-1')
    pl_sct_1.add_output_queue(iq_sta_sct_1, iq_sta_sct_1.name)
    pl_sct_2 = SCTPipeline(config, tracker=tracker2, input_queue=iq_sct_2, name='SCTPipeline-2')
    pl_sct_2.add_output_queue(iq_sta_sct_2, iq_sta_sct_2.name)
    
    pl_sync = SyncPipeline(config, [iq_sync_1, iq_sync_2])
    pl_sync.add_output_queue(iq_sta_sync, iq_sta_sync.name)
    
    roi_2 = load_roi(kwargs['roi'], pl_camera_2_noretimg.width, pl_camera_2_noretimg.height)
    if kwargs['camera_1'] == kwargs['camera_2']:
        if kwargs['matches'] is not None:   # pair of camera 1
            homo = load_homo(kwargs['matches'])
            roi_2 = cv2.perspectiveTransform(roi_2, np.linalg.inv(homo)) # type: ignore
        else:   # pair of camera 2
            homo = None
        roi_1 = roi_2
        
    else:   # 2 different cameras
        homo = load_homo(kwargs['matches'])
        roi_1 = cv2.perspectiveTransform(roi_2, np.linalg.inv(homo)) # type: ignore
    scene_1 = Scene(pl_camera_1_noretimg.width, pl_camera_1_noretimg.height, roi_1, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-1')
    scene_2 = Scene(pl_camera_2_noretimg.width, pl_camera_2_noretimg.height, roi_2, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-2')
    pl_sta = STAPipeline(config, [scene_1, scene_2], homo, [iq_sta_sct_1, iq_sta_sct_2], iq_sta_sync, name='STAPipeline')
    pl_sta.add_output_queue(iq_exp_sta, iq_exp_sta.name)

    pl_exp = ExportPipeline(config, iq_exp_sta, kwargs['out_sta_txt'])

    # start
    pl_camera_1_noretimg.start()
    pl_camera_2_noretimg.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_camera_1_noretimg.join()     # offline
        pl_camera_2_noretimg.join()     # offline
    
    pl_sct_1.start()
    pl_sct_2.start()
    pl_sync.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_sct_1.join()                 # offline
        pl_sct_2.join()                 # offline
        pl_sync.join()                  # offline

    pl_sta.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_sta.join()                   # offline

    # export
    pl_exp.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_exp.join()                   # offline

    if config.get('RUNNING_MODE') == 'online':
        pl_camera_1_noretimg.join()
        pl_camera_2_noretimg.join()
        pl_sct_1.join()
        pl_sct_2.join()
        pl_sync.join()
        pl_sta.join()
        pl_exp.join()


if __name__ == '__main__':

    kwargs = {
        'config': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/mct/utils/config.yaml',
        
        'meta_1': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/meta/121_00004_2023-02-28_18-00-00-000000.yaml',
        'meta_2': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/meta/127_00004_2023-02-28_18-00-00-000000.yaml',
        'camera_1': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/videos/121_00004_2023-02-28_18-00-00-000000.avi',
        'camera_2': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/videos/127_00004_2023-02-28_18-00-00-000000.avi',
        
        'sct_1': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/YOLOv8l_pretrained-640-ByteTrack/sct/121_00004_2023-02-28_18-00-00-000000.txt',
        'sct_2': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/YOLOv8l_pretrained-640-ByteTrack/sct/127_00004_2023-02-28_18-00-00-000000.txt',

        'roi': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/roi_127.txt',
        'matches': '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v3/matches_121_to_127.txt',

        'out_sta_txt': 'test.txt',
        'out_sct_vid_1': 'test.avi',
        'out_sct_vid_2': 'test.avi',
        'out_sta_vid': 'test.avi'

    }
    
    main(kwargs)

