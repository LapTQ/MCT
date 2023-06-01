from typing import Union, List, Dict
from queue import Queue
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
from copy import deepcopy

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from general import load_roi, load_homo, calc_loc                   # type: ignore
from map_utils import hungarian, map_mono
from vis_utils import plot_box, plot_skeleton_kpts, plot_loc, plot_roi
from filter import FilterBase, IQRFilter, GMMFilter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(thread)d\t|%(funcName)s\t|%(lineno)d\t|%(levelname)s\t|%(message)s',
    handlers=[
        #logging.FileHandler("~/Downloads/log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)


class Config:

    def __init__(
            self, 
            config_path: str, 
            name='Config'
    ) -> None:
        
        self.name = name

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            logging.info(f'{self.name}:\t loading config at {config_path}')
        self._check_config()

        self.stopped = False
        self.pausing = False

        self.lock = Lock()
        
        logging.info(f'{self.name}:\t initilized')

    
    def get(self, attr:str):
        return self.config[attr]


    def stop(self) -> None:
        self.lock.acquire()
        self.stopped = True
        logging.info(f'{self.name}:\t stop locked')
        self.lock.release()


    def is_stopped(self) -> bool:
        return self.stopped

    
    def switch_pausing(self) -> None:
        self.lock.acquire()
        self.pausing = not self.pausing
        logging.info(f'{self.name}:\t pausing switched')
        self.lock.release()


    def is_pausing(self) -> bool:
        return self.pausing

    
    def _check_config(self):
        assert self.config.get('RUNNING_MODE') in ['online', 'offline']
        
        # queue
        assert self.config.get('QUEUE_MAXSIZE') >= 0
        assert isinstance(self.config.get('QUEUE_GET_BLOCK'), bool)
        assert isinstance(self.config.get('QUEUE_PUT_BLOCK'), bool)
        assert isinstance(self.config.get('QUEUE_TIMEOUT'), (float, int)) or self.config.get('QUEUE_TIMEOUT') is None
        assert isinstance(self.config.get('QUEUE_GET_MANY_SIZE'), int) or self.config.get('QUEUE_GET_MANY_SIZE') == 'all'

        # camera pipeline
        
        # SCT pipeline
        assert self.config.get('DETECTION_MODE') in ['pose', 'box']
        assert self.config.get('TRACKING_MODE') in ['bytetrack']
        
        # STA pipeline
        assert self.config.get('LOC_INFER_MODE') in [1, 2, 3]
        assert self.config.get('FP_FILTER') in [None, 'gmm', 'iqr']
        assert isinstance(self.config.get('FP_REMAP'), bool)
        assert self.config.get('MIN_SAMPLE_SIZE_TO_FP_FILTER') >= 1
        assert self.config.get('MIN_TIME_CORRESPONDENCES') >= 1
        assert isinstance(self.config.get('TIME_DIFF_THRESH'), (float, int)) or self.config.get('TIME_DIFF_THRESH') is None
        assert self.config.get('DIST_WINDOW_SIZE') >= 1 and self.config.get('DIST_WINDOW_SIZE') % 2 == 1
        assert self.config.get('DIST_WINDOW_BOUNDARY') >= 0

        # display pipeline
        assert self.config.get('DISPLAY_MODE') in ['show', 'save']


class MyQueue:

    def __init__(
            self, 
            maxsize: int, 
            name='Queue'
    ) -> None:
        self.name = name
        self.maxsize = maxsize
        self.queue = Queue(maxsize)

        self.lock = Lock()

        logging.debug(f'{self.name}:\t initilized')
    

    def get(self, block=True, timeout=None):
        ret = self.queue.get(block, timeout)
        logging.debug(f'{self.name}:\t dequeue 1 item, containing {self.queue.qsize()}')
        return ret
    

    def put(self, item, block=True, timeout=None) -> None:
        self.queue.put(item, block, timeout)
        logging.debug(f'{self.name}:\t enqueue 1 item, containing {self.queue.qsize()}')


    def empty(self) -> bool:
        return self.queue.empty()


    def get_many(self, size='all', block=True, timeout=None) -> list:
        
        ret = []
        
        self.lock.acquire()
        
        if size == 'all':
            old_queue = self.queue
            self.queue = Queue(self.maxsize)
            
            logging.debug(f'{self.name}:\t dequeue {len(ret)} items (/{size} requested).')
            
            self.lock.release()
            
            while not old_queue.empty():
                ret.append(old_queue.get(block, timeout))
        else:
            assert isinstance(size, int) and size > 0, 'size must be a positive integer'
            for _ in range(size):
                if not self.queue.empty():
                    ret.append(self.queue.get(block, timeout))
            
            logging.debug(f'{self.name}:\t dequeue {len(ret)} items (/{size} requested).')
            
            self.lock.release()

        return ret


class Pipeline(ABC):

    @abstractmethod
    def __init__(
            self, 
            config: Config,
            output_queues: Union[List[MyQueue], MyQueue, None] = None, 
            name='Pipeline component'
    ) -> None:
        
        self.config = config
        self.name = name
        self.thread = Thread(target=self._start, args=(), name=self.name)

        self.output_queues = output_queues
        self._check_output_queues()   

        self.lock = Lock()
            

    @abstractmethod
    def _start(self) -> None:
        pass
    

    def start(self) -> None:
        self.thread.start()
    

    def join(self) -> None:
        self.thread.join()


    def stop(self) -> None:
        logging.info(f'{self.name}:\t triggering config stop...')
        self.output_queues = {}
        self.config.stop()


    def is_stopped(self) -> bool:
        return self.config.is_stopped() is True


    def trigger_pause(self):
        while self.config.is_pausing():
            pass


    def _check_output_queues(self) -> None:
        if self.output_queues is None:
            self.output_queues = []
        elif isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]
        self.output_queues = {i: q for i, q in enumerate(self.output_queues)}


    def switch_pausing(self):
        self.config.switch_pausing()

    
    def _put_to_output_queues(self, item, msg=None):
        self.lock.acquire()
        for k, oq in self.output_queues.items():                    # type: ignore
            oq.put(
                item,
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )
        
            logging.debug(f"{self.name}:\t put to {oq.name} with message: {msg}")
        self.lock.release()
        

    def add_output_queue(self, queue, key):
        self.lock.acquire()
        assert key not in self.output_queues, f'{key} already exists in output_queues'
        self.output_queues[key] = queue # type: ignore
        self.lock.release()

    
    def remove_output_queue(self, key):
        self.lock.acquire()
        del self.output_queues[key] # type: ignore
        self.lock.release()


class Camera(Pipeline):

    def __init__(
            self,
            config: Config, 
            source: Union[int, str],
            meta: Union[dict, None] = None, 
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            ret_img: bool = True,
            name='Camera thread'
    ) -> None:
        super().__init__(config, output_queues, name)
        
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        self.meta = meta
        self._check_meta()

        self.ret_img = ret_img

        if self.config.get('CAMERA_FRAME_WIDTH') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('CAMERA_FRAME_WIDTH'))
        if self.config.get('CAMERA_FRAME_HEIGHT') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('CAMERA_FRAME_HEIGHT'))
        if self.config.get('CAMERA_FPS') is not None:
            assert self.meta is None, 'camera_fps must not be set when capturing videos from disk'
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('CAMERA_FPS'))

        ##### START HERE #####
        self.signin_sid = None
        ##### END HERE #####
        
        logging.info(f'{self.name}:\t initilized')

    
    def _start(self) -> None:

        if self.meta is None:
            frame_id = 0
        else:   # if reading from video on disk
            frame_id = self.meta['start_frame_id'] - 1

        start_time = time.time()

        while not self.is_stopped():

            self.trigger_pause()

            if not self.cap.isOpened():
                logging.info(f'{self.name}:\t problem connecting to {self.source}')
                self.stop()
                break

            if self.ret_img:
                # handle out-of-memory exclusively for offline
                if self.config.get('RUNNING_MODE') == 'offline':
                    time.sleep(self.config.get('VIS_OFFLINE_PUT_SLEEP'))
                ret, frame = self.cap.read()
            else:
                ret, frame = True, None     # for running offline without visualizing
                if frame_id == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    ret = False

            if not ret:
                logging.info(f'{self.name}:\t disconnected from {self.source}')
                if self.config.get('RUNNING_MODE') == 'online' or self.ret_img:
                    logging.info(f'{self.name}:\t sleeping a few second before stop all')
                    time.sleep(self.config.get('ONLINE_SLEEP_BEFORE_STOP'))
                    self.stop()
                break
            
            frame_id += 1

            if self.meta is None:
                frame_time = datetime.now().timestamp()
            else:   # if reading from video on disk
                frame_time = datetime.strptime(self.meta['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (frame_id - self.meta['start_frame_id']) / self.fps

            ##### START HERE #####
            # mock check-in
            if '<1>' in self.name:
                if frame_id == 184:
                    self.signal_signin(4)
                elif frame_id == 479:
                    self.signal_signin(3)
                elif frame_id == 878:
                    self.signal_signin(5)

            signin_sid = self._observe_signin()
            if signin_sid is not None:
                logging.info(f'{self.name}: get sign-in signal')
            ##### END HERE #####
            
            out_item = {
                'frame_img': frame,
                'frame_id': frame_id,
                'frame_time': frame_time,
                'signin_sid': signin_sid
            }
            
            logging.debug(f"{self.name}:\t sleep")
            end_time = time.time()
            # if reading from video on disk, then sleep according to fps to sync time.
            sleep = 0 if self.meta is None or self.config.get('RUNNING_MODE') == 'offline' \
                else max(0, self.config.get('CAMERA_SLEEP_MUL_FACTOR') / self.fps - (end_time - start_time))
            time.sleep(sleep)
            
            self._put_to_output_queues(out_item)

            start_time = end_time

        self.cap.release()

    
    @property
    def width(self):
        if self.meta is not None:
            return int(self.meta['width'])
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
    
    @property
    def height(self):
        if self.meta is not None:
            return int(self.meta['height'])
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    @property
    def fps(self):
        if self.meta is not None:
            return float(self.meta['fps'])
        else:
            return float(self.cap.get(cv2.CAP_PROP_FPS))


    def _check_meta(self) -> None:
        pass

    
    ###### START HERE ######
    def _observe_signin(self):
        signin_sid, self.signin_sid = self.signin_sid, None
        return signin_sid
        

    def signal_signin(self, sid):
        self.signin_sid = sid
    ###### END HERE ######
        

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
            logging.info(f'{self.name}:\t load tracking result from .txt at {txt_path}')
        
        self.name = name

        logging.info(f'{self.name}:\t initialized with DETECTION_MODE={self.detection_mode}, TRACKING_MODE={self.tracking_mode}')
        
    
    def infer(self, img: Union[np.ndarray, None], frame_id: Union[int, None] = None) -> np.ndarray:

        if self.txt_path is not None:
            assert  isinstance(frame_id, int), 'frame_id must be provided in mock test'

            # if detection_mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
            # if detection_mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
            dets = self.seq[self.seq[:, 0] == frame_id]

            logging.debug(f'{self.name}:\t frame {frame_id} detected {len(dets)} people')

            return dets

        else:
            raise NotImplementedError()


class Scene:

    def __init__(
            self,
            width: Union[int, float, None] = None,
            height: Union[int, float, None] = None,
            roi: Union[np.ndarray, None] = None,
            roi_test_offset=0,
            name='Scene'
    ) -> None:

        self.width = width
        self.height = height
        
        self.roi = roi   
        self._check_roi()

        self.roi_test_offset = roi_test_offset

        self.name = name

        logging.debug(f'{self.name}:\t initialized')
    

    def is_in_roi(self, x: np.ndarray) -> Union[bool, np.ndarray]:
        not_np = False
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            not_np = True
        
        x = np.float32(x.reshape(-1, 2))    # type: ignore

        ret = [cv2.pointPolygonTest(self.roi, p, True) >= self.roi_test_offset for p in x]      # type: ignore

        if not_np and len(ret) == 1:
            return ret[0]
        else:
            return np.array(ret, dtype=bool)
        
    
    def has_roi(self) -> bool:
        return self.roi is not None
    

    def _check_roi(self):
        if self.roi is not None:
            assert isinstance(self.roi, np.ndarray)
            self.roi = np.int32(self.roi)   # type: ignore


class SCT(Pipeline):

    def __init__(
            self, 
            config: Config, 
            tracker: Tracker,
            input_queue: MyQueue,
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            name='SCT'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.tracker = tracker
        
        self.input_queue = input_queue

        logging.info(f'{self.name}:\t initialized')

    
    def _start(self) -> None:

        start_time = time.time()
        
        while not self.is_stopped():

            self.trigger_pause()

            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            logging.debug(f'{self.name}:\t processing {len(items)} frames')
            
            for item in items:
                dets = self.tracker.infer(item['frame_img'], item['frame_id'])
                
                out_item = {
                    'frame_id': item['frame_id'],
                    'sct_output': dets,
                    'sct_detection_mode': self.tracker.detection_mode,
                    'sct_tracking_mode': self.tracker.tracking_mode,
                    ##### START HERE #####
                    'signin_sid': item['signin_sid']
                    ##### END HERE #####
                }

                logging.debug(f'{self.name}:\t sleep due to .txt tracking result')
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.config.get('SCT_TXT_SLEEP') - (end_time - start_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item)

                start_time = end_time

            if self.config.get('RUNNING_MODE') == 'offline':
                break


class SyncFrame(Pipeline):

    def __init__(
            self, 
            config: Config, 
            input_queues: List[MyQueue],
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            name='SyncFrame'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.input_queues = input_queues
        self._check_input_queues()

        self.wait_list = self._new_wait_list()

        logging.debug(f'{self.name}:\t initialized')

    
    def _start(self) -> None:

        start_time = time.time()

        while not self.is_stopped():

            self.trigger_pause()
            
            # load both the coming and waiting list
            for c in range(len(self.input_queues)):
                self.wait_list[c].extend(
                    self.input_queues[c].get_many(
                        size=self.config.get('QUEUE_GET_MANY_SIZE'),
                        block=self.config.get('QUEUE_GET_BLOCK'),
                        timeout=self.config.get('QUEUE_TIMEOUT')
                    )
                )
            
            # do not map if the number of pairs are so small
            if min(len(self.wait_list[0]), len(self.wait_list[1])) < self.config.get('MIN_TIME_CORRESPONDENCES'):
                logging.debug(f'{self.name}:\t wait list not enough for mapping time, waiting...')
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

            # add un-processed items to wait list
            if len(c1_adict_matched) > 0:
                logging.debug(f'{self.name}:\t processing {len(c1_adict_matched)} pairs of frames')
                self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
                self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[0])} of {self.input_queues[0].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[1])} of {self.input_queues[1].name} to wait list')
            
            logging.debug(f'{self.name}:\t processing {len(c1_adict_matched)} frames')
            
            for i, j in zip(c1_adict_matched, c2_adict_matched):

                logging.debug(f"{self.name}:\t sync frame_id={adict['frame_id'][0][i]} ({datetime.fromtimestamp(adict['frame_time'][0][i]).strftime('%M-%S-%f')}) with frame_id={adict['frame_id'][1][j]} ({datetime.fromtimestamp(adict['frame_time'][1][j]).strftime('%M-%S-%f')})")

                out_item = {
                    'frame_id_match': (adict['frame_id'][0][i], adict['frame_id'][1][j])
                }

                logging.debug(f'{self.name}:\t sleep')
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.config.get('SCT_TXT_SLEEP') - (end_time - start_time))
                time.sleep(sleep)

                self._put_to_output_queues(out_item)

                start_time = end_time
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break
    

    def _check_input_queues(self):
        assert len(self.input_queues) == 2, f'currently support only bipartite mapping, got {len(self.input_queues)}'


    def _new_wait_list(self):
        return [[] for _ in range(len(self.input_queues))]


class STA(Pipeline):

    def __init__(
            self, 
            config: Config, 
            scenes: List[Scene],
            homo: Union[np.ndarray, None],
            sct_queues: List[MyQueue],
            sync_queue: MyQueue,
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            name='STA'
    ) -> None:
        super().__init__(config, output_queues, name)
        """
        homo: tranform the view of scene[0] to scene[1]
        """

        self.sct_queues = sct_queues
        self._check_sct_queues()

        self.sync_queue = sync_queue

        self.scenes = scenes
        self._check_scenes()

        self.homo = homo

        self.history = []       # [(frame_id_1, frame_id_2, ({id_1: loc_1, ...}, {id_2: loc_2, ...}), [(mid_1, mid_2), ...]), ...]
        self.distances = []     # record distances for FP elimination
        self.wait_list = self._new_wait_list()

        logging.info(f'{self.name}:\t initialized')


    def _start(self) -> None:

        start_time = time.time()
        
        while not self.is_stopped():

            self.trigger_pause()

            t0 = time.time() ################
            
            for wl, iq in zip(self.wait_list, self.sct_queues + [self.sync_queue]):     # type: ignore
                wl.extend(
                    iq.get_many(
                        size=self.config.get('QUEUE_GET_MANY_SIZE'),
                        block=self.config.get('QUEUE_GET_BLOCK'),
                        timeout=self.config.get('QUEUE_TIMEOUT')
                    )
                )

            active_list = self.wait_list
            self.wait_list = self._new_wait_list()

            # TODO CHUYEN 1 SO FRAME CUOI VE WAITING LIST => WINDOW

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
            
            if len(c1_adict_matched) <= self.config.get('DIST_WINDOW_BOUNDARY'):  # critical because there might be no matches (e.g Sync result comes slower), so we need to wait more
                self.wait_list = active_list
                continue

            # add un-processed items to wait list, postponse the last BOUNDARY frames for the next iteration
            self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1 - self.config.get('DIST_WINDOW_BOUNDARY'):])
            self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1 - self.config.get('DIST_WINDOW_BOUNDARY'):])
            self.wait_list[2].extend(active_list[2][sync_adict_matched[-1] + 1 - self.config.get('DIST_WINDOW_BOUNDARY'):])
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[0])} of {self.sct_queues[0].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[1])} of {self.sct_queues[1].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[2])} of {self.sync_queue.name} to wait list')

            # using continuous indexes from 0 -> T-1 rather than discrete indexes
            T = len(c1_adict_matched)
            logging.debug(f'{self.name}:\t processing {T} pairs of frames')
            del adict['frame_id_match']
            for k in adict:
                adict[k][0] = [adict[k][0][idx] for idx in c1_adict_matched]
                adict[k][1] = [adict[k][1][idx] for idx in c2_adict_matched]

            t1 = time.time() ################
            
            logging.debug(f"{self.name}:\t infer location from detection with LOC_INFER_MODE={self.config.get('LOC_INFER_MODE')}")
            adict['in_roi'] = [[], []]

            tr1 = tr2 = tr3 = tr4 = 0 #################

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
                    t5 = time.time()    ########
                    locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'), (scene.width / 2, scene.height))   # type: ignore
                    t6 = time.time()    ########
                    
                    

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
                    
                    t7 = time.time()    ########
                    
                    dets_in_roi = dets[in_roi_idxs]
                    locs_in_roi = locs[in_roi_idxs]
                    if (c == 0 and (not self.config.get('STA_PSEUDO_SAME_CAMERA') or self.homo is not None)) \
                        or (c == 1 and self.config.get('STA_PSEUDO_SAME_CAMERA') and self.homo is not None) :  # if cam 1, then transform to view of cam 2
                        locs = cv2.perspectiveTransform(locs.reshape(-1, 1, 2), self.homo)
                        locs = locs.reshape(-1, 2) if locs is not None else np.empty((0, 2))
                        
                        locs_in_roi = cv2.perspectiveTransform(locs_in_roi.reshape(-1, 1, 2), self.homo)
                        locs_in_roi = locs_in_roi.reshape(-1, 2) if locs_in_roi  is not None else np.empty((0, 2))
                    adict['in_roi'][c].append(dets_in_roi)
                    logging.debug(f'{self.name}:\t camera {c + 1} frame {adict["frame_id"][c][t]} found {len(dets_in_roi)}/{len(dets)} objects in ROI')

                    t8 = time.time()    ########
                    
                    # store location history, both inside-only and inide-outside
                    assert dets_in_roi.shape[1] in (10, 61, 9), 'expect track_id is of index 1' # ATTENTION: 9 is for output of CVAT only @@
                    self.history[-1][2][c].update({id: loc for id, loc in zip(np.int32(dets[:, 1]), locs.tolist())})                     # type: ignore
                    self.history[-1][3][c].update({id: loc for id, loc in zip(np.int32(dets_in_roi[:, 1]), locs_in_roi.tolist())})       # type: ignore

                    t9 = time.time()    ########

                    tr1 += (t6-t5)
                    tr2 += (t7-t6)
                    tr3 += (t8-t7)
                    tr4 += (t9-t8)

            
            t2 = time.time() ################
            
            ub = 2e9
            logging.debug(f"{self.name}:\t calculating cost with WINDOW_SIZE={self.config.get('DIST_WINDOW_SIZE')}, WINDOW_BOUNDARY={self.config.get('DIST_WINDOW_BOUNDARY')}")
            for iter_n in range(2 if self.config.get('FP_FILTER') and self.config.get('FP_REMAP') else 1):
                matches = []
                for t in range(T - self.config.get('DIST_WINDOW_BOUNDARY')):    # postponse the last BOUNDARY frames for the next iteration
                    
                    h1_ids = adict['in_roi'][0][t][:, 1]
                    h2_ids = adict['in_roi'][1][t][:, 1]

                    cost = np.empty((len(h1_ids), len(h2_ids)), dtype='float32')
                    gate = np.ones((len(h1_ids), len(h2_ids)), dtype='int32')
                    for i1, c1_id in enumerate(h1_ids):
                        for i2, c2_id in enumerate(h2_ids):
                            c1_locs, c2_locs = self._retrieve_window_locations(
                                c1_id, c2_id,
                                t - T,
                                self.config.get('DIST_WINDOW_SIZE'),
                                self.config.get('DIST_WINDOW_BOUNDARY')
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
                        logging.debug(f'{self.name}:\t pair frame_id ({adict["frame_id"][0][t]}, {adict["frame_id"][1][t]}) found {len(mi1s)} matches')
                    for i1, i2 in zip(mi1s, mi2s):  # type: ignore
                        h1 = h1_ids[i1]
                        h2 = h2_ids[i2]
                        dist = cost[i1, i2]
                        matches.append([t, h1, h2, dist])
                
                matches = np.array(matches).reshape(-1, 4)
                
                # FP filter
                d = self.distances + matches[:, 3].tolist()
                if self.config.get('FP_FILTER') and len(d) >= self.config.get('MIN_SAMPLE_SIZE_TO_FP_FILTER'):
                    filter = self._create_fp_filter()
                    ub = filter(np.array(d).reshape(-1, 1))             # type: ignore
                    matches = matches[matches[:, 3] <= ub]                  
                    
                    logging.debug(f"{self.name}:\t applied FP_FLITER={self.config.get('FP_FILTER')}")
            
            self.distances.extend(matches[:, 3].tolist())       # type: ignore
            
            for m in matches:               # type: ignore
                t = int(m[0].item())
                self.history[t - T][4].append(np.int32(m[1:3]).tolist())    # correct only because t in [0, T-1]

            self.history = self.history[:len(self.history) - self.config.get('DIST_WINDOW_BOUNDARY')]   # postponse the last BOUNDARY frames for the next iteration
            
            t3 = time.time() ################

            for his in self.history[-max(T - self.config.get('DIST_WINDOW_BOUNDARY'), 0):]:     # postponse the last BOUNDARY frames for the next iteration
                out_item = {
                    'frame_id_1': his[0],
                    'frame_id_2': his[1],
                    'locs': his[2],
                    'locs_in_roi': his[3],
                    'matches': his[4]
                }

                logging.debug(f'{self.name}:\t sleep')
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.config.get('SCT_TXT_SLEEP') - (end_time - start_time))
                time.sleep(sleep)    
                
                self._put_to_output_queues(out_item)

                start_time = end_time

            # logging.info(f'{self.name}:\t processing time t0={(t1-t0)/T:.4f}, t1={(t2-t1)/T:.4f}, t2={(t3-t2)/T:.4f}, tr1={tr1/T:.4f}, tr2={tr2/T:.4f}, tr3={tr3/T:.4f}, tr4={tr4/T:.4f}')  ################
            
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
        

    def _retrieve_window_locations(self, c1_id, c2_id, t, window_size, window_boundary) -> tuple:
        """Return a tuple of 2 np arrays for locations
        
        Arguments:
            t: index in the self.history (e.g -1 or -5)
            window_size: max size (frame unit) of the window, must be an odd number.
            window_boundary: boundary (frame unit) to ONE SIDE
        """
        assert window_size % 2 == 1, 'window size must be an odd number.'
        
        locs = [self.history[t][3][0][c1_id]], [self.history[t][3][1][c2_id]]
        
        ns = [0, 0]     # counter for left, right
        slices = [
            [self.history[j][3] for j in range(t - 1, t - 1 - window_boundary, -1)], 
            [self.history[j][3] for j in range(t + 1, t + window_boundary + 1)]
        ]

        for i, slice in enumerate(slices):
            for j in range(window_boundary):

                if j >= len(slice) or ns[i] >= (window_size - 1) // 2:
                    break

                if c1_id in slice[j][0] and c2_id in slice[j][1]:
                    locs[0].append(slice[j][0][c1_id])
                    locs[1].append(slice[j][1][c2_id])
                    ns[i] += 1
        logging.debug(f'{self.name}:\t window size for ID pairs {c1_id, c2_id}: nl, nr = {ns}')

        return np.array(locs[0]), np.array(locs[1])

                
    def _check_sct_queues(self):
        assert len(self.sct_queues) == 2, f'currently support only bipartite mapping, got {len(self.sct_queues)}'


    def _check_scenes(self):
        assert len(self.scenes) == len(self.sct_queues), 'the number of scenes must == number of SCT queues'


    def _new_wait_list(self):
        assert len(self.sct_queues) == 2, f'currently support only bipartite mapping, got {len(self.sct_queues)}'
        return [[], [], []]     # the last list is for matches


class Visualize(Pipeline):

    def __init__(
            self, 
            config: Config, 
            mode: str,
            annot_queue: MyQueue,
            video_queue: Union[MyQueue, None] = None,
            scene: Union[Scene, None] = None,
            homo: Union[np.ndarray, None] = None,
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            name='Visualizer'
    ) -> None:
        super().__init__(config, output_queues, name)
        
        self.mode = mode
        self._check_mode()

        self.annot_queue = annot_queue
        
        self.video_queue = video_queue
        self._check_video_queue()

        self.scene = scene
        self.homo = homo

        self.wait_list = self._new_wait_list()

        logging.info(f'{self.name}:\t initialized')

    
    def _start(self):

        while not self.is_stopped():

            self.trigger_pause()
            
            for wl, iq in zip(self.wait_list, [self.annot_queue, self.video_queue]):    # type: ignore
                wl.extend(
                    iq.get_many(                                                # type: ignore
                        size=self.config.get('QUEUE_GET_MANY_SIZE'),
                        block=self.config.get('QUEUE_GET_BLOCK'),
                        timeout=self.config.get('QUEUE_TIMEOUT')
                    )
                )

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

            # TODO Toc do visualize cham qua
            
            if self.mode == 'SCT':
            
                c1_adict_matched, c2_adict_matched = self._match_index(*adict['frame_id'])

                if len(c1_adict_matched) == 0:  # critical because there might be no matches (e.g SCT result comes slower), wo we need to wait more
                    self.wait_list = active_list
                    continue

                # add un-processed items to wait list
                self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
                self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])
                logging.debug(f'{self.name}:\t putting {len(self.wait_list[0])} of {self.annot_queue.name} to wait list')
                logging.debug(f'{self.name}:\t putting {len(self.wait_list[1])} of {self.video_queue.name} to wait list')   # type: ignore

                # using continuous indexes from 0 -> T-1 rather than discrete indexes
                T = len(c1_adict_matched)
                logging.debug(f'{self.name}:\t processing {T} pairs of frames')
                for k in adict:
                    if len(adict[k][0]) > 0:
                        adict[k][0] = [adict[k][0][idx] for idx in c1_adict_matched]
                    if len(adict[k][1]) > 0:
                        adict[k][1] = [adict[k][1][idx] for idx in c2_adict_matched]                   
                
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

                    if detection_mode == 'box':
                        frame_img = plot_box(frame_img, dets, self.config.get('VIS_SCT_BOX_THICKNESS'))
                    elif detection_mode == 'pose':
                        frame_img = plot_box(frame_img, dets, self.config.get('VIS_SCT_BOX_THICKNESS'))
                        for kpt in dets[:, 10:]:
                            frame_img = plot_skeleton_kpts(frame_img, kpt.T, 3)
                    else:
                        raise NotImplementedError()
                    
                    if self.scene is not None:
                        locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'))
                        frame_img = plot_loc(frame_img, np.concatenate([dets[:, :2], locs], axis=1), self.config.get('VIS_SCT_LOC_RADIUS'))

                    ##### START HERE #####
                    if 'signin_sid' in adict:
                        signin_sid = adict['signin_sid'][0][t]
                        if signin_sid is not None:
                            cv2.putText(frame_img, f'user<{signin_sid}> signed in', (300, 100), cv2.FONT_HERSHEY_TRIPLEX, 8.0, (0, 255, 0), thickness=3)
                    ##### END HERE #####
                    
                    out_item = {
                        'frame_img': frame_img,                   # type: ignore
                        'frame_id': adict['frame_id'][1][t]
                    }

                    self._put_to_output_queues(out_item, f"frame_id={adict['frame_id'][1][t]}")

            
            elif self.mode == 'STA':
                for k in adict:
                    adict[k] = adict[k][0]
                T = len(adict['frame_id_1'])
                                
                for t in range(T):
                    
                    # handle out-of-memory for exclusively offline
                    self.trigger_pause()
                    if self.is_stopped():
                        break
                                        
                    frame_id_1 = adict['frame_id_1'][t]
                    frame_id_2 = adict['frame_id_2'][t]
                    frame_img = np.zeros((self.scene.height, self.scene.width, 3), dtype='uint8')               # type: ignore
                    if self.config.get('STA_PSEUDO_SAME_CAMERA') and self.homo is not None:
                        roi = cv2.perspectiveTransform(np.float32(self.scene.roi), self.homo).astype('int32')   # type: ignore
                    else:
                        roi = self.scene.roi                # type: ignore
                    frame_img = plot_roi(frame_img, roi, self.config.get('VIS_ROI_THICKNESS'))       # type: ignore

                    cv2.putText(frame_img, f"frames={frame_id_1, frame_id_2}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=self.config.get('VIS_SCT_LOC_TEXTTHICKNESS'))

                    # plot points
                    for c in range(2):
                        for id, (x, y) in adict['locs'][t][c].items():
                            frame_img = plot_loc(
                                frame_img, 
                                [[-1, id, x, y]], 
                                self.config.get('VIS_SCT_LOC_RADIUS'), 
                                [f'{id} ({c})'], 
                                self.config.get('VIS_SCT_LOC_TEXTTHICKNESS')
                            )
                    
                    # plot matches
                    for (id1, id2) in adict['matches'][t]:
                        cv2.line(
                            frame_img,
                            np.int32(adict['locs'][t][0][id1]),
                            np.int32(adict['locs'][t][1][id2]),
                            color=(0, 255, 0),
                            thickness=self.config.get('VIS_STA_MATCH_THICKNESS')
                        )
                    
                    out_item = {
                        'frame_img': frame_img,                   # type: ignore
                        'frame_id': (frame_id_1, frame_id_2),
                    }
                    
                    # handle out-of-memory for exclusively offline
                    if self.config.get('RUNNING_MODE') == 'offline':
                        time.sleep(self.config.get('VIS_OFFLINE_PUT_SLEEP'))

                    self._put_to_output_queues(out_item, f'frame_ids={frame_id_1, frame_id_2}')

                if self.config.get('RUNNING_MODE') == 'offline':
                    self.stop()

            else:
                raise NotImplementedError()
            
    
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


    def _check_mode(self):
        assert self.mode in ['SCT', 'STA']
    

    def _check_video_queue(self):
        if self.mode == 'SCT':
            assert isinstance(self.video_queue, MyQueue), f'visualizing SCT requires video input queue, got {type(self.video_queue)}'
        elif self.mode == 'STA':
            assert self.video_queue is None, 'visualizing STA does not require video input queue'
            self.video_queue = MyQueue(maxsize=0, name='Mock-Video-Queue')


class Display(Pipeline):

    def __init__(
            self,
            config: Config,
            input_queue: MyQueue,
            width: Union[int, None] = None,
            height: Union[int, None] = None,
            fps: Union[int, None] = None,
            path: Union[str, None] = None,
            name='Display thread'
    ) -> None:
        super().__init__(config, None, name)
        
        self.input_queue = input_queue
        
        self.mode = self.config.get('DISPLAY_MODE')
        self.path = path
        self._check_mode()
        
        self.width = width
        self.height = height
        self.fps = fps
        self._check_fps()
        
        logging.info(f'{self.name}:\t initilized')
    
    def _start(self) -> None:

        if self.mode == 'show':
            self._setup_window()
        elif self.mode == 'save':
            writer_created = False

        while not self.is_stopped():

            self.trigger_pause()

            logging.debug(f"{self.name}:\t take {self.config.get('QUEUE_GET_MANY_SIZE')} from {self.input_queue.name}")
            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            for item in items:

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

                logging.debug(f'{self.name}:\t {self.mode} from {self.input_queue.name}\t frame_id={frame_id}')
                
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


class Export(Pipeline):

    def __init__(
            self,
            config: Config,
            input_queue: MyQueue,
            path: str, 
            name='Export'
    ) -> None:
        super().__init__(config, None, name)  # type: ignore

        self.input_queue = input_queue

        self.path = path
        self._check_path()

        logging.info(f'{self.name}:\t initialized')


    def _start(self) -> None:

        f = open(self.path, 'w')
        
        while not self.is_stopped():

            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            logging.debug(f'{self.name}:\t processing {len(items)} frames')

            for item in items:
                print(str(item), file=f)
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break
        
        f.close()

    
    def _check_path(self):
        parent, filename = os.path.split(self.path)
        if not os.path.exists(parent) and parent != '':
            logging.debug(f'{self.name}:\t create directory {parent}')
            os.makedirs(parent)


##### START HERE #####

class Staff:

    MAX_AGE = 30

    def __init__(self, sid, cid, tid):
        self.sid = sid
        
        self.cid = cid
        self.tid = tid
        self.age = -1
        self.mat_his = []

    
    def update(self, cur, mat):
        
        cid, tid = self.cid, self.tid
        
        if mat is None:
            if self.age >= 0:
                self.age += 1
        else:
            self.mat_his.append(mat)
            self.age = 0
        
        if self.age > Staff.MAX_AGE:
            if len(self.mat_his) > 0:
                freq = {}
                for i, m in enumerate(self.mat_his):
                    freq[m] = freq.get(m, 0) + i                # more weight on newer candidate
                self.cid, self.tid = max(freq, key=freq.get)    # type: ignore
                self.age = 0
                self.mat_his = []
        
        return cid, tid
            
            


class StaffMap(Pipeline):

    def __init__(
            self, 
            config: Config,
            sct_queues: dict,
            sta_queues: dict,
            checkin_scene: Scene,
            checkin_cid: int,
            output_queues: Union[List[MyQueue], MyQueue, None] = None,
            name='Staff Map'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.sct_queues = sct_queues
        self.sta_queues = sta_queues
        
        self.checkin_scene = checkin_scene
        self.checkin_cid = checkin_cid
        self._check_checkin_cid()

        self.staffs = []

        self.wait_list = self._new_wait_list()
        
    
    def _start(self) -> None:

        start_time = time.time()
        
        while True:# not self.is_stopped():

            self.trigger_pause()

            for i, q in enumerate([self.sct_queues, self.sta_queues]):
                for k, v in q.items():
                    if k not in self.wait_list[i]:
                        self.wait_list[i][k] = []
                    self.wait_list[i][k].extend(v.get_many(
                        size=self.config.get('QUEUE_GET_MANY_SIZE'),
                        block=self.config.get('QUEUE_GET_BLOCK'),
                        timeout=self.config.get('QUEUE_TIMEOUT')
                    ))

            active_list = self.wait_list
            self.wait_list = self._new_wait_list()

            sta_midxs = self._match_sta_index(active_list[1])
            sct_midxs, sta_midxs = self._match_sct_index(active_list[0], active_list[1], sta_midxs)

            if len(list(sta_midxs.values())[0]) == 0:
                self.wait_list = active_list
                continue

            T = len(list(sta_midxs.values())[0])

            for wl, al, idxs in zip(self.wait_list, active_list, [sct_midxs, sta_midxs]):   # type: ignore
                for k in al:
                    if k not in wl:
                        wl[k] = []
                    wl[k].extend(al[k][idxs[k][-1] + 1:])
                    al[k] = [al[k][idx] for idx in idxs[k]]

            for t in range(T):
                signin_sid = active_list[0][self.checkin_cid][t]['signin_sid']
                if signin_sid is not None:
                    dets = active_list[0][self.checkin_cid][t]['sct_output']
                    locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'), (self.checkin_scene.width / 2, self.checkin_scene.height)) # type: ignore
                    in_roi_idxs = self.checkin_scene.is_in_roi(locs)
                    dets_in_roi = dets[in_roi_idxs]
                    assert len(dets_in_roi) == 1
                    tid = int(dets_in_roi[0][1])
                    self.staffs.append(Staff(signin_sid, self.checkin_cid, tid))
                
                stf_pres = [(stf.cid, stf.tid) for stf in self.staffs]
                obj_curs = {(cid, int(d[1])): d
                            for cid, cv in active_list[0].items() 
                            for d in cv[t]['sct_output']}

                mat_curs = {}
                for (cid1, cid2), cv in active_list[1].items():
                    for tid1, tid2 in cv[t]['matches']:
                        mat_curs[(cid1, tid1)] = (cid2, tid2)
                        mat_curs[(cid2, tid2)] = (cid1, tid1)

                for cid, cv in active_list[0].items():
                    print(f'CID<{cid}>(frame_id={cv[t]["frame_id"]})', end='\t')
                print()
                
                for i, (cid, tid) in enumerate(stf_pres):
                    self.staffs[i].update(
                        obj_curs.get((cid, tid), None),
                        mat_curs.get((cid, tid), None)
                    )
                    print(f'Staff<{self.staffs[i].sid}>(cid={self.staffs[i].cid}, tid={self.staffs[i].tid})', end='\t')
                    
                print()

            
                end_time = time.time()
                sleep = 0 if self.config.get('RUNNING_MODE') == 'offline' \
                    else max(0, self.config.get('SCT_TXT_SLEEP') - (end_time - start_time)) + 0.01  # TODO fix this
                time.sleep(sleep)

                start_time = end_time

                






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
        sct_ks = []
        sct_idxs = []
        new_sta_idxs = {}
        for i, (pk, idxs) in enumerate(sta_idxs.items()):
            new_sta_idxs[pk] = []
            for cj, cid in enumerate(pk):
                if cid in sct_ks:
                    continue
                
                sct_ks.append(cid)
                sct_idxs.append([])

                write_sta = True if len(new_sta_idxs[pk]) == 0 else False

                j, q = 0, 0
                while j < len(sct_dict[cid]) and q < len(idxs):
                    sta_fid = sta_dict[pk][idxs[q]][f'frame_id_{cj + 1}']
                    sct_fid = sct_dict[cid][j]['frame_id']
                    if sta_fid == sct_fid:
                        sct_idxs[-1].append(j)
                        if write_sta:
                            new_sta_idxs[pk].append(idxs[q])
                        j += 1
                        q += 1
                    elif sta_fid < sct_fid:
                        q += 1
                    else:
                        j += 1

        return {k: v for k, v in zip(sct_ks, sct_idxs)}, new_sta_idxs       # type: ignore

    
    def _new_wait_list(self):
        return [{}, {}] # sct and sta



##### END HERE #####



def main(kwargs):

    config = Config(kwargs['config'])

    # input queues
    iq_sct_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-1-Input-Queue')
    iq_sct_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-2-Input-Queue')
    
    iq_sync_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-1-Input-Queue')
    iq_sync_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-2-Input-Queue')
    
    
    iq_sta_sct_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-1-InputSCT-Queue')
    iq_sta_sct_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-2-InputSCT-Queue')
    iq_sta_sync = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-InputSync-Queue')

    iq_exp_sta = MyQueue(config.get('QUEUE_MAXSIZE'), name='ExportSTA-Input-Queue')
    
    iq_vis_sct_annot_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-1-InputAnnot-Queue')
    iq_vis_sct_annot_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-2-InputAnnot-Queue')
    iq_vis_sct_vid_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-1-InputVideo-Queue')
    iq_vis_sct_vid_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-2-InputVideo-Queue')
    iq_vis_sta_annot = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSTA-InputAnnot-Queue')

    iq_dis_sct_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='DisSCT-1-Input-Queue')
    iq_dis_sct_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='DisSCT-2-Input-Queue')
    iq_dis_sta = MyQueue(config.get('QUEUE_MAXSIZE'), name='DisSTA-Input-Queue')

    # ONLY USE META IF CAPTURING VIDEOS
    meta_1 = yaml.safe_load(open(kwargs['meta_1'], 'r'))
    meta_2 = yaml.safe_load(open(kwargs['meta_2'], 'r'))
    pl_camera_1_retimg = Camera(config, kwargs['camera_1'], meta=meta_1, output_queues=[iq_vis_sct_vid_1], ret_img=True, name='Camera-1-RetImg')
    pl_camera_2_retimg = Camera(config, kwargs['camera_2'], meta=meta_2, output_queues=[iq_vis_sct_vid_2], ret_img=True, name='Camera-2-RetImg')
    pl_camera_1_noretimg = Camera(config, kwargs['camera_1'], meta=meta_1, output_queues=[iq_sct_1, iq_sync_1], ret_img=False, name='Camera-1-NoRetImg')
    pl_camera_2_noretimg = Camera(config, kwargs['camera_2'], meta=meta_2, output_queues=[iq_sct_2, iq_sync_2], ret_img=False, name='Camera-2-NoRetImg')
    
    tracker1 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_1'], name='Tracker-1')
    tracker2 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_2'], name='Tracker-2')
    pl_sct_1 = SCT(config, tracker=tracker1, input_queue=iq_sct_1, output_queues=[iq_sta_sct_1, iq_vis_sct_annot_1], name='SCT-1')
    pl_sct_2 = SCT(config, tracker=tracker2, input_queue=iq_sct_2, output_queues=[iq_sta_sct_2, iq_vis_sct_annot_2], name='SCT-2')
    
    pl_sync = SyncFrame(config, [iq_sync_1, iq_sync_2], iq_sta_sync)
    
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
    pl_sta = STA(config, [scene_1, scene_2], homo, [iq_sta_sct_1, iq_sta_sct_2], iq_sta_sync, [iq_vis_sta_annot, iq_exp_sta], name='STA')

    pl_exp = Export(config, iq_exp_sta, kwargs['out_sta_txt'])

    pl_vis_sct_1 = Visualize(config, mode='SCT', annot_queue=iq_vis_sct_annot_1, video_queue=iq_vis_sct_vid_1, scene=scene_1, output_queues=iq_dis_sct_1, name='VisSCT-1')
    pl_vis_sct_2 = Visualize(config, mode='SCT', annot_queue=iq_vis_sct_annot_2, video_queue=iq_vis_sct_vid_2, scene=scene_2, output_queues=iq_dis_sct_2, name='VisSCT-2')
    pl_vis_sta = Visualize(config, mode='STA', annot_queue=iq_vis_sta_annot, video_queue=None, scene=scene_2, homo=homo, output_queues=iq_dis_sta, name='VisSTA')

    pl_dis_sct_1 = Display(config, input_queue=iq_dis_sct_1, fps=pl_camera_1_retimg.fps, path=kwargs['out_sct_vid_1'], name='DisSCT-1') # type: ignore
    pl_dis_sct_2 = Display(config, input_queue=iq_dis_sct_2, fps=pl_camera_2_retimg.fps, path=kwargs['out_sct_vid_2'], name='DisSCT-2') # type: ignore
    pl_dis_sta = Display(config, input_queue=iq_dis_sta, fps=pl_camera_1_noretimg.fps, path=kwargs['out_sta_vid'], name='DisSTA')       # type: ignore

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

    # visualize and display
    # pl_camera_1_retimg.start()
    # pl_camera_2_retimg.start()

    # pl_vis_sct_1.start()
    # pl_vis_sct_2.start()
    # pl_vis_sta.start()

    # pl_dis_sct_1.start()
    # pl_dis_sct_2.start()
    # pl_dis_sta.start()

    if config.get('RUNNING_MODE') == 'online':
        pl_camera_1_noretimg.join()
        pl_camera_2_noretimg.join()
        pl_sct_1.join()
        pl_sct_2.join()
        pl_sync.join()
        pl_sta.join()
        pl_exp.join()


def main2(kwargs):

    config = Config(kwargs['config'])

    # input queues
    iq_sct_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-<1>-Input-Queue')
    iq_sct_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-<2>-Input-Queue')
    iq_sct_3 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-<3>-Input-Queue')
    
    iq_sync_12 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-<*1, 2>-Input-Queue')
    iq_sync_21 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-<1, *2>-Input-Queue')
    iq_sync_23 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-<*2, 3>-Input-Queue')
    iq_sync_32 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-<2, *3>-Input-Queue')
    
    
    iq_sta_sct_12 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<*1, 2>-InputSCT-Queue')
    iq_sta_sct_21 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<1, *2>-InputSCT-Queue')
    iq_sta_sync_12 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<1, 2>-InputSync-Queue')
    iq_sta_sct_23 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<*2, 3>-InputSCT-Queue')
    iq_sta_sct_32 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<2, *3>-InputSCT-Queue')
    iq_sta_sync_23 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-<2, 3>-InputSync-Queue')

    # ONLY USE META IF CAPTURING VIDEOS
    meta_1 = yaml.safe_load(open(kwargs['meta_1'], 'r'))
    meta_2 = yaml.safe_load(open(kwargs['meta_2'], 'r'))
    meta_3 = yaml.safe_load(open(kwargs['meta_3'], 'r'))

    pl_camera_1_noretimg = Camera(config, kwargs['camera_1'], meta=meta_1, output_queues=[iq_sct_1, iq_sync_12], ret_img=False, name='Camera-<1>-NoRetImg')
    pl_camera_2_noretimg = Camera(config, kwargs['camera_2'], meta=meta_2, output_queues=[iq_sct_2, iq_sync_21, iq_sync_23], ret_img=False, name='Camera-<2>-NoRetImg')
    pl_camera_3_noretimg = Camera(config, kwargs['camera_3'], meta=meta_3, output_queues=[iq_sct_3, iq_sync_32], ret_img=False, name='Camera-<3>-NoRetImg')
    
    tracker1 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_1'], name='Tracker-<1>')
    tracker2 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_2'], name='Tracker-<2>')
    tracker3 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path=kwargs['sct_3'], name='Tracker-<3>')
    pl_sct_1 = SCT(config, tracker=tracker1, input_queue=iq_sct_1, output_queues=[iq_sta_sct_12], name='SCT-<1>')
    pl_sct_2 = SCT(config, tracker=tracker2, input_queue=iq_sct_2, output_queues=[iq_sta_sct_21, iq_sta_sct_23], name='SCT-<2>')
    pl_sct_3 = SCT(config, tracker=tracker3, input_queue=iq_sct_3, output_queues=[iq_sta_sct_32], name='SCT-<3>')
    
    pl_sync_12 = SyncFrame(config, [iq_sync_12, iq_sync_21], iq_sta_sync_12)
    pl_sync_23 = SyncFrame(config, [iq_sync_23, iq_sync_32], iq_sta_sync_23)
    
    roi_21 = load_roi(kwargs['roi_21'], pl_camera_2_noretimg.width, pl_camera_2_noretimg.height)
    homo_12 = load_homo(kwargs['matches_12'])
    roi_12 = cv2.perspectiveTransform(roi_21, np.linalg.inv(homo_12)) # type: ignore
    roi_32 = load_roi(kwargs['roi_32'], pl_camera_3_noretimg.width, pl_camera_3_noretimg.height)
    homo_23 = load_homo(kwargs['matches_23'])
    roi_23 = cv2.perspectiveTransform(roi_32, np.linalg.inv(homo_23)) # type: ignore
    scene_12 = Scene(pl_camera_1_noretimg.width, pl_camera_1_noretimg.height, roi_12, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-<*1, 2>')
    scene_21 = Scene(pl_camera_2_noretimg.width, pl_camera_2_noretimg.height, roi_21, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-<1, *2>')
    scene_23 = Scene(pl_camera_2_noretimg.width, pl_camera_2_noretimg.height, roi_23, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-<*2, 3>')
    scene_32 = Scene(pl_camera_3_noretimg.width, pl_camera_3_noretimg.height, roi_32, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-<2, *3>')
    pl_sta_12 = STA(config, [scene_12, scene_21], homo_12, [iq_sta_sct_12, iq_sta_sct_21], iq_sta_sync_12, name='STA-<1, 2>')
    pl_sta_23 = STA(config, [scene_23, scene_32], homo_23, [iq_sta_sct_23, iq_sta_sct_32], iq_sta_sync_23, name='STA-<2, 3>')

    # start
    pl_camera_1_noretimg.start()
    pl_camera_2_noretimg.start()
    pl_camera_3_noretimg.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_camera_1_noretimg.join()     # offline
        pl_camera_2_noretimg.join()     # offline
        pl_camera_3_noretimg.join()     # offline
    
    pl_sct_1.start()
    pl_sct_2.start()
    pl_sct_3.start()
    pl_sync_12.start()
    pl_sync_23.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_sct_1.join()                 # offline
        pl_sct_2.join()                 # offline
        pl_sct_3.join()                 # offline
        pl_sync_12.join()                  # offline
        pl_sync_23.join()                  # offline

    pl_sta_12.start()
    pl_sta_23.start()
    if config.get('RUNNING_MODE') == 'offline':
        pl_sta_12.join()                   # offline
        pl_sta_23.join()                   # offline


    if config.get('RUNNING_MODE') == 'online':
        pl_camera_1_noretimg.join()
        pl_camera_2_noretimg.join()
        pl_camera_3_noretimg.join()
        pl_sct_1.join()
        pl_sct_2.join()
        pl_sct_3.join()
        pl_sync_12.join()
        pl_sync_23.join()
        pl_sta_12.join()
        pl_sta_23.join()
        



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
    
    # main(kwargs)

    kwargs2 = {
        'config': 'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/config_pred_mct_trackertracker_18.yaml',
        
        'meta_1': 'data/recordings/2d_v4/meta/41_00011_2023-04-15_08-30-00-000000.yaml',
        'meta_2': 'data/recordings/2d_v4/meta/42_00011_2023-04-15_08-30-00-000000.yaml',
        'meta_3': 'data/recordings/2d_v4/meta/43_00011_2023-04-15_08-30-00-000000.yaml',
        'camera_1': 'data/recordings/2d_v4/videos/41_00011_2023-04-15_08-30-00-000000.avi',
        'camera_2': 'data/recordings/2d_v4/videos/42_00011_2023-04-15_08-30-00-000000.avi',
        'camera_3': 'data/recordings/2d_v4/videos/43_00011_2023-04-15_08-30-00-000000.avi',
        
        'sct_1': 'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/41_00011_2023-04-15_08-30-00-000000.txt',
        'sct_2': 'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/42_00011_2023-04-15_08-30-00-000000.txt',
        'sct_3': 'data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/sct/43_00011_2023-04-15_08-30-00-000000.txt',

        'roi_21': 'data/recordings/2d_v4/roi_42.txt',
        'roi_32': 'data/recordings/2d_v4/roi_43.txt',
        'matches_12': 'data/recordings/2d_v4/matches_41_to_42.txt',
        'matches_23': 'data/recordings/2d_v4/matches_42_to_43.txt',

    }

    main2(kwargs2)



    
