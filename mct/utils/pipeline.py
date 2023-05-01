from typing import Any, Union
from queue import Queue
from threading import Lock, Thread
from datetime import datetime
import time
import cv2
import logging
import sys
import yaml
from abc import ABC, abstractmethod
import numpy as np

from general import load_roi, load_homo, calc_loc
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

        self.stopped = False
        self.pausing = False

        self.lock = Lock()
        
        logging.info(f'{self.name}:\t initilized')

    
    def get(self, attr:str):
        return self.config[attr]


    def stop(self) -> None:
        self.lock.acquire()
        self.stopped = True
        logging.info(f'{self.name}\t stop locked')
        self.lock.release()


    def is_stopped(self) -> bool:
        return self.stopped

    
    def switch_pausing(self) -> None:
        self.lock.acquire()
        self.pausing = not self.pausing
        logging.info(f'{self.name}\t pausing switched')
        self.lock.release()

    def is_pausing(self) -> bool:
        return self.pausing


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
            output_queues: Union[list[MyQueue], MyQueue, None] = None, 
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


    def switch_pausing(self):
        self.config.switch_pausing()

    
    def _put_to_output_queues(self, item, msg=None):
        for oq in self.output_queues:                    # type: ignore
            oq.put(
                item,
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )
        
            logging.info(f"{self.name}:\t put to {oq.name} with message: {msg}")


class Camera(Pipeline):

    def __init__(
            self,
            config: Config, 
            source: Union[int, str],
            meta: Union[dict, None] = None, 
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            name='Camera thread'
    ) -> None:
        super().__init__(config, output_queues, name)
        
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        self.meta = meta
        self._check_meta()

        if self.config.get('CAMERA_FRAME_WIDTH') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('CAMERA_FRAME_WIDTH'))
        if self.config.get('CAMERA_FRAME_HEIGHT') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('CAMERA_FRAME_HEIGHT'))
        if self.config.get('CAMERA_FPS') is not None:
            assert self.meta is None, 'camera_fps must not be set when capturing videos from disk'
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('CAMERA_FPS'))
        
        logging.info(f'{self.name}:\t initilized')

    
    def _start(self) -> None:

        if self.meta is None:
            frame_id = 0
        else:   # if reading from video on disk
            frame_id = self.meta['start_frame_id'] - 1

        while not self.is_stopped():

            self.trigger_pause()

            if not self.cap.isOpened():
                logging.info(f'{self.name}:\t problem connecting to {self.source}')
                self.stop()
                break

            if self.config.get('RUNNING_MODE') != 'offline':
                ret, frame = self.cap.read()
            else:
                ret, frame = True, None
                if frame_id == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    ret = False

            if not ret:
                logging.info(f'{self.name}:\t disconnected from {self.source}')
                if self.config.get('RUNNING_MODE') != 'offline':
                    self.stop()
                break
            
            frame_id += 1

            if self.meta is None:
                frame_time = datetime.now().timestamp()
            else:   # if reading from video on disk
                frame_time = datetime.strptime(self.meta['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (frame_id - self.meta['start_frame_id']) / self.fps
            
            out_item = {
                'frame_img': frame,
                'frame_id': frame_id,
                'frame_time': frame_time
            }
            
            self._put_to_output_queues(out_item, f"frame_id={frame_id}, frame_time={datetime.fromtimestamp(frame_time).strftime('%Y-%m-%d_%H-%M-%S-%f')}")

            # if reading from video on disk, then sleep according to fps to sync time.
            sleep = self.config.get('CAMERA_SLEEP') + (0 if self.meta is None else 0.01 / self.fps)
            logging.debug(f"{self.name}:\t sleep {sleep}")
            time.sleep(sleep)
        
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
            return int(self.meta['fps'])
        else:
            return int(self.cap.get(cv2.CAP_PROP_FPS))


    def _check_meta(self) -> None:
        pass
        

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
            self.seq = np.loadtxt(txt_path)
            logging.info(f'{self.name}:\t load tracking result from .txt at {txt_path}')
        
        self.name = name

        logging.info(f'{self.name}:\t initialized with detection_mode={self.detection_mode}, tracking_mode={self.tracking_mode}')
        
    
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


class SCT(Pipeline):

    def __init__(
            self, 
            config: Config, 
            tracker: Tracker,
            input_queue: MyQueue,
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            name='SCT'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.tracker = tracker
        
        self.input_queue = input_queue

        logging.info(f'{self.name}:\t initialized')

    
    def _start(self) -> None:
        
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
                    'sct_tracking_mode': self.tracker.tracking_mode
                }

                self._put_to_output_queues(out_item, f"frame_id={item['frame_id']}, sct shape={dets.shape}")
            
            logging.debug(f"{self.name}:\t sleep {self.config.get('SCT_TXT_SLEEP')}")
            time.sleep(self.config.get('SCT_TXT_SLEEP'))

            if self.config.get('RUNNING_MODE') == 'offline':
                break


class SyncFrame(Pipeline):

    def __init__(
            self, 
            config: Config, 
            input_queues: list[MyQueue],
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            name='SyncFrame'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.input_queues = input_queues
        self._check_input_queues()

        self.wait_list = self._new_wait_list()

        logging.debug(f'{self.name}:\t initialized')

    
    def _start(self) -> None:

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
                logging.info(f'{self.name}:\t processing {len(c1_adict_matched)} pairs of frames')
                self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
                self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[0])} of {self.input_queues[0].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[1])} of {self.input_queues[1].name} to wait list')

            for i, j in zip(c1_adict_matched, c2_adict_matched):

                logging.debug(f"{self.name}:\t sync frame_id={adict['frame_id'][0][i]} ({datetime.fromtimestamp(adict['frame_time'][0][i]).strftime('%M-%S-%f')}) with frame_id={adict['frame_id'][1][j]} ({datetime.fromtimestamp(adict['frame_time'][1][j]).strftime('%M-%S-%f')})")

                out_item = {
                    'frame_id_match': (adict['frame_id'][0][i], adict['frame_id'][1][j])
                }

                self._put_to_output_queues(out_item, f'frame_ids={out_item["frame_id_match"]}')
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break
    

    def _check_input_queues(self):
        assert len(self.input_queues) == 2, f'currently support only bipartite mapping, got {len(self.input_queues)}'


    def _new_wait_list(self):
        return [[] for _ in range(len(self.input_queues))]


class Display(Pipeline):

    def __init__(
            self,
            config: Config,
            input_queue: MyQueue,
            name='Display thread'
    ) -> None:
        super().__init__(config, None, name)
        
        self.input_queue = input_queue

        logging.info(f'{self.name}:\t initilized')
    
    def _start(self) -> None:

        self._setup_window()

        spf = 0

        while not self.is_stopped():

            self.trigger_pause()

            t0 = time.time()

            logging.debug(f"{self.name}:\t take {self.config.get('QUEUE_GET_MANY_SIZE')} from {self.input_queue.name}")
            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            for item in items:

                frame_img = item['frame_img']
                frame_id = item['frame_id']

                logging.debug(f'{self.name}:\t display from {self.input_queue.name}\t frame_id={frame_id}')
                cv2.imshow(self.name, frame_img)

                key = cv2.waitKey(self.config.get('DISPLAY_FPS'))
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord(' '):                    
                    self.switch_pausing()
                    cv2.waitKey(0)
                    self.switch_pausing()
            
            if len(items) > 0:
                spf = spf * 0.5 + (time.time() - t0) / len(items) * 0.5
                logging.debug(f'{self.name}:\t FPS = {1/spf:.3f}')
            
        cv2.destroyAllWindows()

    
    def _setup_window(self) -> None:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)


class StackFrames(Pipeline):

    def __init__(
            self, 
            config: Config, 
            input_queues: list[MyQueue],
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            shape: Union[list[int], tuple[int], None] = None,
            name='StackFrames'
    ) -> None:
        super().__init__(config, output_queues, name)

        self.input_queues = input_queues
        self._check_input_queues()

        self.shape = shape
        self._check_shape()

        logging.info(f'{self.name}:\t initialized')

    
    def _start(self):

        while not self.is_stopped():

            self.trigger_pause()

            t0 = time.time()

            items_2D = []
            for queue in self.input_queues:
                items_2D.append(queue.get_many(
                    size=self.config.get('QUEUE_GET_MANY_SIZE'),
                    block=self.config.get('QUEUE_GET_BLOCK'),
                    timeout=self.config.get('QUEUE_TIMEOUT')
                ))

            adict = {
                'frame_img': [],
                'frame_id': [],
                'frame_time': []
            }
            for items in items_2D:
                for k in adict:
                    adict[k].append([])
                for k in adict:
                    for item in items:
                        adict[k][-1].append(item[k])
            # each array in frame_time, frame_id should be in increasing order of time

            # TODO THIS PART MIGHT NEED IMPROVEMENT (slowness of map_timestamp, take the 1st queue as benchmark)
            # #### take the first queue's result as benchmark
            maps = [map_mono(adict['frame_time'][0], adict['frame_time'][i], return_matrix=False) 
                    for i in range(1, len(adict['frame_time']))]
            dict_map = {}
            for j, (idx0_ls, idxi_ls) in enumerate(maps):
                for idx0, idxi in zip(idx0_ls, idxi_ls):
                    if idx0 not in dict_map:
                        dict_map[idx0] = [idx0]
                    if j + 1 > len(dict_map[idx0]):
                        dict_map[idx0].append(None)
                    dict_map[idx0].append(idxi)
            maps = sorted(list(dict_map.values()))
            # maps should be in increasing order of time
            ###############################################

            for map in maps:
                out_item = {
                    'frame_img': [(adict['frame_img'][j][idx] if idx is not None else None) for j, idx in enumerate(map)],
                    'frame_id': [adict['frame_id'][j][idx] for j, idx in enumerate(map)],
                    'frame_time': [adict['frame_time'][j][idx] for j, idx in enumerate(map)]
                }

                frame_img = out_item['frame_img']
                ncols, nrows = self.shape                   # type: ignore
                frame_img.extend([None for _ in range(ncols * nrows - len(frame_img))])
                H, W = frame_img[0].shape[:2]
                collage = np.empty((H * nrows, W * ncols, 3), dtype='uint8')
                for i, img in enumerate(frame_img):
                    if img is None:
                        img = np.zeros((H, W), dtype='uint8')
                    elif i > 0:
                        img = cv2.resize(img, (W, H))
                    col, row = i % ncols, i // ncols
                    collage[H * row: H * (row + 1), W * col: W * (col + 1), :] = img
                out_item['frame_img'] = collage

                self._put_to_output_queues(out_item)
            
            if self.config.get('RUNNING_MODE') == 'offline':
                break


    def _check_input_queues(self):
        assert len(self.input_queues) > 1, 'input_queues must have at least 2 elements'

    
    def _check_shape(self):
        n_queues = len(self.input_queues)
        if self.shape is not None:
            assert isinstance(self.shape, (list, tuple)), 'shape must be list or tuple'
            assert len(self.shape) == 2, 'shape must have 2 elements for width and height'
            assert self.shape[0] * self.shape[1] == n_queues, 'shape[0] * shape[1] must equal number of input queues'
        else:
            import math
            ncols, nrows = 1, n_queues
            while True:
                
                if n_queues % ncols == 0:
                    nrows = n_queues // ncols
                
                if ncols >= nrows:
                    break

                ncols += 1
            self.shape = (ncols, nrows)
            logging.debug(f'{self.name}:\t calculate grid shape = {self.shape}')


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


class STA(Pipeline):

    def __init__(
            self, 
            config: Config, 
            scenes: list[Scene],
            homo: Union[np.ndarray, None],
            sct_queues: list[MyQueue],
            sync_queue: MyQueue,
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
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
        
        while not self.is_stopped():

            self.trigger_pause()

            t0 = time.time()
            
            for wl, iq in zip(self.wait_list, self.sct_queues + [self.sync_queue]):
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
            # TODO thong ke toc do => realtime + sleep sao cho memory usage on dinh

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

            # add un-processed items to wait list
            self.wait_list[0].extend(active_list[0][c1_adict_matched[-1] + 1:])
            self.wait_list[1].extend(active_list[1][c2_adict_matched[-1] + 1:])
            self.wait_list[2].extend(active_list[2][sync_adict_matched[-1] + 1:])
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[0])} of {self.sct_queues[0].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[1])} of {self.sct_queues[1].name} to wait list')
            logging.debug(f'{self.name}:\t putting {len(self.wait_list[2])} of {self.sync_queue.name} to wait list')

            # using continuous indexes from 0 -> T-1 rather than discrete indexes
            T = len(c1_adict_matched)
            logging.info(f'{self.name}:\t processing {T} pairs of frames')
            del adict['frame_id_match']
            for k in adict:
                adict[k][0] = [adict[k][0][idx] for idx in c1_adict_matched]
                adict[k][1] = [adict[k][1][idx] for idx in c2_adict_matched]

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
                    locs = calc_loc(dets, self.config.get('LOC_INFER_MODE'))
                    
                    # filter in ROI
                    in_roi_idxs = scene.is_in_roi(locs)
                    dets_in_roi = dets[in_roi_idxs]
                    locs_in_roi = locs[in_roi_idxs]
                    if c == 0 and homo is not None:  # if cam 1, then transform to view of cam 2
                        locs = cv2.perspectiveTransform(locs.reshape(-1, 1, 2), homo)
                        locs = locs.reshape(-1, 2) if locs is not None else np.empty((0, 2))
                        
                        locs_in_roi = cv2.perspectiveTransform(locs_in_roi.reshape(-1, 1, 2), homo)
                        locs_in_roi = locs_in_roi.reshape(-1, 2) if locs_in_roi  is not None else np.empty((0, 2))
                    adict['in_roi'][c].append(dets_in_roi)
                    logging.info(f'{self.name}:\t camera {c + 1} frame {adict["frame_id"][c][t]} found {len(dets_in_roi)}/{len(dets)} objects in ROI')

                    # store location history, both inside-only and inide-outside
                    assert dets_in_roi.shape[1] in (10, 61), 'expect track_id is of index 1'
                    self.history[-1][2][c].update({id: loc for id, loc in zip(np.int32(dets[:, 1]), locs)})                     # type: ignore
                    self.history[-1][3][c].update({id: loc for id, loc in zip(np.int32(dets_in_roi[:, 1]), locs_in_roi)})       # type: ignore

            matches = []
            for t in range(T):
                
                h1_ids = adict['in_roi'][0][t][:, 1]
                h2_ids = adict['in_roi'][1][t][:, 1]

                cost = np.empty((len(h1_ids), len(h2_ids)), dtype='float32')
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
                
                mi1s, mi2s = hungarian(cost)
                if len(mi1s) > 0:
                    logging.info(f'{self.name}:\t pair frame_id ({adict["frame_id"][0][t]}, {adict["frame_id"][1][t]}) found {len(mi1s)} matches')
                for i1, i2 in zip(mi1s, mi2s):
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
            
            self.distances.extend(matches[:, 3].tolist())

            for m in matches:
                t = int(m[0].item())
                self.history[t - T][4].append(m[1:3])    # correct only because t in [0, T-1]

            for his in self.history[-T:]:
                out_item = {
                    'frame_id_1': his[0],
                    'frame_id_2': his[1],
                    'locs': his[2],
                    'locs_in_roi': his[3],
                    'matches': his[4]
                }
                
                self._put_to_output_queues(out_item, f'frame_ids={his[0], his[1]}, #matches={len(his[4])}')

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
        
        locs = [self.history[t][2][0][c1_id]], [self.history[t][2][1][c2_id]]
        
        ns = [0, 0]     # counter for left, right
        slices = [
            [self.history[j][2] for j in range(t - 1, t - 1 - window_boundary, -1)], 
            [self.history[j][2] for j in range(t + 1, t + window_boundary + 1)]
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
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            name='Visualizer'
    ) -> None:
        super().__init__(config, output_queues, name)
        
        self.mode = mode
        self._check_mode()

        self.annot_queue = annot_queue
        
        self.video_queue = video_queue
        self._check_video_queue()

        self.scene = scene

        self.wait_list = self._new_wait_list()

        logging.debug(f'{self.name}:\t initialized')

    
    def _start(self):

        while not self.is_stopped():

            self.trigger_pause()

            t0 = time.time()
            
            for wl, iq in zip(self.wait_list, [self.annot_queue, self.video_queue]):
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
                for j, item in enumerate(al):
                    for k, v in item.items():
                        if k not in adict:
                            adict[k] = [[], []]
                        adict[k][i].append(v)
            
            if len(adict) == 0:
                continue
            
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

                    
                    out_item = {
                        'frame_img': frame_img,                   # type: ignore
                        'frame_id': adict['frame_id'][1][t]
                    }

                    self._put_to_output_queues(out_item, f"frame_id={adict['frame_id'][1][t]}")
            
            elif self.mode == 'STA':
                for k in adict:
                    adict[k] = adict[k][0]
                T = len(adict['frame_id_1'])
                
                t5 = 0
                
                for t in range(T):
                    
                    t6 = time.time()
                    
                    frame_id_1 = adict['frame_id_1'][t]
                    frame_id_2 = adict['frame_id_2'][t]
                    frame_img = np.zeros((self.scene.height, self.scene.width, 3), dtype='uint8')              # type: ignore
                    frame_img = plot_roi(frame_img, self.scene.roi, self.config.get('VIS_ROI_THICKNESS'))   # type: ignore

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
                    
                    t5 = 0.5 * t5 + 0.5 * (time.time() - t6)

                    self._put_to_output_queues(out_item, f'frame_ids={frame_id_1, frame_id_2}')
            
                logging.info(f't5 = {t5}')
            
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


if __name__ == '__main__':

    config = Config('/media/tran/003D94E1B568C6D11/Workingspace/MCT/mct/utils/config.yaml')

    queue_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Display-1-Input-Queue')
    queue_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-1-Input-Queue')
    queue_4 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Display-2-Input-Queue')
    queue_5 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-2-Input-Queue')
    queue_9 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-1-Input-Queue')
    queue_10 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Sync-2-Input-Queue')
    queue_12 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-1-InputVideo-Queue')
    queue_13 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-2-InputVideo-Queue')
    
    # ONLY USE META IF CAPTURING VIDEOS
    meta_1 = yaml.safe_load(open('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/meta/41_00007_2023-04-11_08-30-00-000000.yaml', 'r'))
    meta_2 = yaml.safe_load(open('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/meta/42_00007_2023-04-11_08-30-00-000000.yaml', 'r'))
    camera_1 = Camera(config, '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/videos/41_00007_2023-04-11_08-30-00-000000.avi', meta=meta_1, output_queues=[queue_2, queue_9], name='Camera-1')
    camera_2 = Camera(config, '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/videos/42_00007_2023-04-11_08-30-00-000000.avi', meta=meta_2, output_queues=[queue_5, queue_10], name='Camera-2')
    
    queue_11 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-InputSync-Queue')
    sync = SyncFrame(config, [queue_9, queue_10], queue_11)
    queue_3 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-1-InputSCT-Queue')
    queue_6 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-2-InputSCT-Queue')
    queue_14 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-1-InputAnnot-Queue')
    queue_15 = MyQueue(config.get('QUEUE_MAXSIZE'), name='VisSCT-2-InputAnnot-Queue')
    tracker1 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path='/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv7pose-pretrained-640-ByteTrack/sct/41_00007_2023-04-11_08-30-00-000000.txt', name='Tracker-1')
    tracker2 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path='/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv7pose-pretrained-640-ByteTrack/sct/42_00007_2023-04-11_08-30-00-000000.txt', name='Tracker-2')
    sct_1 = SCT(config, tracker=tracker1, input_queue=queue_2, output_queues=[queue_3])
    sct_2 = SCT(config, tracker=tracker2, input_queue=queue_5, output_queues=[queue_6])
    # queue_7 = MyQueue(config.get('QUEUE_MAXSIZE'), name='StackFrames-Output-Queue')
    # stackframes = StackFrames(config, input_queues=[queue_1, queue_4], output_queues=queue_7, shape=(2,1))

    queue_8 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-Output-Queue')
    roi_2 = load_roi('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/roi_42.txt', camera_2.width, camera_2.height)
    homo = load_homo('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/matches_41_to_42.txt')
    roi_1 = cv2.perspectiveTransform(roi_2, np.linalg.inv(homo))
    scene_1 = Scene(camera_1.width, camera_1.height, roi_1, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-1')
    scene_2 = Scene(camera_2.width, camera_2.height, roi_2, config.get('ROI_TEST_OFFSET'), name='Scene-Cam-2')
    sta = STA(config, [scene_1, scene_2], homo, [queue_3, queue_6], queue_11, queue_8)

    vis_sct_1 = Visualize(config, mode='SCT', annot_queue=queue_14, video_queue=queue_12, scene=scene_1, output_queues=queue_1)
    vis_sct_2 = Visualize(config, mode='SCT', annot_queue=queue_15, video_queue=queue_13, scene=scene_2, output_queues=queue_4)
    
    # queue_16 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Display-Input-Queue')
    # vis_sta = Visualize(config, mode='STA', annot_queue=queue_8, video_queue=None, scene=scene_2, output_queues=queue_16)

    # display_1 = Display(config, input_queue=queue_16, name='Display 1')
    # display_2 = Display(config, input_queue=queue_4, name='Display 2')

    camera_1.start()
    camera_2.start()

    camera_1.join()
    camera_2.join()
    
    sync.start()
    sct_1.start()
    sct_2.start()

    sync.join()
    sct_1.join()
    sct_2.join()

    sta.start()

    # vis_sct_1.start()
    # vis_sct_2.start()
    # vis_sta.start()
    
    # display_1.start()
    # display_2.start()

    config.stop()
    
