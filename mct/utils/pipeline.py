
from typing import Union
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

from map_utils import map_timestamp


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(levelname)s\t|%(funcName)s\t|%(lineno)d\t|%(message)s',
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
            logging.debug(f'{self.name}:\t loading config at {config_path}')

        self.stopped = False

        self.lock = Lock()
        
        logging.info(f'{self.name}:\t initilized')

    
    def get(self, attr:str):
        return self.config[attr]


    def stop(self) -> None:
        self.lock.acquire()
        self.stopped = True
        logging.debug(f'{self.name}\t stop locked')
        self.lock.release()

    def is_stopped(self) -> bool:
        return self.stopped is True


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
            name='Pipeline component'
    ) -> None:
        
        self.config = config
        self.name = name
        self.thread = Thread(target=self._start, args=(), name=self.name)    
    
    @abstractmethod
    def _start(self) -> None:
        pass
    
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.config.stop()

    def is_stopped(self) -> bool:
        return self.config.is_stopped() is True


class Camera(Pipeline):

    def __init__(
            self,
            config: Config, 
            source: Union[int, str],
            meta: Union[dict, None] = None, 
            output_queues: Union[list[MyQueue], MyQueue, None] = None,
            name='Camera thread'
    ) -> None:
        super().__init__(config, name)
        
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        self.meta = meta
        self._check_meta()
        
        self.output_queues = output_queues
        self._check_output_queues()
        
        logging.debug(f'Initilized {self.name}')

    
    def _start(self) -> None:

        if self.config.get('CAMERA_FRAME_WIDTH') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('CAMERA_FRAME_WIDTH'))
        if self.config.get('CAMERA_FRAME_HEIGHT') is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('CAMERA_FRAME_HEIGHT'))
        if self.meta is None and self.config.get('CAMERA_FPS') is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('CAMERA_FPS'))


        if self.meta is None:
            frame_id = 0
        else:   # if reading from video on disk
            frame_id = self.meta['start_frame_id'] - 1

        while not self.is_stopped():

            t0 = time.time()

            if not self.cap.isOpened():
                logging.info(f'{self.name}:\t disconnected from {self.source}')
                self.stop()
                break

            frame_id += 1

            if self.meta is None:
                frame_time = datetime.now().timestamp()
            else:   # if reading from video on disk
                frame_time = datetime.strptime(self.meta['start_time'], '%Y-%m-%d_%H-%M-%S-%f').timestamp() + (frame_id - self.meta['start_frame_id']) / self.meta['fps']
            
            ret, frame = self.cap.read()

            if not ret:
                logging.info(f'{self.name}:\t disconnected from {self.source}')
                self.stop()
                break
            
            for queue in self.output_queues:
                queue.put(
                    {
                        'frame_img': frame,
                        'frame_id': frame_id,
                        'frame_time': frame_time
                    },
                    block=self.config.get('QUEUE_GET_BLOCK'),
                    timeout=self.config.get('QUEUE_TIMEOUT')
                )
            
                t1 = time.time()
                logging.debug(f'{self.name}:\t put to {queue.name}\t frame_id={frame_id}, frame_time={frame_time} from {self.source} [{t1 - t0:.6f} seconds]')

            # if reading from video on disk, then sleep according to fps to sync time.
            sleep = self.config.get('CAMERA_SLEEP') if self.meta is None else 0.01 / self.meta['fps']
            logging.debug(f"{self.name}:\t sleep {sleep}")
            time.sleep(sleep)
        
        self.cap.release()
        

    def _check_meta(self) -> None:
        pass
        
    
    def _check_output_queues(self) -> None:
        if self.output_queues is None:
            self.output_queues = []
        elif isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]


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
        
        # if using offline mock tracking result
        if txt_path is not None:
            self.seq = np.loadtxt(txt_path)
            logging.debug(f'{self.name}:\t load tracking result from .txt at {txt_path}')
        
        self.name = name

        logging.debug(f'{self.name}:\t initialized with detection_mode={self.detection_mode}, tracking_mode={self.tracking_mode}')
        
    
    def infer(self, img: Union[np.ndarray, None], frame_id: Union[int, None] = None) -> np.ndarray:

        if self.txt_path is not None:
            assert  isinstance(frame_id, int), 'frame_id must be provided in mock test'

            # if detection_mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
            # if detection_mode == 'box' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
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
        super().__init__(config, name)

        self.tracker = tracker
        
        self.input_queue = input_queue
        self.output_queues = output_queues
        self._check_output_queues()

    
    def _start(self) -> None:
        
        while not self.is_stopped():

            t0 = time.time()

            logging.debug(f"{self.name}:\t take {self.config.get('QUEUE_GET_MANY_SIZE')} from {self.input_queue.name}")
            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            for item in items:

                item['sct_output'] = self.tracker.infer(item['frame_img'], item['frame_id'])
                item['sct_detection_mode'] = self.tracker.detection_mode
                item['sct_tracking_mode'] = self.tracker.tracking_mode

                for queue in self.output_queues:
                    queue.put(
                    item,
                    block=self.config.get('QUEUE_GET_BLOCK'),
                    timeout=self.config.get('QUEUE_TIMEOUT')
                )
            
                    t1 = time.time()
                    logging.debug(f'{self.name}:\t put SCT result from {self.input_queue.name} to {queue.name}\t frame_id={item["frame_id"]}, frame_time={item["frame_time"]} [{t1 - t0:.6f} seconds]')
            
            logging.debug(f"{self.name}:\t sleep {self.config.get('SCT_TXT_SLEEP')}")
            time.sleep(self.config.get('SCT_TXT_SLEEP'))


    def _check_output_queues(self) -> None:
        if self.output_queues is None:
            self.output_queues = []
        elif isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]


class Display(Pipeline):

    def __init__(
            self,
            config: Config,
            input_queue: MyQueue,
            name='Display thread'
    ) -> None:
        super().__init__(config, name)
        
        self.input_queue = input_queue

        logging.debug(f'{self.name}:\t initilized')
    
    def _start(self) -> None:

        self._setup_window()

        while not self.is_stopped():

            logging.debug(f"{self.name}:\t take {self.config.get('QUEUE_GET_MANY_SIZE')} from {self.input_queue.name}")
            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            for item in items:

                frame_img = item['frame_img']
                frame_id = item['frame_id']
                frame_time = item['frame_time']

                logging.debug(f'{self.name}:\t display from {self.input_queue.name}\t frame_id={frame_id} and frame_time={frame_time}')
                cv2.imshow(self.name, frame_img)
                key = cv2.waitKey(self.config.get('DISPLAY_FPS'))
                if key == ord('q'):
                    self.stop()
                    break
            
        cv2.destroyAllWindows()

    
    def _setup_window(self) -> None:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)


class StackFrames(Pipeline):

    def __init__(
            self, 
            config: Config, 
            input_queues: list[MyQueue],
            output_queues: Union[list[MyQueue], MyQueue],
            shape: Union[list[int], tuple[int], None] = None,
            name='StackFrames'
    ) -> None:
        super().__init__(config, name)

        self.input_queues = input_queues
        self._check_input_queues()
        self.output_queues = output_queues
        self._check_output_queues()

        self.shape = shape
        self._check_shape()

        logging.info(f'{self.name}:\t initialized')

    
    def _start(self):

        while not self.is_stopped():

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
            maps = [map_timestamp(adict['frame_time'][0], adict['frame_time'][i], return_matrix=False) 
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
                ncols, nrows = self.shape
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

                for output_queue in self.output_queues:
                    output_queue.put(
                    out_item,
                    block=self.config.get('QUEUE_GET_BLOCK'),
                    timeout=self.config.get('QUEUE_TIMEOUT')
                )
            
                    t1 = time.time()
                    logging.debug(f'{self.name}:\t put merged frames from {", ".join([input_queue.name for input_queue in self.input_queues])} to {output_queue.name}\t frame_id={out_item["frame_id"]}, frame_time={out_item["frame_time"]} [{t1 - t0:.6f} seconds]')


    def _check_input_queues(self):
        assert len(self.input_queues) > 1, 'input_queues must have at least 2 elements'

    
    def _check_output_queues(self):
        if isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]

    
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


class 


class STA(Pipeline):

    def __init__(
            self, 
            config: Config, 
            input_queues: list[MyQueue],
            output_queues: Union[list[MyQueue], MyQueue],
            name='STA'
    ) -> None:
        super().__init__(config, name)

        self.input_queues = input_queues
        self._check_input_queues()
        
        self.output_queues = output_queues
        self._check_output_queues()

        # [(frame_id_1, frame_time_1, frame_id_2, frame_time_2, [(id_1, id_2, distance), ...]), ...]
        self.matches = []

        logging.debug(f'{self.name}:\t initialized')


    def _start(self) -> None:
        
        while not self.is_stopped():

            t0 = time.time()

            cam1_items, cam2_items = [
                input_queue.get_many(
                    size=self.config.get('QUEUE_GET_MANY_SIZE'),
                    block=self.config.get('QUEUE_GET_BLOCK'),
                    timeout=self.config.get('QUEUE_TIMEOUT')
                )
                for input_queue in self.input_queues
            ]

            



    
    def _check_input_queues(self):
        assert len(self.input_queues) == 2, f'currently support only bipartite mapping, got {len(self.input_queues)}'


    def _check_output_queues(self):
        if isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]






if __name__ == '__main__':

    config = Config('/media/tran/003D94E1B568C6D11/Workingspace/MCT/mct/utils/config.yaml')

    # queue_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='StackFrames-1-Input-Queue')
    queue_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-1-Input-Queue')
    # queue_4 = MyQueue(config.get('QUEUE_MAXSIZE'), name='StackFrames-2-Input-Queue')
    queue_5 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-2-Input-Queue')
    
    # ONLY USE META IF CAPTURING VIDEOS
    meta_1 = yaml.safe_load(open('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/meta/41_00001_2023-04-05_08-30-00-000000.yaml', 'r'))
    meta_2 = yaml.safe_load(open('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/meta/42_00001_2023-04-05_08-30-00-000000.yaml', 'r'))
    camera_1 = Camera(config, '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/videos/41_00001_2023-04-05_08-30-00-000000.avi', meta=meta_1, output_queues=[queue_2], name='Camera 1')
    camera_2 = Camera(config, '/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/videos/42_00001_2023-04-05_08-30-00-000000.avi', meta=meta_2, output_queues=[queue_5], name='Camera 2')
    
    
    queue_3 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-1-Input-Queue')
    queue_6 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-2-Input-Queue')
    tracker1 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path='/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv8l_pretrained-640-ByteTrack/sct/41_00001_2023-04-05_08-30-00-000000.txt', name='Tracker 1')
    tracker2 = Tracker(detection_mode=config.get('DETECTION_MODE'), tracking_mode=config.get('TRACKING_MODE'), txt_path='/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/recordings/2d_v4/YOLOv8l_pretrained-640-ByteTrack/sct/42_00001_2023-04-05_08-30-00-000000.txt', name='Tracker 2')
    sct_1 = SCT(config, tracker=tracker1, input_queue=queue_2, output_queues=queue_3)
    sct_2 = SCT(config, tracker=tracker2, input_queue=queue_5, output_queues=queue_6)
    # queue_7 = MyQueue(config.get('QUEUE_MAXSIZE'), name='StackFrames-Output-Queue')
    # stackframes = StackFrames(config, input_queues=[queue_1, queue_4], output_queues=queue_7, shape=(2,1))
    # display_1 = Display(config, input_queue=queue_1, name='Display 1')
    # display_2 = Display(config, input_queue=queue_4, name='Display 2')

    queue_8 = MyQueue(config.get('QUEUE_MAXSIZE'), name='STA-Output-Queue')
    sta = STA(config, [queue_3, queue_6], queue_8)

    camera_1.start()
    camera_2.start()
    # stackframes.start()
    # display_1.start()
    # display_2.start()
    sct_1.start()
    sct_2.start()

    sta.start()

    
