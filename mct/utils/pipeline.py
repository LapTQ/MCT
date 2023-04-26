
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


logging.basicConfig(
    level=logging.DEBUG,
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
            logging.info(f'{self.name}:\t loading config at {config_path}')

        self.stopped = False

        self.lock = Lock()
        
        logging.debug(f'{self.name}:\t initilized')

    
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

        frame_id = 0

        while not self.is_stopped():

            t0 = time.time()

            if not self.cap.isOpened():
                logging.info(f'{self.name}:\t disconnected from {self.source}')
                self.stop()
                break

            frame_id += 1
            frame_time = datetime.now()
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

            logging.debug(f"{self.name}:\t sleep {self.config.get('CAMERA_SLEEP')}")
            time.sleep(self.config.get('CAMERA_SLEEP'))
            
        
        self.cap.release()
        

    def _check_meta(self) -> None:
        pass
        
    
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
        
        if txt_path is not None:
            self.seq = np.loadtxt(txt_path)
        
        self.name = name

        logging.info(f'{self.name}:\t initialized')
        
    
    def infer(self, img: Union[np.ndarray, None], frame_id: Union[int, None] = None) -> np.ndarray:

        if self.txt_path is not None:
            assert  isinstance(frame_id, int), 'frame_id must be provided in mock test'

            dets = self.seq[self.seq[:, 0] == frame_id]

            logging.debug(f'{self.name}:\t detected {len(dets)} people')

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
        
        while not self.config.is_stopped():

            t0 = time.time()

            logging.debug(f"{self.name}:\t take {self.config.get('QUEUE_GET_MANY_SIZE')} from {self.input_queue.name}")
            items = self.input_queue.get_many(
                size=self.config.get('QUEUE_GET_MANY_SIZE'),
                block=self.config.get('QUEUE_GET_BLOCK'),
                timeout=self.config.get('QUEUE_TIMEOUT')
            )

            for item in items:

                item['sct_output'] = self.tracker.infer(item['frame_img'], item['frame_id'])

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



if __name__ == '__main__':

    config = Config('/media/tran/003D94E1B568C6D12/Workingspace/MCT/mct/utils/config.yaml')

    queue_1 = MyQueue(config.get('QUEUE_MAXSIZE'), name='Display-Input-Queue')
    queue_2 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-1-Input-Queue')
    camera_1 = Camera(config, '/media/tran/003D94E1B568C6D12/Workingspace/MCT/data/recordings/2d_v4/videos/41_00001_2023-04-05_08-30-00-000000.avi', output_queues=[queue_1, queue_2], name='Camera 1')
    
    
    queue_3 = MyQueue(config.get('QUEUE_MAXSIZE'), name='SCT-1-Output-Queue')
    tracker = Tracker(txt_path='/media/tran/003D94E1B568C6D12/Workingspace/MCT/data/recordings/2d_v4/YOLOv8l_pretrained-640-ByteTrack/sct/41_00001_2023-04-05_08-30-00-000000.txt')
    sct_1 = SCT(config, tracker=tracker, input_queue=queue_2, output_queues=queue_3)
    display = Display(config, input_queue=queue_1)

    camera_1.start()
    display.start()
    sct_1.start()

    
