
from typing import Union
from queue import Queue
from threading import Lock, Thread
from datetime import datetime
import time
import cv2
import logging
import sys
import yaml


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
            logging.info(f'Loading config at {config_path}')

        self.stopped = False

        self.lock = Lock()
        
        logging.debug(f'Initilized {self.name}')

    
    def get(self, attr:str):
        return self.config[attr]


    def stop(self):
        self.lock.acquire()
        self.stopped = True
        logging.debug(f'config stop locked')
        self.lock.release()

    def is_stopped(self):
        return self.stopped is True


class MyQueue:

    def __init__(self, maxsize=0, name='Queue'):
        self.name = name
        self.maxsize = maxsize
        self.queue = Queue(maxsize)

        self.lock = Lock()

        logging.debug(f'Initilized {self.name}')
    
    def get(self, block=True, timeout=None):
        ret = self.queue.get(block, timeout)
        logging.debug(f'Dequeue an item from {self.name}. {self.name} is containing {self.queue.qsize()}')
        return ret
    
    def put(self, item, block=True, timeout=None):
        self.queue.put(item, block, timeout)
        logging.debug(f'Eequeue an item to {self.name}. {self.name} is containing {self.queue.qsize()}')

    def empty(self):
        return self.queue.empty
    
    def get_all(self) -> list:
        
        self.lock.acquire()
        
        old_queue = self.queue
        self.queue = Queue(self.maxsize)

        
        ret = []
        while not old_queue.empty():
            ret.append(old_queue.get())
        self.lock.release()

        #logging.debug(f'Dequeue all {old_queue.qsize()} items from {self.name}.')
        
        return ret


class Camera:

    def __init__(
            self,
            config:Config, 
            source,
            meta:Union[dict, None] = None, 
            output_queues:Union[list[MyQueue], MyQueue, None] = None,
            name='Camera thread'
    ) -> None:
        
        self.config = config
        
        self.name = name
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        self.meta = meta
        self._check_meta()
        
        self.output_queues = output_queues
        self._check_output_queues()

        self.thread = Thread(target=self._start, args=(), name=self.name)
        
        logging.debug(f'Initilized {self.name}')

    
    def _start(self) -> None:

        frame_id = 0

        while not self.is_stopped():

            t0 = time.time()

            if not self.cap.isOpened():
                logging.info(f'Camera thread {self.name} from {self.source} was disconnected')
                self.stop()
                break

            frame_id += 1
            frame_time = datetime.now()
            ret, frame = self.cap.read()

            if not ret:
                logging.info(f'{self.name} from {self.source} was disconnected')
                self.stop()
                break
            
            for queue in self.output_queues:
                queue.put(
                    {
                        'frame_img': frame,
                        'frame_id': frame_id,
                        'frame_time': frame_time
                    },
                    block=False
                )
            
                t1 = time.time()
                logging.debug(f'{self.name} put a frame with frame_id={frame_id}, frame_time={frame_time} from {self.source} to {queue.name} [{t1 - t0:.6f} seconds]')
        
        self.cap.release()

    
    def start(self) -> None:
        self.thread.start()

    def stop(self):
        self.config.stop()

    def is_stopped(self):
        return self.config.is_stopped()
        

    def _check_meta(self) -> None:
        pass
        
    
    def _check_output_queues(self) -> None:
        if self.output_queues is None:
            self.output_queues = []
        elif isinstance(self.output_queues, MyQueue):
            self.output_queues = [self.output_queues]


class Display:

    def __init__(
            self,
            config:Config,
            input_queue:MyQueue,
            name='Display thread'
    ) -> None:
        
        self.config = config
        
        self.name = name
        
        self.input_queue = input_queue

        self.thread = Thread(target=self._start, args=(), name=self.name)

        logging.debug(f'Initilized {self.name}')
    
    def _start(self):

        self._setup_window()

        while not self.is_stopped():

            # if self.input_queue.empty():
            #     continue

            items = self.input_queue.get_all()

            for item in items:

                frame_img = item['frame_img']
                frame_id = item['frame_id']
                frame_time = item['frame_time']

                logging.debug(f'{self.name} is displaying frame from {self.input_queue.name} with frame_id={frame_id} and frame_time={frame_time}')
                cv2.imshow(self.name, frame_img)
                key = cv2.waitKey(self.config.get('DISPLAY_FPS'))
                if key == ord('q'):
                    self.stop()
                    break
            
        cv2.destroyAllWindows()


    def start(self):
        self.thread.start()

    
    def stop(self):
        self.config.stop()

    def is_stopped(self):
        return self.config.is_stopped()

    
    def _setup_window(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)



"""
            SCT_queues:Union[list[MyQueue], MyQueue, None] = None,
            MCT_queues:Union[list[MyQueue], MyQueue, None] = None,
    def _check_camera_queues(self):
        raise NotImplementedError
    
    def _check_SCT_queues(self):
        raise NotImplementedError
    
    def _check_MCT_queues(self):
        raise NotImplementedError
"""



if __name__ == '__main__':

    config = Config('/media/tran/003D94E1B568C6D12/Workingspace/MCT/mct/utils/config.yaml')

    queue_1 = MyQueue()
    camera = Camera(config, '/media/tran/003D94E1B568C6D12/Workingspace/MCT/data/recordings/2d_v4/videos/41_00001_2023-04-05_08-30-00-000000.avi', output_queues=queue_1)
    display = Display(config, input_queue=queue_1)

    camera.start()
    display.start()

    
