from typing import Union
from threading import Lock, Thread
import logging
import yaml
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


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

    
    def _put_to_output_queues(self, item):
        for k, oq in self.output_queues.items():
            logger.debug(f"{self.name}:\t put to queue {k}, previous size: {oq.qsize()}")
            oq.put(item, block=True)
        

    def add_output_queue(self, queue, key):
        self.lock.acquire()
        assert key not in self.output_queues, f'{key} already exists in output_queues'
        self.output_queues[key] = queue
        self.lock.release()
        
        logger.info(f'{self.name}:\t added output queue with key {key}. Having {len(self.output_queues)} output queues.')   # type: ignore

    
    def remove_output_queue(self, key):
        self.lock.acquire()
        q = self.output_queues[key]
        del self.output_queues[key]
        self.lock.release()

        logger.info(f'{self.name}:\t removed output queue with key {key}. Remaining {len(self.output_queues)} output queues.')  # type: ignore
        
        return q

