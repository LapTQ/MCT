from pathlib import Path

from abc import ABC, abstractmethod
from mct.detection.detector import DetectorBase, YOLOv5
from mct.tracking.tracker import TrackerBase, SORT
from mct.tracking.kalmanbox import KalmanBox
from mct.utils.vid_utils import LoaderBase


HERE = Path(__file__).parent

class SCTBase(ABC):

    # TODO type checking
    @abstractmethod
    def create_detector(self) -> DetectorBase:
        pass

    @abstractmethod
    def create_tracker(self, loader: LoaderBase) -> TrackerBase:
        pass


class SimpleSCT(SCTBase):
    """YOLOv5 + SORT"""

    def create_detector(self) -> DetectorBase:
        detector = YOLOv5.Builder(HERE / './configs/yolov5s.yaml').get_product()
        return detector

    def create_tracker(self, loader: LoaderBase) -> TrackerBase:
        kalmanbox_builder = KalmanBox.Builder(HERE / './configs/kalmanboxstandard.yaml')
        tracker = SORT.Builder(HERE/'./configs/sort.yaml', loader, kalmanbox_builder).get_product()
        return tracker

