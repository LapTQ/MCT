from abc import ABC, abstractmethod
from mct.detection.detector import DetectorBase, DetectorDirector, YOLOv5Builder
from mct.tracking.tracker import TrackerBase, TrackerDirector, SORTBuilder
from mct.utils.vid_utils import LoaderBase


class SCTBase(ABC):

    # TODO type checking
    @abstractmethod
    def create_detector(self) -> DetectorBase:
        pass

    @abstractmethod
    def create_tracker(self, loader: LoaderBase) -> TrackerBase:
        pass


class SCT(SCTBase):
    """YOLOv5 + SORT"""

    def create_detector(self) -> DetectorBase:
        director = DetectorDirector()
        builder = YOLOv5Builder()
        director.set_builder(builder)
        director.build_YOLOv5()
        return builder.get_product()

    def create_tracker(self, loader: LoaderBase) -> TrackerBase:
        director = TrackerDirector()
        builder = SORTBuilder(loader)
        director.set_builder(builder)
        director.build_SORT()
        return builder.get_product()

