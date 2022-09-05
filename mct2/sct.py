from abc import ABC, abstractmethod
from mct2.detection.detector import DetectorBase, YOLOv5, DetectorDirector, YOLOv5Builder
from mct2.tracking.tracker import TrackerBase, SORT, TrackerDirector, SORTBuilder


class SCTBase(ABC):

    # TODO type checking
    @abstractmethod
    def create_detector(self) -> DetectorBase:
        pass

    @abstractmethod
    def create_tracker(self) -> TrackerBase:
        pass


class SCT(SCTBase):
    """YOLOv5 + SORT"""

    def create_detector(self) -> YOLOv5:
        director = DetectorDirector()
        builder = YOLOv5Builder()
        director.set_builder(builder)
        director.build_YOLOv5()
        return builder.get_product()

    def create_tracker(self) -> SORT:
        director = TrackerDirector()
        builder = SORTBuilder()
        director.set_builder(builder)
        director.build_SORT()
        return builder.get_product()

