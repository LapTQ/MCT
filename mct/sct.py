from pathlib import Path
from typing import Union
import numpy as np

from abc import ABC, abstractmethod
from mct.detection.detector import DetectorBase, YOLOv5
from mct.tracking.tracker import TrackerBase, SORT
from mct.tracking.kalmanbox import KalmanBox
from mct.utils.vid_utils import LoaderBase


HERE = Path(__file__).parent

class SCTBase(ABC):

    # TODO type checking
    @abstractmethod
    def predict(self, frame: np.ndarray, BGR: bool) -> np.ndarray:
        pass


class SimpleSCT(SCTBase):
    """YOLOv5 + SORT"""

    class Builder:

        def __init__(self, loader: LoaderBase) -> None:
            self._reset()

            self._product.detector = YOLOv5.Builder(str(HERE / './configs/yolov5s.yaml')).get_product()

            kalmanbox_builder = KalmanBox.Builder(str(HERE / './configs/kalmanboxstandard.yaml'))
            self._product.tracker = SORT.Builder(str(HERE / './configs/sort.yaml'), loader,
                                                 kalmanbox_builder).get_product()

            self._product.frame_count = 0

        def _reset(self):
            self._product = SimpleSCT()

        def get_product(self) -> SCTBase:
            product = self._product
            self._reset()
            return product

    def predict(self, frame: np.ndarray, BGR: bool) -> np.ndarray:

        dets = self.detector.predict(frame, BGR=BGR)  # [[x1, y1, x2, y2, conf], ...]

        # TODO refactor: adapter
        # TODO tại sao trong code của kalman không dùng tới conf của dets? kiểm tra lại thông số của kalman xem có liên quan không
        tracklets = self.tracker.update(dets)  # [[id, x1, y1, x2, y2, conf]...]

        ret = np.concatenate([np.array([self.frame_count] * len(tracklets)).reshape(-1, 1), tracklets],
                             axis=1)  # [[frame, id, x1, y1, x2, y2, conf]...]

        self.frame_count += 1

        return ret



