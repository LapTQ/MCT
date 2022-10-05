from abc import ABC, abstractmethod

import numpy as np
from filterpy.kalman import KalmanFilter
from pathlib import Path
import yaml

from mct.utils.img_utils import xyxy2xysr, xysr2xyxy


HERE = Path(__file__).parent


class KalmanBoxBase(ABC):

    class Builder(ABC):

        @abstractmethod
        def __init__(self):
            pass

        @abstractmethod
        def _reset(self):
            pass

        @abstractmethod
        def get_product(self):
            pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, box: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass


class KalmanBox(KalmanBoxBase):

    count = 0

    class Builder(KalmanBoxBase.Builder):

        def __init__(self, cfg_path:str):
            """Construct from YAML"""

            # setting from YAML
            with open(cfg_path, 'r') as f:
                self._cfg = yaml.load(f, Loader=yaml.FullLoader)

            self._reset()


        def set_box(self, box):
            """box: [x1, y1, x2, y2, conf]"""
            self._product.kf.x[:4] = xyxy2xysr(box[:4]).reshape(4, 1)
            self._product.conf = box[4].item()

            return self

        def _reset(self) -> None:
            self._product = KalmanBox()

            KalmanBox.count += 1
            self._product.id = KalmanBox.count

            self._product.age = 0
            self._product.hit_streak = 0
            self._product.history = []

            self._product.kf = KalmanFilter(dim_x=7, dim_z=4)
            self._product.kf.F = np.array(self._cfg['F'], dtype='float32')
            self._product.kf.H = np.array(self._cfg['H'], dtype='float32')
            self._product.kf.P = np.array(self._cfg['P'], dtype='float32')
            self._product.kf.Q = np.array(self._cfg['Q'], dtype='float32')
            self._product.kf.R = np.array(self._cfg['R'], dtype='float32')

        def get_product(self) -> KalmanBoxBase:
            product = self._product
            self._reset()
            return product

    def predict(self):
        """return [x1, y1, x2, y2]"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        # TODO bat/tat dieu kien nay (QUAN TRONG)
        if self.age > 0:
            self.hit_streak = 0         # cần thiết vì để đảm bảo cả min_hits frame đầu tiên đều được detect
        self.age += 1
        self.history.append(xysr2xyxy(self.kf.x[:4].reshape(4,)))
        return self.history[-1]

    def update(self, box):
        """box: [x1, y1, x2, y2, conf]"""
        # TODO handle default box = np.empty((0, 5))
        self.age = 0
        self.hit_streak += 1
        # TODO check the use of conf in box
        # TODO check history
        self.history = []
        self.kf.update(xyxy2xysr(box[:4]).reshape(4, 1))
        self.conf = box[4]

    def get_state(self):
        """return [x1, y1, x2, y2, conf]"""
        return np.concatenate([xysr2xyxy(self.kf.x[:4].reshape(4,)), [self.conf]], axis=0)


if __name__ == '__main__':

    kalmanbox_builder = KalmanBox.Builder('../configs/kalmanboxstandard.yaml')
    kb = kalmanbox_builder.set_box(np.array([1, 2, 3, 4, 5])).get_product()
    kb.update([3, 4, 5, 6, 7])
    print(kb.conf)








