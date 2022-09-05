from abc import ABC, abstractmethod

import numpy as np
from filterpy.kalman import KalmanFilter
from pathlib import Path
import yaml

from mct.utils.img_utils import xyxy2xysr, xysr2xyxy


HERE = Path(__file__).parent


class KalmanBoxBase(ABC):

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, box: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass


class KalmanBoxBuilder(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_product(self):
        pass


class KalmanBoxStandard(KalmanBoxBase):

    count = 0

    def __init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        KalmanBoxStandard.count += 1
        self.id = KalmanBoxStandard.count
        self.conf = None

        self.age = 0
        self.hit_streak = 0
        self.history = []

    def predict(self):
        """
        return [x1, y1, x2, y2]
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        # TODO tune dieu kien nay (QUAN TRONG)
        # if self.age > 0:
        #     self.hit_streak = 0         # cần thiết vì để đảm bảo cả min_hits frame đầu tiên đều được detect
        self.age += 1
        self.history.append(xysr2xyxy(self.kf.x[:4].reshape(4,)))
        return self.history[-1]

    def update(self, box):
        # box: [x1, y1, x2, y2, conf]
        # TODO handle default box = np.empty((0, 5))
        self.age = 0
        self.hit_streak += 1
        # TODO check the use of conf in box
        # TODO check history
        self.history = []
        self.kf.update(xyxy2xysr(box[:4]).reshape(4, 1))
        self.conf = box[4]

    def get_state(self):
        """
        return [x1, y1, x2, y2, conf]
        """
        return np.concatenate([xysr2xyxy(self.kf.x[:4].reshape(4,)), [self.conf]], axis=0)


class KalmanBoxStandardBuilder(KalmanBoxBuilder):

    def __init__(self):
        self._kalmanbox = None

        with open(HERE/'../configs/kalmanboxstandard.yaml', 'r') as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.reset()

    def reset(self) -> None:
        self._kalmanbox = KalmanBoxStandard()

    def get_product(self) -> KalmanBoxBase:
        product = self._kalmanbox
        self.reset()
        return product

    def set_initial_z(self, box) -> None:
        # [id, x1, y1, x2, y2, conf]
        self._kalmanbox.kf.x[:4] = xyxy2xysr(box[:4]).reshape(4, 1)
        self._kalmanbox.conf = box[4].item()

    def set_F(self) -> None:
        self._kalmanbox.kf.F = np.array(self._cfg['F'], dtype='float32')

    def set_H(self) -> None:
        self._kalmanbox.kf.H = np.array(self._cfg['H'], dtype='float32')

    def set_P(self) -> None:
        self._kalmanbox.kf.P = np.array(self._cfg['P'], dtype='float32')

    def set_Q(self) -> None:
        self._kalmanbox.kf.Q = np.array(self._cfg['Q'], dtype='float32')

    def set_R(self) -> None:
        self._kalmanbox.kf.R = np.array(self._cfg['R'], dtype='float32')


class KalmanBoxDirector:

    def __init__(self):
        self._builder = None

    def set_builder(self, builder: KalmanBoxBuilder) -> None:
        self._builder = builder

    def build_KalmanBoxStandard(self, box: np.ndarray):
        self._builder.reset()
        self._builder.set_initial_z(box)
        self._builder.set_F()
        self._builder.set_H()
        self._builder.set_P()
        self._builder.set_Q()
        self._builder.set_R()


