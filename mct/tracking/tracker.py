from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
import yaml

from mct.utils.img_utils import iou_associate
from mct.tracking.kalmanbox import KalmanBoxDirector, KalmanBoxStandardBuilder
from mct.utils.vid_utils import LoaderBase


HERE = Path(__file__).parent


class TrackerBase(ABC):

    @abstractmethod
    def update(self, dets: np.ndarray) -> np.ndarray:
        pass


class TrackerBuilder(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> TrackerBase:
        pass


class SORT(TrackerBase):

    def __init__(self) -> None:
        self.max_age = None
        self.min_hits = None
        self.iou_threshold = None

        self.frame_count = 0
        self.objects = []   # temporarily observed Kalman objects, not "displayed objects"

    def create_KalmanBox(self, box):
        director = KalmanBoxDirector()
        builder = KalmanBoxStandardBuilder()
        director.set_builder(builder)
        director.build_KalmanBoxStandard(box)
        return builder.get_product()

    # TODO thu xoa default dets=np.empty di -> cho lam output cua detection
    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """
        dets: [[x1, y1, x2, y2, conf],...]
        Return [[frame, id, x1, y1, x2, y2], ...]
        """
        self.frame_count += 1

        # get PREDICTED boxes from existing Kalman objects, [[x1, y1, x2, y2, 0],...]>
        preds = []
        for t in range(len(self.objects) - 1, -1, -1):
            # TODO predict()[0]?
            pos = self.objects[t].predict()  # [x1, y1, x2, y2]

            if np.any(np.isnan(pos)):
                self.objects.pop(t)
            else:
                # TODO preds.append(pos) if pos is [x1, y1, x2, y2, conf]?
                preds.append([pos[0], pos[1], pos[2], pos[3], 0])
        preds.reverse()
        preds = np.array(preds).reshape(-1, 5)

        # matched = [[dets_index, preds_index],...], unmatched_dets = [index,...], unmatched_preds = [index,...]
        matched, unmatched_dets, unmatched_preds = iou_associate(dets, preds, self.iou_threshold)

        for m in matched:
            # TODO check the use of conf
            self.objects[m[1]].update(dets[m[0]])

        for d in unmatched_dets:
            self.objects.append(self.create_KalmanBox(dets[d]))

        ret = []
        for i in range(len(self.objects) - 1, -1, -1):
            obj = self.objects[i]
            # TODO sao self.frame_count <= (thay vi <???? => mat frame 3)
            if obj.age <= self.max_age and (self.frame_count <= self.min_hits or obj.hit_streak >= self.min_hits):
                ret.append(np.concatenate([[obj.id], obj.get_state()]))  # [id] + [x1, y1, x2, y2, conf]
            if obj.age > self.max_age:
                self.objects.pop(i)

        return np.array(ret).reshape(-1, 6)


class SORTBuilder(TrackerBuilder):

    def __init__(self, loader: LoaderBase) -> None:
        self._tracker = None

        with open(HERE/'../configs/sort.yaml', 'r') as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.loader = loader

        self.reset()

    def reset(self) -> None:
        self._tracker = SORT()

    def set_max_age(self) -> None:
        self._tracker.max_age = int(self._cfg['max_age'] * self.loader.get_fps())
        print('[CFG] SORT max_age:', self._cfg['max_age'])

    def set_min_hits(self) -> None:
        self._tracker.min_hits = int(self._cfg['min_hits'] * self.loader.get_fps())
        print('[CFG] SORT min_hits:', self._cfg['min_hits'])

    def set_iou_threshold(self) -> None:
        self._tracker.iou_threshold = self._cfg['iou_threshold']
        print('[CFG] SORT iou_threshold:', self._cfg['iou_threshold'])

    def get_product(self) -> TrackerBase:
        product = self._tracker
        self.reset()
        return product


class TrackerDirector:

    def __init__(self) -> None:
        self._builder = None

    def set_builder(self, builder: TrackerBuilder) -> None:
        self._builder = builder

    def build_SORT(self) -> None:
        self._builder.reset()
        self._builder.set_max_age()
        self._builder.set_min_hits()
        self._builder.set_iou_threshold()


if __name__ == '__main__':

    # TODO danh gia ket qua tung modun
    pass
