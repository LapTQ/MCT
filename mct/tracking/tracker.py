from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
import yaml

from mct.utils.img_utils import iou_associate
from mct.tracking.kalmanbox import KalmanBoxBase, KalmanBox
from mct.utils.vid_utils import LoaderBase


HERE = Path(__file__).parent


class TrackerBase(ABC):

    @abstractmethod
    def update(self, dets: np.ndarray) -> np.ndarray:
        pass


class SORT(TrackerBase):

    class Builder:

        def __init__(self, cfg_path: str, loader: LoaderBase, kalmanbox_builder: KalmanBoxBase.Builder):
            self._reset()

            self._product.frame_count = 0
            self._product.objects = []  # temporarily observed Kalman objects, not "displayed objects"

            # setting from YAML
            with open(cfg_path, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            self._product.max_age = int(cfg['max_age'] * loader.get_fps())
            self._product.min_hits = int(cfg['min_hits'] * loader.get_fps())
            self._product.iou_threshold = cfg['iou_threshold']

            self._product.kalmanbox_builder = kalmanbox_builder

            print('[CFG] SORT max_age:', cfg['max_age'])
            print('[CFG] SORT min_hits:', cfg['min_hits'])
            print('[CFG] SORT iou_threshold:', cfg['iou_threshold'])

        def _reset(self) -> None:
            self._product = SORT()

        def get_product(self) -> KalmanBoxBase:
            product = self._product
            self._reset()
            return product

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
            self.objects.append(self.kalmanbox_builder.set_box(dets[d]).get_product())

        ret = []
        for i in range(len(self.objects) - 1, -1, -1):
            obj = self.objects[i]
            # TODO sao self.frame_count <= (thay vi <???? => mat frame 3)
            # TODO xem todo o KalmanBox, viec bat tat todo co anh huong rat lon den visualize @@
            if obj.age <= self.max_age and (self.frame_count <= self.min_hits or obj.hit_streak >= self.min_hits):
                ret.append(np.concatenate([[obj.id], obj.get_state()]))  # [id] + [x1, y1, x2, y2, conf]
            if obj.age > self.max_age:
                self.objects.pop(i)

        return np.array(ret).reshape(-1, 6)

