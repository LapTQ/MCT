from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
import yaml

import sys
sys.path.append(sys.path[0] + '/../..')

from mct.utils.img_utils import iou_associate
from mct.tracking.kalmanbox import KalmanBoxBase
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
            self._product.id_count = 0
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

        def get_product(self) -> TrackerBase:
            product = self._product
            self._reset()
            return product

    # TODO thu xoa default dets=np.empty di -> cho lam output cua detection
    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """
        dets: [[x1, y1, KalmanBoxBasex2, y2, conf],...]
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
            self.id_count += 1
            self.objects.append(self.kalmanbox_builder.set_box(dets[d]).set_id(self.id_count).get_product())

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


if __name__ == '__main__':
    import sys

    sys.path.append(sys.path[0] + '/../..')

    from mct.utils.vid_utils import ImageFolderLoader
    from mct.tracking.kalmanbox import KalmanBox
    from tqdm import tqdm
    import numpy as np
    import os

    kalmanbox_builder = KalmanBox.Builder(HERE / '../configs/kalmanboxstandard.yaml')

    root = str(HERE/'../../output/det')
    out_dir = str(HERE / '../../eval/TrackEval/data/trackers/mot_challenge/MOT17-train/SCT/data')
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(root):
        basename = os.path.splitext(filename)[0]
        loader = ImageFolderLoader.Builder(os.path.join(str(HERE/'../../data/MOT17/train'), basename, 'img1'),
                                           os.path.join(str(HERE/'../../data/MOT17/train'), basename, 'seqinfo.ini')).get_product()

        output_path = out_dir + '/' + basename + '.txt'
        txt_buffer = []
        out_txt = open(output_path, 'w')

        tracker = SORT.Builder(str(HERE / '../configs/sort.yaml'), loader, kalmanbox_builder).get_product()

        dets_seq = np.loadtxt(os.path.join(root, filename), delimiter=',')
        for frame_count in tqdm(range(int(dets_seq[:, 0].max()))):
            frame_count += 1

            dets = dets_seq[dets_seq[:, 0] == frame_count][:, 1:] ## [[x1, y1, x2, y2, conf], ...]
            tracklets = tracker.update(dets)  # [[id, x1, y1, x2, y2, conf]...]

            ret = np.concatenate([np.array([frame_count] * len(tracklets)).reshape(-1, 1), tracklets],
                                 axis=1)  # [[frame, id, x1, y1, x2, y2, conf]...]
            for obj in ret:
                # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
                txt_buffer.append(
                    f'{int(obj[0])}, {int(obj[1])}, {obj[2]:.2f}, {obj[3]:.2f}, {(obj[4] - obj[2]):.2f}, {(obj[5] - obj[3]):.2f}, {obj[6]:.6f}, -1, -1, -1')

        loader.release()

        print('\n'.join(txt_buffer), file=out_txt)
        print('[INFO] MOT17-format .txt saved in', output_path)
