import os
from pathlib import Path
import yaml

import numpy as np

from tracking.utils.img_utils import iou_associate, xyxy2xysr, xysr2xyxy
from filterpy.kalman import KalmanFilter

HERE = Path(__file__).parent


class KalmanBox:

    count = 0

    def __init__(self, box):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # TODO conf
        self.kf.x[:4] = xyxy2xysr(box[:4]).reshape(4, 1)

        KalmanBox.count += 1
        self.id = KalmanBox.count

        self.age = 0
        self.hit_streak = 0
        self.history = []

        self.load_config()

    def load_config(self):
        with open(HERE/'models'/'kalman'/'config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        # TODO tune (QUAN TRONG)
        self.kf.F = np.array(config['F'], dtype='float32')
        self.kf.H = np.array(config['H'], dtype='float32')
        self.kf.P = np.array(config['P'], dtype='float32')
        self.kf.Q = np.array(config['Q'], dtype='float32')
        self.kf.R = np.array(config['R'], dtype='float32')

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

    def get_state(self):
        """
        return [x1, y1, x2, y2]
        """
        # TODO return conf?
        return xysr2xyxy(self.kf.x[:4].reshape(4,))


class SORT:

    def __init__(self):
        self.max_age = None
        self.min_hits = None
        self.iou_threshold = None

        self.load_config()

        # TODO use frame from input
        self.frame_count = 0
        self.objects = []   # temporarily observed Kalman objects, not "displayed objects"

    def load_config(self):
        with open(HERE/'models'/'sort'/'config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        print('[CFG] SORT max_age:', config['max_age'])
        print('[CFG] SORT min_hits', config['min_hits'])
        print('[CFG] SORT iou_threshold:', config['iou_threshold'])

        self.max_age = config['max_age']
        self.min_hits = config['min_hits']
        self.iou_threshold = config['iou_threshold']

    # TODO thu xoa default dets=np.empty di -> cho lam output cua detection
    def update(self, dets=np.empty((0, 5))):
        """
        dets: [[x1, y1, x2, y2, conf],...]
        Return [[frame, id, x1, y1, x2, y2], ...] if ret is True
        """

        self.frame_count += 1

        # preds = <get PREDICTED boxes from existing Kalman objects, [[x1, y1, x2, y2, 0],...]>
        # <remove NaN predicted-positions boxes>
        preds = []
        for t in range(len(self.objects) - 1, -1, -1):
            # TODO predict()[0]?
            pos = self.objects[t].predict() # [x1, y1, x2, y2]

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
            self.objects.append(KalmanBox(dets[d]))

        ret = []
        for i in range(len(self.objects) - 1, -1, -1):
            obj = self.objects[i]
            # TODO age <= self.max_age? (ban dau <=1)
            # TODO sao self.frame_count <= (thay vi <???? => mat frame 3)
            if obj.age <= self.max_age and (self.frame_count <= self.min_hits or obj.hit_streak >= self.min_hits):
                ret.append(np.concatenate([[obj.id], obj.get_state()]))     # [id] + [x1, y1, x2, y2, (conf?)]
            if obj.age > self.max_age:
                self.objects.pop(i)

        return np.array(ret).reshape(-1, 5)


if __name__ == '__main__':
    seq_dets = np.loadtxt('../output/dets.txt', delimiter=',')

    # TODO tham so
    tracker = SORT()
    out_file = open('../output/ests.txt', 'w')

    for frame in range(int(seq_dets[:, 0].max())):
        frame += 1

        dets = seq_dets[seq_dets[:, 0] == frame][:, 2:7]    # [x1, y1, w, h, conf]
        dets[:, 2:4] = dets[:, :2] + dets[:, 2:4]   # [x1, y1, x2, y2, conf]

        # TODO return frame
        ests = tracker.update(dets)     # [id, x1, y1, x2, y2]

        for e in ests:
            # TODO confidence score???
            print("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (frame, e[0], e[1], e[2], e[3] - e[1], e[4] - e[2]), file=out_file)




