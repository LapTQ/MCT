import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import yaml

import torch


HERE = Path(__file__).parent


class DetectorBase(ABC):

    @abstractmethod
    def predict(self, img: np.ndarray, BGR: bool) -> np.ndarray:
        pass


class DetectorBuilder(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def set_model(self) -> None:
        pass

    @abstractmethod
    def set_input_size(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> DetectorBase:
        pass


class YOLOv5(DetectorBase):

    def __init__(self) -> None:
        self.model = None
        self.input_size = None


    def predict(self, img: np.ndarray, BGR: bool) -> np.ndarray:
        """
        return [[x1, y1, x2, y2, conf], ...]
        """
        if BGR:
            img = img[:, :, ::-1]

        preds = self.model([img], size=self.input_size)  # RGB, include NMS
        ret = preds.xyxy[0][:, :5].numpy()  # preds.xyxy = [[[x1, y1, x2, y2, conf, class_id], ...]]
        return ret


class YOLOv5Builder(DetectorBuilder):

    def __init__(self):
        self._detector = None

        with open(HERE/'../configs/yolov5s.yaml', 'r') as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.reset()

    def reset(self) -> None:
        self._detector = YOLOv5()

    # TODO config loader
    def set_model(self) -> None:
        assert os.path.exists(HERE/self._cfg['weights']), 'No such file or directory'
        print('[CFG] YOLOv5 model:', HERE/self._cfg['weights'])
        self._detector.model = torch.hub.load('ultralytics/yolov5', 'custom', path=HERE/self._cfg['weights'])

        print('[CFG] YOLOv5 iou threshold:', self._cfg['iou'])
        self._detector.model.iou = self._cfg['iou']

        print('[CFG] YOLOv5 confidence threshold:', self._cfg['conf'])
        self._detector.model.conf = self._cfg['conf']

    def set_input_size(self) -> None:
        print('[CFG] YOLOv5 input size:', self._cfg['size'])
        self._detector.input_size = self._cfg['size']

    def get_product(self) -> DetectorBase:
        product = self._detector
        self.reset()
        return product


class DetectorDirector:

    def __init__(self) -> None:
        self._builder = None

    def set_builder(self, builder: DetectorBuilder) -> None:
        self._builder = builder

    def build_YOLOv5(self) -> None:
        self._builder.reset()
        self._builder.set_model()
        self._builder.set_input_size()


if __name__ == '__main__':
    director = DetectorDirector()
    builder = YOLOv5Builder()
    director.set_builder(builder)
    director.build_YOLOv5()
    detector = builder.get_product()

    video_loader = cv2.VideoCapture('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/street.mp4')
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    from mct2.utils.vis_utils import plot_box

    buf = []
    frame_count = 0

    for _ in range(int(video_loader.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame_count += 1
        print('frame:', frame_count)
        ret, frame = video_loader.read()

        if not ret or frame is None:
            break

        dets = detector.predict(frame, BGR=True)

        show_img = plot_box(frame, dets[:, :4])
        cv2.imshow('show', show_img)
        cv2.waitKey(1)

        for det in dets:
            # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
            buf.append(
                f'{frame_count}, -1, {det[0].item():.6f}, {det[1].item():.6f}, {(det[2] - det[0]).item():.6f}, {(det[3] - det[1]).item():.6f}, {det[4].item():.6f}, -1, -1, -1')

    # print('\n'.join(buf), file=open('../../output/dets.txt', 'w'))
    video_loader.release()

