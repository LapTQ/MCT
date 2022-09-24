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
        """return [[x1, y1, x2, y2, conf], ...]"""
        pass


class YOLOv5(DetectorBase):

    class Builder:

        def __init__(self, cfg_path):
            """Construct from YAML"""
            self._reset()

            cfg_parent = Path(cfg_path).parent
            with open(cfg_path, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            # set model
            weight_path = cfg_parent / cfg['weights']
            assert os.path.exists(weight_path), 'No such file or directory'
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model.to(device)
            model.iou = cfg['iou']
            model.conf = cfg['conf']
            self._product.model = model

            print('[CFG] YOLOv5 model:', weight_path)
            print('[CFG] YOLOv5 device:', device)
            print('[CFG] YOLOv5 iou threshold:', cfg['iou'])
            print('[CFG] YOLOv5 confidence threshold:', cfg['conf'])

            # set input size
            self._product.input_size = cfg['size']
            print('[CFG] YOLOv5 input size:', cfg['size'])

        def _reset(self) -> None:
            self._product = YOLOv5()

        def get_product(self) -> DetectorBase:
            product = self._product
            self._reset()
            return product

    def predict(self, img: np.ndarray, BGR: bool) -> np.ndarray:
        """
        return [[x1, y1, x2, y2, conf], ...]
        """
        if BGR:
            img = img[:, :, ::-1]

        size = max(img.shape) if self.input_size == -1 else self.input_size
        preds = self.model([img], size=size)  # RGB, include NMS
        ret = preds.xyxy[0][:, :5].cpu().numpy()  # preds.xyxy = [[[x1, y1, x2, y2, conf, class_id], ...]]
        return ret


HERE / '../configs/yolov5s.yaml'


if __name__ == '__main__':
    detector = YOLOv5.Builder(HERE / '../configs/yolov5s.yaml').get_product()

    # video_loader = cv2.VideoCapture('/media/tran/003D94E1B568C6D11/Workingspace/MCT/data/fish.mp4')
    # cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    # from mct.utils.vis_utils import plot_box
    #
    # buf = []
    # frame_count = 0
    #
    # for _ in range(int(video_loader.get(cv2.CAP_PROP_FRAME_COUNT))):
    #     frame_count += 1
    #     print('frame:', frame_count)
    #     ret, frame = video_loader.read()
    #
    #     if not ret or frame is None:
    #         break
    #
    #     dets = detector.predict(frame, BGR=True)
    #
    #     show_img = plot_box(frame, dets[:, :4])
    #     cv2.imshow('show', show_img)
    #     cv2.waitKey(1)
    #
    #     for det in dets:
    #         # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
    #         buf.append(
    #             f'{frame_count}, -1, {det[0].item():.6f}, {det[1].item():.6f}, {(det[2] - det[0]).item():.6f}, {(det[3] - det[1]).item():.6f}, {det[4].item():.6f}, -1, -1, -1')
    #
    # # print('\n'.join(buf), file=open('../../output/dets.txt', 'w'))
    # video_loader.release()

