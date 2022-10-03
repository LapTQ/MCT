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


if __name__ == '__main__':

    import sys
    sys.path.append(sys.path[0] + '/../..')

    from mct.utils.vid_utils import ImageFolderLoader
    from tqdm import tqdm
    import os


    root = str(HERE/'../../data/MOT17/train')
    out_dir = str(HERE / '../../output/det')
    os.makedirs(out_dir, exist_ok=True)


    detector = YOLOv5.Builder(HERE / '../configs/yolov5s.yaml').get_product()

    for dir in os.listdir(root):
        input_path = os.path.join(root, dir, 'img1')
        loader = ImageFolderLoader.Builder(input_path, None).get_product()
        output_path = out_dir + '/' + dir + '.txt'
        txt_buffer = []
        out_txt = open(output_path, 'w')

        frame_count = 0
        pbar = tqdm(range(len(loader)))
        for _ in pbar:

            ret, frame = loader.read()

            # terminal condition
            condition = not ret or frame is None
            if condition:
                break

            frame_count += 1
            dets = detector.predict(frame, BGR=True)  # [[x1, y1, x2, y2, conf], ...]

            for obj in dets:
                # [[frame, x1, y1, x2, y2, conf]]
                txt_buffer.append(
                    f'{frame_count}, {float(obj[0])}, {float(obj[1])}, {float(obj[2])}, {float(obj[3])}, {float(obj[4])}')

        loader.release()
        print('\n'.join(txt_buffer), file=out_txt)
        print('[INFO] MOT17-format .txt saved in', output_path)
        out_txt.close()

