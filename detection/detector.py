import os
from pathlib import Path

import torch
import yaml
import cv2

HERE = Path(__file__).parent

class YOLOv5:

    def __init__(self):
        self.model = None
        self.size = None

        self.load_config()

    def load_config(self):
        with open(HERE/'models'/'yolov5'/'config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        print('[CFG] YOLOv5 model:', config['model'])

        weights = HERE/'models'/'yolov5'/config['weights']
        print('[CFG] YOLOv5 weights:', weights)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

        print('[CFG] YOLOv5 confidence threshold:', config['conf'])
        self.model.conf = config['conf']

        print('[CFG] YOLOv5 NMS IoU threshold:', config['conf'])
        self.model.iou = config['iou']

        print('[CFG] YOLOv5 image size:', config['size'])
        self.size = config['size']

    def predict(self, img, BGR):
        """
        return [[x1, y1, x2, y2, conf], ...]
        """
        if BGR:
            img = img[:, :, ::-1]

        preds = self.model([img], size=self.size) # RGB, include NMS
        ret = preds.xyxy[0][:, :5].numpy()  # preds.xyxy = [[[x1, y1, x2, y2, conf, class_id], ...]]
        return ret


if __name__ == '__main__':
    detector = YOLOv5()
    video_loader = cv2.VideoCapture('/media/tran/003D94E1B568C6D11/Workingspace/MCT/videos/street.mp4')

    buf = []
    frame_count = 0


    for _ in range(int(video_loader.get(cv2.CAP_PROP_FRAME_COUNT))):
        frame_count += 1
        print('frame:', frame_count)
        ret, frame = video_loader.read()

        if not ret or frame is None:
            break

        dets = detector.predict(frame, BGR=True)

        for det in dets:
            # [frame, id, x1, y1, w, h, conf, -1, -1, -1]
            buf.append(f'{frame_count}, -1, {det[0].item():.6f}, {det[1].item():.6f}, {(det[2] - det[0]).item():.6f}, {(det[3] - det[1]).item():.6f}, {det[4].item():.6f}, -1, -1, -1')

    print('\n'.join(buf), file=open('../output/dets.txt', 'w'))
    video_loader.release()