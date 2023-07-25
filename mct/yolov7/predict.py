import logging
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from .utils.datasets import letterbox
from .utils.general import non_max_suppression_kpt
from .utils.plots import output_to_keypoint


logger = logging.getLogger(__name__)


class YOLOv7Pose(object):
    def __init__(
        self,
        weight,
        device,
        conf_thres,
        iou_thres,
        test_size
    ):
        self.weight = weight
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.test_size = test_size
        self.device = device

        self._load_model()

    
    def _load_model(self):
        self.model = torch.load(self.weight, map_location=self.device)['model']
        self.model.float().eval()

        if self.device.type == 'cuda':
            self.model.half().to(self.device)
        

    def inference(self, img):
        image, (ratio_w, ratio_h), (pad_w, pad_h) = letterbox(img, self.test_size, stride=64, auto=True)
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if self.device.type == 'cuda':
            image = image.half().to(self.device)
        with torch.no_grad():
            output, _ = self.model(image)  # shape [1, 34425, 57]
            output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)

        # rescale to original image size
        if len(output) > 0:
            output[:, 2:4] = (output[:, 2:4] - [pad_w, pad_h]) / [ratio_w, ratio_h]
            output[:, 4:6] = output[:, 4:6] / [ratio_w, ratio_h]
            for i in range(17):
                output[:, 7 + i*3 : 9 + i*3] = (output[:, 7 + i*3 : 9 + i*3] - [pad_w, pad_h]) / [ratio_w, ratio_h]

        # output is of [[batch_id, class_id, x, y, w, h, conf, *kpts], ...]
        return output



            
    