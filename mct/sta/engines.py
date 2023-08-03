from abc import ABC, abstractmethod
import queue
import time
import cv2
import numpy as np

from typing import Any, Union
from mct.bytetrack.byte_tracker import BYTETracker
from mct.yolov7.predict import YOLOv7Pose
import logging
import torch
import yaml


logger = logging.getLogger(__name__)


class Scene:

    def __init__(
            self,
            width: Union[int, float, None] = None,
            height: Union[int, float, None] = None,
            roi: Union[np.ndarray, None] = None,
            roi_test_offset: Union[int, float] = 0,
            name='Scene'
    ) -> None:

        self.width = width
        self.height = height

        self.roi = roi
        self._check_roi()
        self.roi_test_offset = roi_test_offset

        self.name = name


    def is_in_roi(self, x: Union[tuple, list, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if point(s) (x, y) is in the scene's roi.

        If x is not a numpy array represent only 1 point, then return bool object.
        """
        is_numpy = True
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            is_numpy = False

        x = np.float32(x.reshape(-1, 2))    # type: ignore
        ret = [cv2.pointPolygonTest(self.roi, p, True) >= self.roi_test_offset for p in x]      # type: ignore

        if not is_numpy and len(ret) == 1:
            return ret[0]
        else:
            return np.array(ret, dtype=bool)


    def has_roi(self) -> bool:
        return self.roi is not None


    def _check_roi(self):
        if self.roi is not None:
            assert isinstance(self.roi, np.ndarray)
            self.roi = np.int32(self.roi)   # type: ignore


class MyQueue(queue.Queue):

    def __init__(self, maxsize: int = 0, name: Any = 'MyQueue') -> None:
        super().__init__(maxsize)
        self.name = name

        logger.debug(f'{self.name}:\t initilized')


class Tracker:

    def __init__(
            self,
            detection_mode: str,
            tracking_mode: str,
            detection_weight: Union[float, None] = None,
            detection_conf_thres: Union[float, None] = None,
            detection_iou_thres: Union[float, None] = None,
            detection_tsize: Union[int, None] = None,
            tracking_config: Union[str, None] = None,
            device: str = 'cuda',
            txt_path: Union[str, None] = None,
            use_real_tracker: bool = False,
            name='Tracker'
    ) -> None:

        self.detection_mode = detection_mode
        self.tracking_mode = tracking_mode

        self.detection_weight = detection_weight
        self.detection_conf_thres = detection_conf_thres
        self.detection_iou_thres = detection_iou_thres
        self.detection_tsize = detection_tsize
        self.tracking_config = tracking_config
        self.device = device
        self.txt_path = txt_path
        self.use_real_tracker = use_real_tracker
        self.name = name

        assert not (self.txt_path is None and not self.use_real_tracker), 'txt_path must be provided if use_real_tracker is False'

        # if using offline mock tracking result
        if txt_path is None:
            self._load_tracker()
        else:
            self._load_txt_result()
            if self.use_real_tracker:
                self._load_tracker()

        logger.info(f'{self.name}:\t initialized with DETECTION_MODE={self.detection_mode}, TRACKING_MODE={self.tracking_mode}')


    def _load_tracker(self):

        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.detector = YOLOv7Pose(
            self.detection_weight,
            self.device,
            self.detection_conf_thres,
            self.detection_iou_thres,
            self.detection_tsize
        )

        # Create as many strong sort instances as there are video sources
        with open(self.tracking_config, 'r') as f:  # type: ignore
            tracking_cfg = yaml.safe_load(f)

        self.tracking = BYTETracker(
            track_thresh=tracking_cfg['track_thresh'],
            match_thresh=tracking_cfg['match_thresh'],
            track_buffer=tracking_cfg['track_buffer'],
            frame_rate=tracking_cfg['frame_rate']
        )

        self._refresh_tracker()

        logger.info(f'{self.name}:\t loaded detector and tracker on {self.device}')

    
    def estimate_fps(self, result_list):
        fps = self.detector.estimate_fps()
        result_list.append(fps)
    
    
    def _refresh_tracker(self):
        self.prev_frame = None
        self.frame_idx = 0
        self.inference_time = 0
        self.tracker_time = 0


    def _load_txt_result(self):
        assert self.txt_path is not None
        try:
            self.seq = np.loadtxt(self.txt_path)
        except:
            self.seq = np.loadtxt(self.txt_path, delimiter=',')

        logger.info(f'{self.name}:\t load tracking result from .txt at {self.txt_path}')


    def _infer_txt(self, frame_id) -> np.ndarray:
        assert  isinstance(frame_id, int), 'frame_id must be provided in mock test'
        return self.seq[self.seq[:, 0] == frame_id]


    def _infer_tracker(self, img) -> np.ndarray:
        assert isinstance(img, np.ndarray)

        start_inference_time = time.time()
        outputs_kpt = self.detector.inference(img)
        end_inference_time = time.time()
        self.inference_time = 0.5 * self.inference_time + 0.5 * (end_inference_time - start_inference_time)
        # outputs_kpt is None or of [[batch_id, class_id, x, y, w, h, conf, *kpts], ...]

        # outputs[0] <=> det
        outputs_kpt = outputs_kpt.reshape(-1, 58)
        # re-format to [[batch_id, class_id, x1, y1, x2, y2, conf, *kpts],...]
        outputs_kpt[:, 2:4] -= outputs_kpt[:, 4:6] / 2
        outputs_kpt[:, 4:6] += outputs_kpt[:, 2:4]

        with torch.no_grad():

            outputs_box = np.concatenate(
                [outputs_kpt[:, 2:7], np.tile([1, 0], len(outputs_kpt)).reshape(-1, 2)],
                axis=1
            )
            outputs_box = torch.from_numpy(outputs_box)
            outputs_box = self.tracking.update(outputs_box, img, return_original_box=True)

        dets = []
        for output in outputs_box:

            bbox = output[0:4]
            id = output[4]
            conf = output[6]

            kpt = outputs_kpt[np.argmin(np.sum(np.square(outputs_kpt[:, 2:6] - bbox), axis=1)), 7:]

            # to MOT format
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_w = output[2] - output[0]
            bbox_h = output[3] - output[1]

            if self.detection_mode == 'box':
                d = [self.frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, 0]
            elif self.detection_mode == 'pose':
                d = [self.frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, conf, -1, -1, -1, *kpt]
            else:
                raise ValueError('detection_mode must be eight "box" or "pose"')

            dets.append(d)

        if len(dets) > 0:
            dets = np.array(dets)
        else:
            dets = np.empty((0, 10 if self.detection_mode == 'box' else 61))

        self.frame_idx += 1
        self.prev_frame = img

        logger.debug(f'{self.name}:\t {1/self.inference_time} FPS detection')

        return dets


    def infer(self, img: Union[np.ndarray, None], frame_id: Union[int, None] = None) -> np.ndarray:
        """During development, we can use .txt file to mock the tracking result.
        if txt is not provided, then use real tracker.
        if txt is provided, we still use it to correct ID switch problem.
            in this case, real tracker can still be turn on to test speed
            or is turned off to test app functions.
        """

        # if detection_mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
        # if detection_mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
        
        if self.txt_path is None:
            dets = self._infer_tracker(img)
        else:
            dets = self._infer_txt(frame_id)
            if self.use_real_tracker:
                self._infer_tracker(img)

        logger.debug(f'{self.name}:\t frame {frame_id} detected {len(dets)} people')

        return dets


class FilterBase(ABC):

    @abstractmethod
    def __init__(
        self,
        name='FP filter'
    ) -> None:

        self.name = name

    @abstractmethod
    def __call__(self, x) -> float:
        pass


class IQRFilter(FilterBase):

    def __init__(
            self,
            q1: Union[int, float] = 25,
            q2: Union[int, float] = 75,
            name='IQR filter'
    ) -> None:
        super().__init__(name)

        self.q1 = q1
        self.q2 = q2

        logger.info(f'{self.name}: \t initialized')


    def __call__(self, x) -> float:

        p1, p2 = np.percentile(x, [self.q1, self.q2])
        iqr = p2 - p1
        ub = p2 + 1.5 * iqr
        logger.debug(f'{self.name}:\t upper bound = {ub}')

        # filter out false matches due to missing detection boxes
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('agg')
        # plt.figure()
        # plt.hist(x.flatten(), bins=42)
        # plt.plot([ub, ub], plt.ylim())
        # plt.savefig('img.png')

        return ub


class GMMFilter(FilterBase):

    def __init__(
            self,
            n_components: int,
            std_coef: float = 3.0,
            name='GMM filter'
    ) -> None:
        super().__init__(name)

        self.n_components = n_components
        self.std_coef = std_coef

        logger.info(f'{self.name}: \t initialized')


    def __call__(self, x) -> float:

        assert len(x.shape) == 2 and x.shape[1] == 1, 'Invalid shape for x, expect (N, 1)'

        np.random.seed(42)
        from sklearn.mixture import GaussianMixture
        gmm_error_handled = False
        reg_covar = 1e-6
        while not gmm_error_handled:
            try:
                logger.debug(f'{self.name}:\t trying reg_covar = {reg_covar}')
                gm = GaussianMixture(n_components=self.n_components, covariance_type='diag', reg_covar=reg_covar).fit(x)
                gmm_error_handled = True
            except:
                logger.warning(f'{self.name}:\t reg_covar failed!')
                reg_covar *= 10
        smaller_component = np.argmin(gm.means_)                # type: ignore
        ub = gm.means_[smaller_component] + self.std_coef * np.sqrt(gm.covariances_[smaller_component])     # type: ignore
        logger.debug(f'{self.name}:\t smaller component has mean = {min(gm.means_)} and std = {np.sqrt(gm.covariances_[smaller_component])}')  # type: ignore
        logger.debug(f'{self.name}:\t upper bound = {ub}')

        # filter out false matches due to missing detection boxes
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('agg')
        # plt.figure()
        # plt.hist(x.flatten(), bins=42)
        # plt.plot([ub, ub], plt.ylim())
        # plt.savefig('img.png')

        return ub