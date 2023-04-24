import argparse
import os
import time
from loguru import logger
import sys
from pathlib import Path

import cv2
import numpy as np

import torch
from torchvision import transforms

HERE = Path(__file__).parent

sys.path.append(str(HERE))

from yolov7_main.utils.datasets import letterbox
from yolov7_main.utils.general import non_max_suppression_kpt
from yolov7_main.utils.plots import output_to_keypoint, plot_skeleton_kpts
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from yolox_main.yolox.exp import get_exp


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam"    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video"    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save-txt", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("--save-vid", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("--device", default="gpu" if torch.cuda.is_available() else 'cpu', type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="please input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    ################################### LAP ####################################
    parser.add_argument("--conf_thres", default=0.25, type=float, help="test conf")
    parser.add_argument("--iou_thres", default=0.45, type=float, help="test nms threshold")
    parser.add_argument('--tracking_method', default='strongsort', type=str)
    parser.add_argument('--tracking_config', default=None, type=Path)
    parser.add_argument('--reid-weights', type=Path, default=HERE / 'osnet_x1_0.pt')

    args = parser.parse_args()
    args.tracking_config = HERE / 'trackers' / args.tracking_method / 'configs' / (args.tracking_method + '.yaml')
    ############################################################################
    return args


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device,
        conf_thres,
        iou_thres,
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.test_size = exp.test_size
        self.device = device
        

    def inference(self, img):
        t0 = time.time()
        image, (ratio_w, ratio_h), (pad_w, pad_h) = letterbox(img, self.test_size, stride=64, auto=True)
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
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

        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        # output is of [[batch_id, class_id, x, y, w, h, conf, *kpts], ...]
        return output
        


def imageflow_demo(predictor, vis_folder, current_time, args, tracker):


    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if vis_folder is not None:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
    basename = os.path.basename(args.path)

    if args.save_txt:
        txt_path = os.path.join(save_folder, os.path.splitext(basename)[0] + '.txt')

    if args.save_vid:
        
        if args.demo == "video":
            save_path = os.path.join(save_folder, basename)
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    prev_frames = None
    frame_idx = 0
    inference_time = 0
    tracker_time = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            
            result_frame = frame
            
            start_inference_time = time.time()
            outputs_kpt = predictor.inference(frame)
            end_inference_time = time.time()
            inference_time = 0.5 * inference_time + 0.5 * (end_inference_time - start_inference_time)
            # outputs_kpt is None or of [[batch_id, class_id, x, y, w, h, conf, *kpts], ...]

            # outputs[0] <=> det
            start_tracker_time = time.time()
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and frame is not None:  # camera motion compensation
                    tracker.tracker.camera_update(prev_frames, frame)

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
                outputs_box = tracker.update(outputs_box, frame, return_original_box=True)
            end_tracker_time = time.time()
            tracker_time = 0.5 * tracker_time + 0.5 * (end_tracker_time - start_tracker_time)

            if len(outputs_box) > 0:

                for j, (output) in enumerate(outputs_box):
                    
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    kpt = outputs_kpt[np.argmin(np.sum(np.square(outputs_kpt[:, 2:6] - bbox), axis=1)), 7:]


                    if args.save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 61 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, conf, -1, -1, -1, *kpt))

                    if args.save_vid:  # Add bbox/seg to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = f'{id} {conf:.2f}'
                        color = colors(c, True)
                        plot_skeleton_kpts(frame, kpt.T, 3)
                        annotator = Annotator(frame, line_width=2, example=None)
                        annotator.box_label(bbox, label, color=color)
                        result_frame = annotator.result()
            
            frame_idx += 1        
            prev_frames = frame

            if args.save_vid:
                vid_writer.write(result_frame)
        else:
            break

    logger.info(f'{inference_time * 1000:.1f}ms inference ({1/inference_time} FPS), {tracker_time * 1000:.1f}ms {args.tracking_method} ({1/tracker_time} FPS) => {(inference_time + tracker_time) * 1000:.1f}ms combined ({1/(inference_time + tracker_time)} FPS)')


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_txt or args.save_vid:
        vis_folder = file_name
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf_thres is not None:
        exp.test_conf = args.conf_thres
    if args.iou_thres is not None:
        exp.iou_thres = args.iou_thres
    if args.tsize is not None:
        exp.test_size = args.tsize

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weigths = torch.load(args.ckpt, map_location=device)

    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    predictor = Predictor(
        model, exp,
        device, args.conf_thres, args.iou_thres
    )
    current_time = time.localtime()

    # Create as many strong sort instances as there are video sources
    tracker = create_tracker(
        args.tracking_method, 
        args.tracking_config, 
        args.reid_weights, 
        device, 
        False
    )
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()
    outputs = [None]

    imageflow_demo(predictor, vis_folder, current_time, args, tracker)


if __name__ == "__main__":
    args = make_parser()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
