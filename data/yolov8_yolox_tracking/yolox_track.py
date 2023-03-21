#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import sys
from pathlib import Path

import cv2

import torch

HERE = Path(__file__).parent

sys.path.append(str(HERE))

from yolox_main.yolox.data.data_augment import ValTransform
from yolox_main.yolox.data.datasets import COCO_CLASSES
from yolox_main.yolox.exp import get_exp
from yolox_main.yolox.utils import fuse_model, get_model_info, postprocess, vis

from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box



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
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="please input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu" if torch.cuda.is_available() else 'cpu', type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    ################################### LAP ####################################
    parser.add_argument('--tracking_method', default='strongsort', type=str)
    parser.add_argument('--tracking_config', default=None, type=Path)
    parser.add_argument('--reid-weights', type=Path, default=HERE / 'osnet_x1_0.pt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

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
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def imageflow_demo(predictor, vis_folder, current_time, args, tracker):


    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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

            start_inference_time = time.time()
            [outputs], img_info = predictor.inference(frame)
            end_inference_time = time.time()
            inference_time = 0.5 * inference_time + 0.5 * (end_inference_time - start_inference_time)
            # output is None or of [[x, y, w, h, obj_ness, cls_ness, cls_id],...]

            annotator = Annotator(frame, line_width=2, example=None)


            # outputs[0] <=> det
            start_tracker_time = time.time()
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and frame is not None:  # camera motion compensation
                    tracker.tracker.camera_update(prev_frames, frame)

            if outputs is None:
                outputs = torch.empty((0, 7), dtype=torch.float32)
            
            with torch.no_grad():
                outputs[:, 0:4] /= img_info["ratio"]
                outputs = outputs.cpu()
                outputs = torch.cat([outputs[outputs[:, 6] == cls] for cls in args.classes], dim=0)
                outputs = tracker.update(outputs, frame)
            end_tracker_time = time.time()
            tracker_time = 0.5 * tracker_time + 0.5 * (end_tracker_time - start_tracker_time)

            if len(outputs) > 0:

                for j, (output) in enumerate(outputs):
                    
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    if args.save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, conf, -1, -1, -1))

                    if args.save_vid:  # Add bbox/seg to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = f'{id} {conf:.2f}'
                        color = colors(c, True)
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

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()

    # Create as many strong sort instances as there are video sources
    tracker = create_tracker(
        args.tracking_method, 
        args.tracking_config, 
        args.reid_weights, 
        torch.device('cuda' if args.device == 'gpu' else 'cpu'), 
        args.half
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
