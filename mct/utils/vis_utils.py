import numpy as np
import cv2


COLORS = [
    # RED, YELLO, BLUE, GREEN, PUPLE
    (83, 50, 250), (55, 250, 250), (255, 221, 21), (102, 255, 102), (240, 120, 240),
    (83, 179, 36), (240, 120, 240), (51, 153, 204), (187, 125, 250), (51, 204, 255),
    (80, 83, 239), (59, 235, 255), (247, 195, 79), (132, 199, 129), (200, 104, 186),
    (53, 57, 229), (45, 192, 251), (229, 155, 3), (71, 160, 67), (176, 39, 156),
]


def plot_box(img, boxes, thickness=4, texts=None):
    """
    boxes: [[frame, id, x1, y1, w, h, conf, ...],...] (MOT format)
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    assert len(boxes.shape) == 2, "Invalid 'boxes' shape, must have dim == 2"

    if texts is not None:
        assert len(texts) == len(boxes)

    img = img.copy()
    ids = np.int32(boxes[:, 1])
    xyxys = np.int32(boxes[:, 2:6])
    xyxys[:, 2:] += xyxys[:, :2]                    # type: ignore
    confs = boxes[:, 6]

    # xyxys[:, 0] = np.clip(xyxys[:, 0], 0, img.shape[1])     # type: ignore
    # xyxys[:, 1] = np.clip(xyxys[:, 1], 0, img.shape[0])     # type: ignore
    # xyxys[:, 2] = np.clip(xyxys[:, 2], 0, img.shape[1])     # type: ignore
    # xyxys[:, 3] = np.clip(xyxys[:, 3], 0, img.shape[0])     # type: ignore

    # TODO parallel
    for i in range(len(boxes)):
        if ids[i] == -1:                            # type: ignore
            color = COLORS[3]
        else:
            color = COLORS[ids[i] % len(COLORS)]    # type: ignore
            space = 5
            expected_text_H = 60

            y_text = xyxys[i, 1] - space if xyxys[i, 1] - space - expected_text_H > 0 else xyxys[i, 1] + space + expected_text_H    # type: ignore
            x_text = xyxys[i, 0] + space                                                                                            # type: ignore
            cv2.putText(
                img, 
                str(ids[i]) + \
                    (f' ({int(confs[i] * 100)})' if confs[i] != -1 else '') + \
                        ((' ' + texts[i]) if texts is not None else ''), 
                (x_text, y_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2.0,
                color, 
                thickness=thickness)    # type: ignore
        cv2.rectangle(img, xyxys[i, :2], xyxys[i, 2:4], color=color, thickness=thickness)                                           # type: ignore

    return img


def plot_loc(img, locs, radius=8, texts=None, text_thickness=2):
    # locs: [[frame, id, x, y, ...],...]
    if not isinstance(locs, np.ndarray):
        locs = np.array(locs)
    
    img = img.copy()

    assert len(locs.shape) == 2, "Invalid 'locs' shape, must have dim == 2"
    
    if texts is not None:
        assert len(texts) == len(locs)

    ids = np.int32(locs[:, 1])
    xyxy = np.int32(locs[:, 2:4])

    for i in range(len(locs)):
        if ids[i] == -1:                            # type: ignore
            color = COLORS[3]
        else:
            color = COLORS[ids[i] % len(COLORS)]    # type: ignore
        
        cv2.circle(img, xyxy[i], radius=radius, color=color, thickness=-1)      # type: ignore
        if texts is not None:
            cv2.putText(img, texts[i], xyxy[i] + [4, -4], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, text_thickness)   # type: ignore
    
    return img


def plot_roi(img, roi, thickness=2, color=(255, 255, 255)):
    img = img.copy()
    roi = np.int32(roi)
    cv2.drawContours(img, [roi], -1, color, thickness=thickness)
    return img


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    im = im.copy()
    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)


    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    
    return im



def draw_track(img, track, id, color, radius=2, **kwargs):

    if not isinstance(track, np.ndarray):
        track = np.array(track)
    if len(track.shape) == 1:
        track = np.expand_dims(track, axis=0)
    assert len(track.shape) == 2 and track.shape[1] == 2, 'Invalid track dimension, must be (2,) or (N, 2)'

    img = img.copy()
    for i in range(len(track)):
        cv2.circle(img, np.int32(track[i]), radius=radius, color=color, thickness=-1)

        if i == len(track) - 1 and 'camid' in kwargs:
            cv2.putText(img, f"{str(kwargs['camid'])} ({id})", (int(track[i, 0]), int(track[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)

        if i == 0:
            continue
        cv2.line(img, np.int32(track[i]), np.int32(track[i - 1]), color=color, thickness=2 * radius)

    return img



