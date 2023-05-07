import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
import os
from tqdm import tqdm
from pathlib import Path
import sys

HERE = Path(__file__).parent

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

    for i, kid in enumerate(range(num_kpts)):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf > 0.5:
                    cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                else:
                    cv2.circle(im, (int(x_coord), int(y_coord)), radius - 2, (255, 0, 0), -1)
            


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


def get_loc(box, kpt, steps):

    locs = [kpt[i*3: i*3 + 2] for i in range(17)]
    conf = kpt[[2 + 3*i for i in range(17)]]
    f = [None, None]
    i = [[15, 13, 11, 5], [16, 14, 12, 6]]
    for c in range(2):
        if conf[i[c][0]] >= 0.5:
            if conf[i[c][1]] >= 0.5:
                f[c] = locs[i[c][0]] + 1/6 * (locs[i[c][0]] - locs[i[c][1]])
            elif conf[i[c][2]] >= 0.5:
                f[c] = locs[i[c][0]] + 1/10 * (locs[i[c][0]] - locs[i[c][2]])
            else:
                f[c] = locs[i[c][0]]
        elif conf[i[c][1]] >= 0.5:
            if conf[i[c][3]] >= 0.5:
                f[c] = locs[i[c][1]] + 6/11 * (locs[i[c][1]] - locs[i[c][3]])
            elif conf[i[c][2]] >= 0.5:
                f[c] = locs[i[c][1]] + (locs[i[c][1]] - locs[i[c][2]])
        elif conf[i[c][2]] >= 0.5:
            if conf[i[c][3]] >= 0.5:
                f[c] = locs[i[c][2]] + (locs[i[c][2]] - locs[i[c][3]])

    a, b, c, d = box
    if f[0] is not None and f[1] is not None:
        return (f[0] + f[1]) / 2
    elif f[0] is None and f[1] is None:
        return (a, b + d/2)
    else:
        x, y = f[0] if f[0] is not None else f[1]
        return (x + a) / 2, y
        
        

    

    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weigths = torch.load(HERE / '../../../mct/weights/yolov7-w6-pose.pt', map_location=device)

model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

for path in ['/home/tran/Pictures/Screenshot from 41_00009_2023-04-13_08-30-00-000000.avi - 1.png']: #tqdm((HERE / 'laptq_in').glob('*')):

    path = str(path)
    image = cv2.imread(path)
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)  # shape [1, 34425, 57]
        output = non_max_suppression_kpt(output, 0.25, 0.45, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        kpt = output[idx, 7:]
        nimg = plot_skeleton_kpts(nimg, kpt.T, 3)
        batch_id, cls_id, a, b, c, d, conf = output[idx, :7].astype('int32')
        cv2.rectangle(nimg, (a - c // 2, b - d // 2), (a + c // 2, b + d // 2), color=(255, 0, 0), thickness=1)

        loc = get_loc([a, b, c, d], kpt, steps=3)
        cv2.circle(nimg, np.int32(loc), 6, (0, 255, 0), -1)
        
    cv2.imwrite(str(HERE/'laptq_out/yolov7_pose_') + os.path.split(path)[1], nimg[:, :, ::-1])


