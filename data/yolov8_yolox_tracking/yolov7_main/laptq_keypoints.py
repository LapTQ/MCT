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
    radius = 2
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
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=1)
    
    return im


def get_loc(box, kpt, steps):

    locs = [kpt[i*3: i*3 + 2] for i in range(17)]
    conf = kpt[[2 + 3*i for i in range(17)]]
    f = [[], []]
    i = [[15, 13, 11, 5], [16, 14, 12, 6]]
    for c in range(2):
        if conf[i[c][0]] >= 0.5:
            if conf[i[c][2]] >= 0.5:
                print('MODE=1')
                f[c].append(locs[i[c][0]] + 1/8.5 * (locs[i[c][0]] - locs[i[c][2]]))
            if conf[i[c][1]] >= 0.5:
                print('MODE=2')
                f[c].append(locs[i[c][0]] + 1/4.5 * (locs[i[c][0]] - locs[i[c][1]]))
        if conf[i[c][1]] >= 0.5:
            if conf[i[c][2]] >= 0.5:
                print('MODE=5')
                f[c].append(locs[i[c][1]] + 5/4.7 * (locs[i[c][1]] - locs[i[c][2]]))
            if conf[i[c][3]] >= 0.5:
                print('MODE=4')
                f[c].append(locs[i[c][1]] + 2/4.8 * (locs[i[c][1]] - locs[i[c][3]]))
        if conf[i[c][2]] >= 0.5:
            print('MODE=6')
            if conf[i[c][3]] >= 0.5:
                f[c].append(locs[i[c][2]] + 4/3 * (locs[i[c][2]] - locs[i[c][3]]))
        
        if len(f[c]) == 0 and conf[i[c][0]] >= 0.5:
            print('MODE=3')
            f[c].append(locs[i[c][0]])

    a, b, c, d = box
    if len(f[0]) > 0  and len(f[1]) > 0:
        print('MODE=7')
        f[0] = np.mean(f[0], axis=0)
        f[1] = np.mean(f[1], axis=0)
        return (f[0] + f[1]) / 2
    elif len(f[0]) == len(f[1]) == 0:
        print('MODE=8')
        return (a, b + d/2)
    else:
        print('MODE=9')
        xy = f[0] if len(f[0]) > 0 else f[1]
        x, y = np.mean(xy, axis=0)
        return (x + a) / 2, y
        
        

    

    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weigths = torch.load(HERE / '../../../mct/weights/yolov7-w6-pose.pt', map_location=device)

model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

for path in tqdm(Path('/home/tran/Pictures/in/').glob('*')):

    path = str(path)
    print(path)
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
        # cv2.rectangle(nimg, (a - c // 2, b - d // 2), (a + c // 2, b + d // 2), color=(255, 0, 0), thickness=1)

        loc = get_loc([a, b, c, d], kpt, steps=3)
        cv2.circle(nimg, np.int32(loc), 2, (0, 255, 0), -1)
        
    cv2.imwrite(str(HERE/'laptq_out/yolov7_pose_') + os.path.split(path)[1], nimg[:, :, ::-1])


