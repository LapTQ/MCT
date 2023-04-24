import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weigths = torch.load('../../../mct/weights/yolov7-w6-pose.pt', map_location=device)

model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

for path in tqdm(["laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 1.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 2.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 3.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 4.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 5.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 6.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 7.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 8.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 9.png",
"laptq_in/Screenshot from 42_00012_2023-04-16_08-30-00-000000.avi - 10.png",
"laptq_in/Screenshot from 43_00011_2023-04-15_08-30-00-000000.avi.png",
"laptq_in/Screenshot from 43_00011_2023-04-15_08-30-00-000000.avi - 1.png",
"laptq_in/Screenshot from 43_00011_2023-04-15_08-30-00-000000.avi - 2.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 1.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 2.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 3.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 4.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 5.png",
"laptq_in/Screenshot from 43_00012_2023-04-16_08-30-00-000000.avi - 6.png"]):

    image = cv2.imread(path)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)  # shape [1, 34425, 57]
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        batch_id, cls_id, a, b, c, d, conf = np.int32(output[idx, :7])
        cv2.rectangle(nimg, (a - c // 2, b - d // 2), (a + c // 2, b + d // 2), color=(255, 0, 0), thickness=1)
        
    cv2.imwrite('laptq_out/yolov7_pose_' + os.path.split(path)[1], nimg[:, :, ::-1])

