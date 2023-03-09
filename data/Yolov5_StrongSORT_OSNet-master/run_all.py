import os
from  multiprocessing import Pool
import math

here = os.getcwd()
def run(vid):
    os.system(
        f'python track.py --tracking-method strongsort --source ../recordings/{vid}  --yolo-weights yolov5/weights/20211107_PersonHand_YOLOV5l_832x832_Satudora-Fisheyes_AWLVN.pt --classes 0 --img 2560 --reid-weights ./osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pt --save-txt')

if __name__ == '__main__':
    vid_list = os.listdir(here + '/../recordings')
    size = 8
    vid_list = [i for i in vid_list if i[0] == '6']
    pool = Pool(size)
    pool.starmap(run, zip(vid_list))
    #for i in range(math.ceil(len(vid_list) / size)):
    #    pool = Pool(size)
    #    pool.starmap(run, zip(vid_list[i*size:(i+1)*size]))



