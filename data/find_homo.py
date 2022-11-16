import cv2
from pathlib import Path
import numpy as np
import argparse

HERE = Path(__file__).parent

CAM1 = str(HERE / 'recordings/21_00000_2022-11-03_14-56-57-643967.avi')
CAM2 = str(HERE / 'recordings/27_00000_2022-11-03_14-56-56-863473.avi')

# rtsp://admin:123456a@@192.168.3.63/live
# rtsp://admin:123456a@@192.168.3.64/live
# rtsp://admin:12345@192.168.3.21/live
# rtsp://admin:12345@192.168.3.27/live




def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--dst', type=str, required=True)
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--video', action='store_true')

    opt = vars(ap.parse_args())

    return opt


def extract_frame(cap):
    while True:
        success, frame = cap.read()
        if success:
            cap.release()
            return frame


def select_matches(src, dst):

    global img
    global src_pts
    global dst_pts
    global is_src

    H1, W1 = src.shape[:2]

    window_name = '<ESC>'

    def on_mouse(event, x, y, flag, param):

        global img
        global src_pts
        global dst_pts
        global is_src

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), radius=10, color=(0, 0, 255), thickness=2)
            if is_src:
                src_pts.append([x, y])
            else:
                dst_pts.append([x - W1, y])
                print(f'[INFO] {src_pts[-1]} matched to {dst_pts[-1]}')
                cv2.line(img, src_pts[-1], (x, y), color=(0, 255, 0), thickness=2)
            is_src = not is_src

            cv2.imshow(window_name, img)

    confirm = False
    is_src = True

    while not confirm:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        img = np.concatenate([src, dst], axis=1)
        cv2.imshow(window_name, img)
        src_pts = []
        dst_pts = []
        cv2.setMouseCallback(window_name, on_mouse)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        confirm = not bool(input('[QUERY] Confirm? [<ENTER> for yes] ').strip())

    return (np.array(src_pts).reshape(-1, 1, 2).astype('float32'),
            np.array(dst_pts).reshape(-1, 1, 2).astype('float32'))


def main(opt):

    if opt['video']:
        cap1 = cv2.VideoCapture(opt['src'])
        cap2 = cv2.VideoCapture(opt['dst'])

        src = extract_frame(cap1)
        dst = extract_frame(cap2)

    else:
        src = cv2.imread(opt['src'])
        dst = cv2.imread(opt['dst'])


    src_pts, dst_pts = select_matches(src, dst)
    H, mask = cv2.findHomography(src_pts, dst_pts) # cv2.RANSAC

    if opt['name'] is not None:
        np.savetxt(opt['name'], H)
        print('[INFO] Homography saved in', opt['name'])
    else:
        print('[INFO] Not save homography')

    src_transformed = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

    collage = np.concatenate([src_transformed, dst], axis=1)
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.imshow('show', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
[INFO] [530, 447] matched to [736, 488]
[INFO] [476, 485] matched to [703, 445]
[INFO] [525, 235] matched to [1034, 528]
[INFO] [627, 268] matched to [984, 622]
[INFO] [616, 626] matched to [532, 499]
[INFO] [710, 403] matched to [753, 659]
[INFO] [814, 635] matched to [450, 631]
[INFO] [708, 283] matched to [963, 698]
'''

if __name__ == '__main__':

    # opt = parse_opt()

    opt = {
        'src': CAM1,
        'dst': CAM2,
        'name': '21_to_27.txt',
        'video': True
    }

    main(opt)