import cv2
from pathlib import Path
import numpy as np
import argparse

HERE = Path(__file__).parent

CAM1 = str(HERE / 'recordings/2d_v2/21_00019_2022-12-02_18-15-20-498917.avi')
CAM2 = str(HERE / 'recordings/2d_v2/27_00019_2022-12-02_18-15-21-292795.avi')

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
    return (np.array([[[1124, 409]], [[1200, 543]], [[1120, 476]], [[1128, 566]], [[1044, 454]], [[1127, 667]], [[1044, 543]],
              [[1186, 497]], [[1037, 378]]], dtype='float32'),
            np.array([[[265, 320]], [[347, 448]], [[240, 393]], [[221, 486]], [[137, 400]], [[208, 605]], [[117, 481]],
                      [[341, 400]],
                      [[161, 319]]], dtype='float32'))

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


def select_ROI(img):

    global show_img
    global points

    show_img = img.copy()
    points = []

    window_name = 'Select ROI: <y> to submit. <ESC> to reset'

    def on_mouse(event, x, y, flag, param):

        global show_img
        global points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print('[INFO] Selected', points[-1])

            cv2.circle(show_img, (x, y), radius=10, color=(0, 0, 255), thickness=2)

            if len(points) >= 2:
                cv2.line(show_img, points[-2], points[-1], color=(0, 255, 0), thickness=2)

            cv2.imshow(window_name, show_img)

    while True:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, show_img)
        cv2.setMouseCallback(window_name, on_mouse)
        key = cv2.waitKey(0)
        if key == 27:
            show_img = img.copy()
            points = []
        elif key == ord('y'):
            if len(points) <= 2:
                print('[WARNING] Need at least 3 points')
                continue
            cv2.line(show_img, points[-1], points[0], color=(0, 255, 0), thickness=2)
            cv2.imshow(window_name, show_img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            break

    points = np.array(points, dtype='float32')
    points[:, 0] /= img.shape[1]
    points[:, 1] /= img.shape[0]
    points = points.reshape(-1, 1, 2)

    return points


def main(opt):

    if opt['video']:
        cap1 = cv2.VideoCapture(opt['src'])
        cap2 = cv2.VideoCapture(opt['dst'])

        src = extract_frame(cap1)
        dst = extract_frame(cap2)

    else:
        src = cv2.imread(opt['src'])
        dst = cv2.imread(opt['dst'])


    while True:
        window_name = 'Select matches: <y> to confirm. <ESC> to reset'
        src_pts, dst_pts = select_matches(src, dst)
        print(src_pts)
        print(dst_pts)
        H, mask = cv2.findHomography(src_pts, dst_pts) # cv2.RANSAC

        src_transformed = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

        show_img = np.concatenate([src_transformed, dst], axis=1)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, show_img)
        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == ord('y'):
                break
        if key == 27:
            continue
        elif key == ord('y'):
            cv2.destroyAllWindows()
            break

    show_img = np.uint8(src_transformed * 0.75 + dst * 0.25)
    contour = select_ROI(show_img)
    print(contour)

    if opt['name'] is not None:
        out_homo_path = str(Path(opt['src']).parent / opt['name'])
        np.savetxt(out_homo_path, H)
        print('[INFO] Homography saved in', out_homo_path)

        out_contour_path = str(Path(opt['src']).parent / ('roi_' + opt['name']))
        np.savetxt(out_contour_path, contour.reshape(-1, 2))
        print('[INFO] Contour saved in', out_contour_path)
    else:
        print('[INFO] Not save homography')
        print('[INFO] Not save contour')



'''
[[[530, 447]], [[476, 485]], [[525, 235]], [[627, 268]], [[616, 626]], [[710, 403]], [[814, 635]], [[708, 283]]]
[[[736, 488]], [[703, 445]], [[1034, 528]], [[984, 622]], [[532, 499]], [[753, 659]], [[450, 631]], [[963, 698]]]

[[[1124, 409]], [[1200, 543]], [[1120, 476]], [[1128, 566]], [[1044, 454]], [[1127, 667]], [[1044, 543]], [[1186, 497]], [[1037, 378]]]
[[[265, 320]], [[347, 448]], [[240, 393]], [[221, 486]], [[137, 400]], [[208, 605]], [[117, 481]], [[341, 400]], [[161, 319]]]
'''


if __name__ == '__main__':

    # opt = parse_opt()
    #
    # opt = {
    #     'src': CAM1,
    #     'dst': CAM2,
    #     'name': '21_to_27.txt',
    #     'video': True
    # }
    #
    # main(opt)
    
    # fix distortion with openCV: search "camera distortion openCV"

    src_pts = np.array([[[1124, 409]], [[1200, 543]], [[1120, 476]], [[1128, 566]], [[1044, 454]], [[1127, 667]], [[1044, 543]],
     [[1186, 497]], [[1037, 378]]], dtype='float32')
    dst_pts = np.array([[[265, 320]], [[347, 448]], [[240, 393]], [[221, 486]], [[137, 400]], [[208, 605]], [[117, 481]], [[341, 400]],
     [[161, 319]]], dtype='float32')

    src = cv2.VideoCapture(CAM1).read()[1]
    dst = cv2.VideoCapture(CAM2).read()[1]

    H, mask = cv2.findHomography(src_pts, dst_pts)  # cv2.RANSAC
    src_transformed = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

    # show_img = np.concatenate([src_transformed, dst], axis=1)
    show_img = np.uint8(src_transformed * 0.75 + dst * 0.25)

    roi = np.loadtxt('recordings/2d_v2/roi_21_to_27.txt', dtype='float32')
    roi[:, 0] *= dst.shape[1]
    roi[:, 1] *= dst.shape[0]
    roi = roi.reshape(-1, 1, 2).astype('int32')
    print(roi)

    cv2.drawContours(show_img, [roi], -1, (0, 255, 0), 3)

    point = [118.0, 483.0]
    print(cv2.pointPolygonTest(roi, point, False))

    # cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    # cv2.imshow('show', show_img)
    # cv2.waitKey(0)


