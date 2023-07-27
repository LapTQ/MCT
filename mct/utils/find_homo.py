import cv2
from pathlib import Path
import numpy as np
import argparse

HERE = Path(__file__).parent

#CAM1 = str(HERE / '../../data/recordings/2d_v1/videos/21_00000_2022-11-03_14-56-57-643967.avi')
#CAM2 = str(HERE / '../../data/recordings/2d_v1/videos/27_00000_2022-11-03_14-56-56-863473.avi')

#CAM1 = str(HERE / '../../data/recordings/2d_v2/videos/21_00019_2022-12-02_18-15-20-498917.avi')
#CAM2 = str(HERE / '../../data/recordings/2d_v2/videos/27_00019_2022-12-02_18-15-21-292795.avi')

#CAM1 = str(HERE / '../../data/recordings/2d_v3/frames/frame_cam121.png')
#CAM2 = str(HERE / '../../data/recordings/2d_v3/frames/frame_cam127.png')

CAM1 = str(HERE / '../../data/recordings/2d_v4/frames/frame_cam42_2.png')
CAM2 = str(HERE / '../../data/recordings/2d_v4/frames/frame_cam43.png')
#CAM1 = str(HERE / '../../data/recordings/2d_v4/frames/frame_cam42_2.png')
#CAM2 = str(HERE / '../../data/recordings/2d_v4/frames/frame_cam43.png')

# rtsp://admin:123456a@@192.168.3.63/live
# rtsp://admin:123456a@@192.168.3.64/live
# rtsp://admin:12345@192.168.3.21/live
# rtsp://admin:12345@192.168.3.27/live

opt = {
        'src': CAM1,
        'dst': CAM2,
        'matches_out_path': None, #str(Path(CAM1).parent.parent / 'matches_42_to_43.txt'), # None
        'roi_out_path': None, #str(Path(CAM2).parent.parent / 'workarea_42.txt'),  # None
        'video': False,
        'draw_match_line': True
    }


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--dst', type=str, required=True)
    ap.add_argument('--matches_out_path', type=str, default=None)
    ap.add_argument('--roi_out_path', type=str, default=None)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--draw_match_line', type=bool, default=True)

    opt = vars(ap.parse_args())

    return opt


def extract_frame(cap):
    while True:
        success, frame = cap.read()
        if success:
            cap.release()
            return frame


def select_matches(src, dst):

    def _select_matches(src, dst):

        # return (np.array([[530, 447], [476, 485], [525, 235], [627, 268], [616, 626], [710, 403], [814, 635], [708, 283]], dtype='int32'),
        #         np.array([[736, 488], [703, 445], [1034, 528], [984, 622], [532, 499], [753, 659], [450, 631], [963, 698]], dtype='int32'))

        global img
        global src_pts
        global dst_pts
        global is_src

        H1, W1 = src.shape[:2]

        window_name = 'SELECT MATCHES (>= 4 matches): <ESC> to reset. <y> to see preview. <q> to abort.'

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
                    if opt['draw_match_line']:
                        cv2.line(img, src_pts[-1], (x, y), color=(0, 255, 0), thickness=2)
                is_src = not is_src

                cv2.imshow(window_name, img)

        is_src = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            img = np.concatenate([src, dst], axis=1)
            cv2.imshow(window_name, img)
            src_pts = []
            dst_pts = []
            cv2.setMouseCallback(window_name, on_mouse)

            key = cv2.waitKey(0)
            if key == 27:
                continue
            elif key == ord('y'):
                if len(src_pts) <= 3:
                    print('[WARNING] Need at least 4 points')
                    continue
                break
            elif key == ord('q'):
                exit(0)

        cv2.destroyAllWindows()

        return (np.array(src_pts, dtype='int32'), 
                np.array(dst_pts, dtype='int32'))
    
    while True:
        window_name = 'PREVIEW RESULT OF SELECTING MATCHES: <y> to submit. <ESC> to reset'
        #src_pts, dst_pts = _select_matches(src, dst)
        matches = np.loadtxt('recordings/2d_v4/matches_42_to_43.txt').astype('int32')
        src_pts, dst_pts = matches[:, :2], matches[:, 2:]
        src_pts = src_pts.astype('float32').reshape(-1, 1, 2)
        dst_pts = dst_pts.astype('float32').reshape(-1, 1, 2)
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
                cv2.destroyAllWindows()
                break
        if key == 27:
            continue
        elif key == ord('y'):
            cv2.destroyAllWindows()
            break
    
    return src_pts, dst_pts, H


def select_ROI(src, dst, homo):
    # if select roi in one image only, then set both src and homo = None
    
    global show_img
    global points
    

    if src is not None and homo is not None:
        src_transformed = cv2.warpPerspective(src, homo, (dst.shape[1], dst.shape[0]))
    else:
        src_transformed = dst.copy()
    window_name = 'Select ROI: <y> to submit. <ESC> to reset'

    def on_mouse(event, x, y, flag, param):

        global show_img
        global points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])       # type: ignore
            print('[INFO] Selected', points[-1])

            cv2.circle(show_img, (x, y), radius=10, color=(0, 0, 255), thickness=2)

            if len(points) >= 2:
                # draw on the added image
                cv2.line(show_img, points[-2], points[-1], color=(0, 255, 0), thickness=2)

                # draw in the destination image
                cv2.line(
                    show_img,
                    (int(W1 + points[-2][0] * W2 / W), int(H + points[-2][1] * H2 / H)),
                    (int(W1 + points[-1][0] * W2 / W), int(H + points[-1][1] * H2 / H)),
                    color=(0, 255, 0),
                    thickness=1
                )

                # draw in the source image
                if homo is not None:
                    homo_inv = np.linalg.inv(homo)
                    [[[p1_x, p1_y]], [[p2_x, p2_y]]] = cv2.perspectiveTransform(
                        np.array([points[-2], points[-1]], dtype='float32').reshape(-1, 1, 2),
                        homo_inv
                    )
                    cv2.line(
                        show_img,
                        (int(p1_x * W1 / W), int(H + p1_y * H1 / H)),
                        (int(p2_x * W1 / W), int(H + p2_y * H1 / H)),
                        color=(0, 255, 0),
                        thickness=1
                    )

            cv2.imshow(window_name, show_img)

    while True:
        points = []
        added = np.uint8(src_transformed * 0.5 + dst * 0.5)
        H, W = added.shape[:2]
        H1, W1 = H // 2, W // 2
        H2, W2 = H1, W - W1
        src_scaled = cv2.resize(src if src is not None else dst, (W1, H1))
        dst_scaled = cv2.resize(dst, (W2, H2))

        show_img = np.concatenate(
            [added,
             np.concatenate([src_scaled, dst_scaled], axis=1)],
            axis=0
        )
    
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, show_img)
        cv2.setMouseCallback(window_name, on_mouse)
        key = cv2.waitKey(0)
        if key == 27:
            continue
        elif key == ord('y'):
            if len(points) <= 2:
                print('[WARNING] Need at least 3 points')
                continue
            cv2.line(show_img, points[-1], points[0], color=(0, 255, 0), thickness=2)
            cv2.imshow(window_name, show_img)
            cv2.waitKey(500)
            break
    
    cv2.destroyAllWindows()

    points = np.array(points, dtype='float32')
    points[:, 0] /= dst.shape[1]
    points[:, 1] /= dst.shape[0]
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

    src_pts, dst_pts, H = select_matches(src, dst)
    contour = select_ROI(src, dst, H)

    matches = np.concatenate([src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2)], axis=1)

    if opt['matches_out_path'] is not None:
        
        np.savetxt(opt['matches_out_path'], matches)
        print('[INFO] Matches saved in', opt['matches_out_path'])
    else:
        print('[INFO] Not save matches\n...Printing result to stdout\nH =', matches)

    if opt['roi_out_path'] is not None:
        np.savetxt(opt['roi_out_path'], contour.reshape(-1, 2))
        print('[INFO] Contour saved in', opt['roi_out_path'])
    else:
        print('[INFO] Not save contour\n...Printing result to stdout\ncontour =', contour.reshape(-1, 2))


def main2(opt):

    if opt['video']:
        cap2 = cv2.VideoCapture(opt['dst'])
        dst = extract_frame(cap2)

    else:
        dst = cv2.imread(opt['dst'])

    contour = select_ROI(None, dst, None)

    if opt['roi_out_path'] is not None:
        np.savetxt(opt['roi_out_path'], contour.reshape(-1, 2))
        print('[INFO] Contour saved in', opt['roi_out_path'])
    else:
        print('[INFO] Not save contour\n...Printing result to stdout\ncontour =', contour.reshape(-1, 2))


if __name__ == '__main__':

    # opt = parse_opt()
    
    main(opt)

