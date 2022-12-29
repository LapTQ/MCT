import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


def xyxy2xysr(boxes):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    assert boxes.shape[1] == 4, 'Invalid box dimension.'
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x = (boxes[:, 0] + boxes[:, 2]) / 2.
    y = (boxes[:, 1] + boxes[:, 3]) / 2.
    s = w * h
    r = w / np.float32(h)
    return np.squeeze(np.stack([x, y, s, r], axis=1))


def xysr2xyxy(boxes):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    assert boxes.shape[1] == 4, 'Invalid box dimension.'
    w = np.sqrt(boxes[:, 2] * boxes[:, 3])
    h = boxes[:, 2] / w
    x1 = boxes[:, 0] - w/2
    y1 = boxes[:, 1] - h/2
    x2 = x1 + w
    y2 = y1 + h
    return np.squeeze(np.stack([x1, y1, x2, y2], axis=1))


def iou_batch(boxes1, boxes2):
    """
    boxes1: [[x1, y1, x2, y2,...], ...]
    boxes2: [[x1, y1, x2, y2,...], ...]
    return matrix dim: len(boxes1) x len(boxes2)
    """
    boxes1 = np.expand_dims(boxes1, 1)
    boxes2 = np.expand_dims(boxes2, 0)

    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
              + (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) - wh)
    return (o)


def iou_associate(dets, preds, thresh):
    """
    dets: [[x1, y1, x2, y2,...], ...]
    preds: [[x1, y1, x2, y2,...], ...]
    return:
        - matched: [[dets_index, preds_index], ...]
        - unmatched_dets: [dets_index, ...]
        - unmatched_preds: [preds_index, ...]
    """
    if len(dets) == 0 or len(preds) == 0:
        return np.empty((0, 2)), np.arange(len(dets)), np.arange(len(preds))

    iou_matrix = iou_batch(dets, preds)    # dim: len(dets) x len(preds)

    a = np.int32(iou_matrix > thresh)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        m = a
    else:
        # TODO do thoi gian cua ham nay so voi lapjv
        d, p = linear_sum_assignment(-iou_matrix)
        b = np.zeros((len(dets), len(preds)), 'int32')
        b[(d, p)] = 1
        m = a * b

    return np.transpose(np.where(m)), np.where(1 - m.sum(1))[0], np.where(1 - m.sum(0))[0]



if __name__ == '__main__':

    R, C = 10, 7

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # object points: (0,0,0), (1,0,0), ... (6,5,0)
    objp = np.zeros((C * R, 3), 'float32')
    objp[:, :2] = np.mgrid[0:R, 0:C].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # For better results, we need at least 10 test patterns
    import glob
    images = glob.glob('../../data/recordings/2d_v2/frames/frame_27_*.png')
    for fnames in images:
        img = cv2.imread(fnames)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # corners are ordered: left-to-right, top-to-bottom
        ret, corners = cv2.findChessboardCorners(gray, (R, C), None)

        if not ret:
            print('[INFO] Failed')
            continue

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    # increase accuracy
        imgpoints.append(corners2)

        '''
        cv2.drawChessboardCorners(img, (R, C), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    # camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv2.imread('../../data/recordings/2d_v2/frame_27_test_0.png')

    # refine the camera matrix based on a free scaling parameter
    # alpha = 0 -> minimum unwanted pixels
    # alpha = 1 -> retain all pixels with some extra black images
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]

    '''
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite('../../output/undist_2.png', dst)
    
    # Re-projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))





