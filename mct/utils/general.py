import numpy as np
from typing import Union, Tuple
from scipy.optimize import linear_sum_assignment


def calc_loc(X, loc_infer_mode: int, mid: Union[Tuple[Union[float, int]], None] = None) -> np.ndarray:
    
    # if detection mode == 'box' => [[frame_id, track_id, x1, y1, w, h, -1, -1, -1, 0],...] (N x 10)
    # if detection mode == 'pose' => [[frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1, *[kpt_x, kpt_y, kpt_conf], ...]] (N x 61)
    # return [[x, y], ...]
    
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    X = X.copy()
    assert len(X.shape) == 2, 'Invalid shape, must be (N, M)'

    # TODO
    if loc_infer_mode == 1:     # box, midpoint of box's bottom edge
        X[:, 2] += X[:, 4] / 2  # type: ignore
        X[:, 3] += X[:, 5]
        return X[:, 2:4]
    
    elif loc_infer_mode == 2:   # box, intersection of box's bottom edge with the segment from box's center to the midpoint image's bottom edge
        assert mid is not None
        X = X[:, 2:6]
        X[:, 2:4] += X[:, :2]
        ret = np.empty(shape=(len(X), 2), dtype='float32')
        mx, my = mid                                                    # type: ignore
        for i, xyxy in enumerate(X):
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2

            a = (xyxy[2] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and xyxy[1] <= y <= xyxy[3]:
                ret[i] = xyxy[2:4]
                continue

            a = (xyxy[3] - cy) / (my - cy)
            x = a * mx + (1 - a) * cx
            if 0 <= a <= 1 and xyxy[0] <= x <= xyxy[2]:
                ret[i] = [x, xyxy[3]]
                continue

            a = (xyxy[0] - cx) / (mx - cx)
            y = a * my + (1 - a) * cy
            if 0 <= a <= 1 and xyxy[1] <= y <= xyxy[3]:
                ret[i] = xyxy[[0, 3]]
        return ret

    elif loc_infer_mode == 3:   # pose

        ret = np.empty(shape=(len(X), 2), dtype='float32')
        boxes, kpts = X[:, 2:6], X[:, 10:61]
        i = [[15, 13, 11, 5], [16, 14, 12, 6]]
        
        for j, (box, kpt) in enumerate(zip(boxes, kpts)):
            locs = [kpt[i*3: i*3 + 2] for i in range(17)]
            conf = kpt[[2 + 3*i for i in range(17)]]
            f = [[], []]
            
            for c in range(2):
                if conf[i[c][0]] >= 0.5:
                    if conf[i[c][2]] >= 0.5:
                        f[c].append(locs[i[c][0]] + 1/8.5 * (locs[i[c][0]] - locs[i[c][2]]))
                    if conf[i[c][1]] >= 0.5:
                        f[c].append(locs[i[c][0]] + 1/4.5 * (locs[i[c][0]] - locs[i[c][1]]))
                if conf[i[c][1]] >= 0.5:
                    if conf[i[c][2]] >= 0.5:
                        f[c].append(locs[i[c][1]] + 5/4.7 * (locs[i[c][1]] - locs[i[c][2]]))
                    if conf[i[c][3]] >= 0.5:
                        f[c].append(locs[i[c][1]] + 2/4.8 * (locs[i[c][1]] - locs[i[c][3]]))
                if conf[i[c][2]] >= 0.5:
                    if conf[i[c][3]] >= 0.5:
                        f[c].append(locs[i[c][2]] + 4/3 * (locs[i[c][2]] - locs[i[c][3]]))
                
                if len(f[c]) == 0 and conf[i[c][0]] >= 0.5:
                    f[c].append(locs[i[c][0]])

            bx1, by1, bw, bh = box
            if len(f[0]) > 0  and len(f[1]) > 0:
                f[0] = np.mean(f[0], axis=0)
                f[1] = np.mean(f[1], axis=0)
                ret[j] = (f[0] + f[1]) / 2
            elif len(f[0]) == len(f[1]) == 0:
                ret[j] = (bx1 + bw/2, by1 + bh)
            else:
                xy = f[0] if len(f[0]) > 0 else f[1]
                x, y = np.mean(xy, axis=0)
                ret[j] = (x + bx1 + bw/2) / 2, y
            
        return ret
        
    raise NotImplementedError()


def hungarian(cost, gate=None):
    """Wrapper for linear_sum_assignment.

    If gate[i, j] == 0, do not match i and j.
    """
    if gate is None:
        gate = np.ones_like(cost, dtype='int32')
    assert cost.shape == gate.shape, 'cost and gate must be of the same size'
    cost = cost.copy()
    outliers = cost[gate == 0]
    thresh = min(outliers) if len(outliers) > 0 else 1e9    # 1e9 instead of Inf due to linear_sum_assignment does not accept maxtrix full of Inf
    cost = np.where(cost < thresh, cost, 1e9)
    i_matches, j_matches = linear_sum_assignment(cost)
    i_matches_new, j_matches_new = [], []
    for i, j in zip(i_matches, j_matches):
        if cost[i, j] < thresh: # do not use gate[i, j] == 1
            i_matches_new.append(i)
            j_matches_new.append(j)
    return i_matches_new, j_matches_new


def map_mono(A, B, diff_thresh=None):
    """Monotonic mapping function.

    All params must be of the same time unit (i.e all are in second, or all are in milisecond)
    """

    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(B, np.ndarray):
        B = np.array(B)

    assert len(A.shape) == len(B.shape) == 1, 'Invalid seq dimension, must be array rank 0 (N,)'

    if diff_thresh is None:
        diff_thresh = float('inf')

    n1 = len(A)
    n2 = len(B)

    M1 = []
    M2 = []
    D = []

    Ai = np.int32(np.round(np.interp(A, B, np.arange(n2))))

    for i1 in range(n1):
        i2 = Ai[i1]             # type: ignore
        d = abs(A[i1] - B[i2])
        
        if len(M2) == 0 or i2 != M2[-1]:
            M1.append(i1)
            M2.append(i2)
            D.append(d)
        elif d < D[-1]:
            M1[-1] = i1
            D[-1] = d
    
    idx = np.array(D) <= diff_thresh        # type: ignore
    M1 = np.array(M1)[idx]                  # type: ignore
    M2 = np.array(M2)[idx]                  # type: ignore

    return M1, M2


def create_global_id_mapper(scts, matches):
    """assuming [[[c1, c2], ...], [[c2, c3], ...], [[c3, c4], ...], ...]"""

    global_ids_mapper = [{} for _ in range(len(matches) + 1)]
    global_id_count = 0

    table = np.full((sum([len(m) for m in matches]), len(matches) + 1), -1, dtype=int)
    i = 0
    for j, m in enumerate(matches):
        for id1, id2 in m:
            table[i, j] = id1
            table[i, j + 1] = id2
            i += 1

    for i_out in range(table.shape[0]):
        rows = []
        k = 0
        if table[i_out, 0] != -1:
            rows.append((i_out, table[i_out].copy()))
            table[i_out] = -1
        while k < len(rows):
            i, row = rows[k]
            for j, id in enumerate(row):
                if id != -1:
                    for new_i in np.where(table[:, j] == id)[0]:
                        if i != new_i:
                            rows.append((new_i, table[new_i].copy()))
                        table[new_i] = -1
            k += 1

        if len(rows) > 0:
            global_id_count += 1
            for row in rows:
                for j, id in enumerate(row[1]):
                    if id != -1:
                        global_ids_mapper[j][id] = global_id_count

    for i, sct in enumerate(scts):
        for id in np.unique(sct[:, 1]).astype('int32'):
            if id not in global_ids_mapper[i]:
                global_id_count += 1
                global_ids_mapper[i][id] = global_id_count

    print(global_ids_mapper)

    return global_ids_mapper