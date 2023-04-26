import numpy as np
from scipy.optimize import linear_sum_assignment


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


def map_timestamp(ATT, BTT, diff_thresh=None, return_matrix=True):
    """Heuristic mapping function.

    all params must be of the same time unit (i.e all are in second, or all are in milisecond)
    """

    T1 = len(ATT)
    T2 = len(BTT)

    if diff_thresh is None:
        diff_thresh = float('inf')

    if not isinstance(ATT, np.ndarray):
        ATT = np.array(ATT)
    if not isinstance(BTT, np.ndarray):
        BTT = np.array(BTT)

    assert len(ATT.shape) == len(BTT.shape) == 1, 'Invalid seq dimension, must be array rank 0 (N,)'

    X = np.zeros((T1, T2), dtype='int32')

    valid_pairs = [(abs(ATT[i] - BTT[j]), i, j)
                   for i in range(T1)
                   for j in range(T2)
                   if abs(ATT[i] - BTT[j]) <= diff_thresh]
    valid_pairs = sorted(valid_pairs)
    M1 = []
    M2 = []

    def _is_crossing(i, j):
        for i_optimal, j_optimal in zip(*np.where(X)):
            if (ATT[i] - ATT[i_optimal])*(BTT[j] - BTT[j_optimal]) <= 0:
                return True
        return False

    for _, i, j in valid_pairs:
        if not np.any(X[i, :]) and not np.any(X[:, j]) and not _is_crossing(i, j):
            X[i, j] = 1
            M1.append(i)
            M2.append(j)
    
    if return_matrix:
        return X
    else:
        return M1, M2

