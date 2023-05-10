import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


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
