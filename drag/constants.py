"This module defines some lbm constants that are used."
import numpy as np

# Velocity vectors
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
# Their weights
w = np.array([ 4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Cardinality to index
idx_C = 0
idx_E = 1
idx_N = 2
idx_W = 3
idx_S = 4
idx_NE = 5
idx_NW = 6
idx_SW = 7
idx_SE = 8

# Mirror of indices
idx_M = {
    idx_C: idx_C,
    idx_E: idx_W,
    idx_N: idx_S,
    idx_W: idx_E,
    idx_S: idx_N,
    idx_NE: idx_SW,
    idx_NW: idx_SE,
    idx_SW: idx_NE,
    idx_SE: idx_NW,
}

# Collections of all, row, and column indices
idx_All = np.arange(9)
col_E = np.array([idx_NE, idx_E, idx_SE])
col_C = np.array([idx_N, idx_C, idx_S])
col_W = np.array([idx_NW, idx_W, idx_SW])
row_N = np.array([idx_NE, idx_N, idx_NW])
row_C = np.array([idx_E, idx_C, idx_W])
row_S = np.array([idx_SE, idx_S, idx_SW])

# The same collections but their mirrors
idx_Allm = np.array([idx_M[i] for i in idx_All])
col_Em = np.array([idx_M[i] for i in col_E])
col_Cm = np.array([idx_M[i] for i in col_C])
col_Wm = np.array([idx_M[i] for i in col_W])
row_Nm = np.array([idx_M[i] for i in row_N])
row_Cm = np.array([idx_M[i] for i in row_C])
row_Sm = np.array([idx_M[i] for i in row_S])
