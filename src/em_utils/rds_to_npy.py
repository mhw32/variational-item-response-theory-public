import numpy as np
from scipy.ndimage.filters import gaussian_filter

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.numpy2ri import numpy2ri


def rds_to_np(Rfile):
    """ 
    Convert .RData to be able to load the array into Python.
    Rfile := (str) location of file to translate to Python.
    """ 
    raw = robjects.r['readRDS'](Rfile)
    return raw


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rds_path', type=str, help='path to .rds file')
    parser.add_argument('npy_path', type=str, help='path to .npy file')
    args = parser.parse_args()

    data = rds_to_np(args.rds_path)
    data = np.array(data)
    np.save(args.npy_path, data)
