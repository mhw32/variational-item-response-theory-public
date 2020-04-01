import os
import json
import numpy as np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('indexfile', type=str)
    parser.add_argument('labelfile', type=str)
    parser.add_argument('emfile', type=str)
    args = parser.parse_args()

    with open(args.emfile) as fp:
        emdata = json.load(fp)
        emdata = np.array(emdata)

    missing_indices = np.load(args.indexfile)
    missing_labels = np.load(args.labelfile)

    if np.ndim(missing_labels) == 1:
        missing_labels = missing_labels[:, np.newaxis]

    correct, count = 0, 0
    for missing_index, missing_label in zip(missing_indices, missing_labels):
        inferred_label = emdata[missing_index[0], missing_index[1]]
        if inferred_label== missing_label[0]:
            correct += 1
        count += 1
    missing_imputation_accuracy = correct / float(count)
    print(missing_imputation_accuracy)
