import os
import json
import numpy as np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('indexfile', type=str)
    parser.add_argument('labelfile', type=str)
    parser.add_argument('emfile', type=str)
    parser.add_argument('goodcolumnsfile', type=str)
    parser.add_argument('badcolumnsfile', type=str)
    args = parser.parse_args()

    with open(args.emfile) as fp:
        emdata = json.load(fp)
        emdata = np.array(emdata)

    missing_indices = np.load(args.indexfile)
    missing_labels = np.load(args.labelfile)
    good_columns = np.load(args.goodcolumnsfile)
    bad_columns = np.load(args.badcolumnsfile)

    assert len(good_columns) == emdata.shape[1]
    full_emdata = np.zeros((emdata.shape[0], len(good_columns) + len(bad_columns)))

    for i in range(len(good_columns)):
        full_emdata[:, good_columns[i]] = emdata[:, i]

    if np.ndim(missing_labels) == 1:
        missing_labels = missing_labels[:, np.newaxis]

    correct, count = 0, 0
    for missing_index, missing_label in zip(missing_indices, missing_labels):
        if missing_index[1] in good_columns:
            inferred_label = full_emdata[missing_index[0], missing_index[1]]
            if inferred_label== missing_label[0]:
                correct += 1
        else:
            correct += 1  # everything in bad_columns is auto corrects
        count += 1
    missing_imputation_accuracy = correct / float(count)
    print(missing_imputation_accuracy)
