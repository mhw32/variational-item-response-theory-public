import os
import json
import numpy as np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('indexfile', type=str)
    parser.add_argument('labelfile', type=str)
    parser.add_argument('emfile', type=str)
    parser.add_argument('goodcolumnsfile1', type=str)
    parser.add_argument('badcolumnsfile1', type=str)
    parser.add_argument('goodcolumnsfile2', type=str)
    parser.add_argument('badcolumnsfile2', type=str)
    parser.add_argument('goodrows', type=str)
    parser.add_argument('badrows', type=str)
    args = parser.parse_args()

    with open(args.emfile) as fp:
        emdata = json.load(fp)
        emdata = np.array(emdata)

    missing_indices = np.load(args.indexfile)
    missing_labels = np.load(args.labelfile)
    good_columns1 = np.load(args.goodcolumnsfile1)
    bad_columns1 = np.load(args.badcolumnsfile1)
    good_columns2 = np.load(args.goodcolumnsfile2)
    bad_columns2 = np.load(args.badcolumnsfile2)
    good_rows = np.load(args.goodrows)
    bad_rows = np.load(args.badrows)

    emdata_step1 = np.zeros((emdata.shape[0], len(good_columns2) + len(bad_columns2)))

    for i in range(len(good_columns2)):
        emdata_step1[:, good_columns2[i]] = emdata[:, i]

    emdata_step2 = np.zeros((emdata.shape[0], len(good_columns1) + len(bad_columns1)))

    for i in range(len(good_columns1)):
        emdata_step2[:, good_columns1[i]] = emdata_step1[:, i]

    emdata_step3 = np.zeros((len(good_rows) + len(bad_rows), emdata_step2.shape[1]))
    for i in range(len(good_rows)):
        emdata_step3[good_rows[i], :] = emdata_step2[i, :]

    if np.ndim(missing_labels) == 1:
        missing_labels = missing_labels[:, np.newaxis]

    correct, count = 0, 0
    true_labels, pred_labels = [], []
    for missing_index, missing_label in zip(missing_indices, missing_labels):
        inferred_label = emdata_step3[missing_index[0], missing_index[1]]
        if inferred_label == missing_label[0]:
            correct += 1
        count += 1
        pred_labels.append(inferred_label)
        true_labels.append(missing_label)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    missing_imputation_accuracy = correct / float(count)
    print(missing_imputation_accuracy)
