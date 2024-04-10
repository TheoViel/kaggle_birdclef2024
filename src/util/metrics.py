import numpy as np
from sklearn.metrics import roc_auc_score

from params import CLASSES


def macro_auc(truth, pred, return_per_class=False):
    aucs = []
    aucs_per_class = {}

    if isinstance(truth, list):
        truth_ = np.zeros_like(pred)
        for i in range(len(truth)):

            if isinstance(truth[i], str):
                truth_[i, CLASSES.index(truth[i])] = 1
            elif isinstance(truth[i], list) and isinstance(truth[i][0], str):
                for t in truth[i]:
                    truth_[i, CLASSES.index(t)] = 1
            else:
                raise NotImplementedError("Expects list of strings or list of list of strings")
        truth = truth_

    for i in range(pred.shape[1]):
        if truth[:, i].sum():
            auc = roc_auc_score(truth[:, i], pred[:, i])
            aucs.append(auc)
            aucs_per_class[CLASSES[i]] = auc
        else:
            aucs_per_class[CLASSES[i]] = -1

    return np.mean(aucs)
