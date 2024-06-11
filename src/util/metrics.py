import numpy as np
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import roc_auc_score

from params import CLASSES


def macro_auc(truth, pred, return_per_class=False):
    """
    Computes the macro AUC score.

    Args:
        truth (array-like): Ground truth labels.
        pred (array-like): Predicted probabilities.
        return_per_class (bool, optional): Return AUC scores per class. Defaults to False.

    Returns:
        float or tuple: Macro AUC score or tuple of mean AUC and AUC scores per class.
    """
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
        if truth[:, i].min() != truth[:, i].max():
            auc = roc_auc_score(truth[:, i], pred[:, i])
            aucs.append(auc)
            aucs_per_class[CLASSES[i]] = auc
        else:
            aucs_per_class[CLASSES[i]] = -1

    if return_per_class:
        return np.mean(aucs), aucs_per_class
    return np.mean(aucs)


def get_correlation_matrix(ps, corr="pearson"):
    """
    Computes the correlation matrix.

    Args:
        ps (dict): Dictionary containing predictions.
        corr (str, optional): Type of correlation coefficient. Defaults to "pearson".

    Returns:
        ndarray: Correlation matrix.
    """
    def sub_corr(sub_1, sub_2, corr_fct):
        corrs = []
        for i, c in enumerate(CLASSES):
            if isinstance(sub_1, np.ndarray):
                corr = corr_fct(sub_1[:, i], sub_2[:, i]).statistic
            else:
                corr = corr_fct(sub_1[c], sub_2[c]).statistic
            corrs.append(corr)
        return np.mean(corrs)

    corr_fct = pearsonr if "pearson" in corr else kendalltau
    corrs = np.eye(len(ps))
    for i, k in enumerate(ps.keys()):
        for j, k2 in enumerate(ps.keys()):
            if i > j:
                corr = sub_corr(ps[k], ps[k2], corr_fct)
                corrs[i, j] = corrs[j, i] = corr
    return corrs
