import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    roc_auc_score, precision_score, matthews_corrcoef, cohen_kappa_score, average_precision_score
from imblearn.metrics import sensitivity_score, specificity_score
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score, concordance_index_ipcw


def ordinal_metrics(y_true, y_pred, alpha=0.5):
    """
    Compute ordinal-aware metrics:
      - Acc: Exact accuracy
      - AdjAcc: Accuracy within ±1 grade
      - NAER: Non-adjacent error rate (>1 grade apart)
      - OAS: Ordinal accuracy score (partial credit for adjacent errors)
      - MAE: Mean absolute error in grade units
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True ordinal labels (ints)
    y_pred : array-like, shape (n_samples,)
        Predicted ordinal labels (ints)
    alpha : float, optional (default=0.5)
        Partial credit for adjacent errors in OAS.
    Returns
    -------
    dict
        Dictionary with metrics.
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Absolute distance in ordinal scale
    dist = np.abs(y_pred - y_true)

    # Exact accuracy
    acc = np.mean(dist == 0)

    # Adjacent accuracy (exact or ±1)
    adj_acc = np.mean(dist <= 1)

    # Non-adjacent error rate (> ±1)
    naer = np.mean(dist > 1)

    # Ordinal Accuracy Score
    oas = np.mean(np.where(dist == 0, 1,
                  np.where(dist == 1, alpha, 0)))

    # Mean absolute error
    mae = np.mean(dist)

    return {
        "Acc": acc,
        "AdjAcc": adj_acc,
        "NAER": naer,
        f"OAS(alpha={alpha})": oas,
        "MAE": mae
    }


def compute_cls_metrics(ground_truth, activations, avg='micro', demical_places=4):

    ground_truth = ground_truth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)

    multi_class = 'ovr'
    ill_avg = avg
    # For binary classification
    if activations.shape[1] == 2:
        activations = activations[:, 1]
        multi_class = 'raise'
        # binary average is illegal for auc
        ill_avg = None
        avg = 'binary'

    mean_acc = accuracy_score(y_true=ground_truth, y_pred=predictions)
    f1 = f1_score(y_true=ground_truth, y_pred=predictions, average=avg)

    try:
        auc = roc_auc_score(y_true=ground_truth, y_score=activations, multi_class=multi_class, average=ill_avg)
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    try:
        ap = average_precision_score(y_true=ground_truth, y_score=activations, average=ill_avg)
    except Exception as error:
        print('Error in computing AP. Error msg:{}'.format(error))
        ap = 0
    bac = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
    sens = sensitivity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    spec = specificity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    prec = precision_score(y_true=ground_truth, y_pred=predictions, average=avg)
    mcc = matthews_corrcoef(y_true=ground_truth, y_pred=predictions)
    kappa = cohen_kappa_score(y1=ground_truth, y2=predictions)

    metrics = {'Accuracy': mean_acc, 'F1': f1, 'AUC': auc, 'AP': ap, 'BAC': bac,
               'Sensitivity': sens, 'Specificity': spec, 'Precision': prec, 'MCC': mcc, 'Kappa': kappa}

    metrics = {k: round(v, demical_places) for k, v in metrics.items()}
    return metrics


def compute_surv_metrics(train_surv, test_surv, risk_prob, surv_prob, times):
    # ---- Plain C-index (full data) ----
    cindex, *_ = concordance_index_censored(
        test_surv["event"], test_surv["time"], risk_prob
    )

    # ---- IPCW-safe test set ----
    test_ipcw, risk_ipcw, surv_ipcw = test_surv, risk_prob, surv_prob
    censor_mask = ~test_ipcw["event"]

    if censor_mask.any():
        max_censor_time = test_ipcw["time"][censor_mask].max()
        drop_mask = (test_ipcw["event"]) & (test_ipcw["time"] >= max_censor_time)
        if drop_mask.any():
            print(f"\n[Warning] Dropping {drop_mask.sum()} event(s) at/after censoring max={max_censor_time} for IPCW-based metrics.")
            test_ipcw = test_ipcw[~drop_mask]
            risk_ipcw = risk_ipcw[~drop_mask]
            surv_ipcw = surv_ipcw[~drop_mask]

        # ensure times fit within new support
        max_time_ipcw = test_ipcw["time"].max()
        times = times[times < max_time_ipcw]

    # ---- IPCW C-index ----
    cindex_ipcw, *_ = concordance_index_ipcw(train_surv, test_ipcw, risk_ipcw)

    # ---- Integrated Brier Score ----
    ibs = integrated_brier_score(train_surv, test_ipcw, surv_ipcw, times)

    # ---- Time-dependent AUC ----
    auc, mean_auc = cumulative_dynamic_auc(train_surv, test_ipcw, 1 - surv_ipcw, times)
    if np.isnan(mean_auc):
        raise ValueError("AUC is NaN — check survival probabilities or time points.")

    # ---- Collect metrics ----
    metrics = {
        "C-index": cindex,
        "C-index IPCW": cindex_ipcw,
        "AUC": mean_auc,
        "IBS": ibs,
    }
    return metrics
