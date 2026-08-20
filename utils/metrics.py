import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def smape(y_true, y_pred, eps=1e-8):
    """对称平均绝对百分比误差，对接近 0 的值更稳定。"""
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_metrics(y_true, y_pred, target_names=None):
    """计算整体指标和每个目标变量的指标。"""
    metrics = {
        "overall_rmse": rmse(y_true, y_pred),
        "overall_mae": mae(y_true, y_pred),
        "overall_mape": mape(y_true, y_pred),
        "overall_smape": smape(y_true, y_pred),
        "overall_r2": r2(y_true, y_pred),
    }

    if target_names is not None:
        for i, name in enumerate(target_names):
            metrics[f"{name}_rmse"] = rmse(y_true[:, i], y_pred[:, i])
            metrics[f"{name}_mae"] = mae(y_true[:, i], y_pred[:, i])
            metrics[f"{name}_mape"] = mape(y_true[:, i], y_pred[:, i])
            metrics[f"{name}_smape"] = smape(y_true[:, i], y_pred[:, i])
            metrics[f"{name}_r2"] = r2(y_true[:, i], y_pred[:, i])

    return metrics
