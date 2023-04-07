import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_macro(y_true, y_pred):
    """
    Macro MAE, returns MAE for each class found in y_true and its macro
    aggregation.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    labs = np.sort(y_true.unique())
    mae_metrics = {}
    sum_mae = 0
    for lab in labs:
        y_true_sub = y_true[y_true == lab]
        y_pred_sub = y_pred[y_true == lab]
        mae = mean_absolute_error(y_true_sub, y_pred_sub)
        sum_mae += mae
        mae_metrics[str(lab)] = mae
    mae_metrics["macro avg"] = sum_mae / len(labs)
    return mae_metrics


def rmse_macro(y_true, y_pred):
    """
    Macro RMSE, returns RMSE for each class found in y_true and its macro
    aggregation.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    labs = np.sort(y_true.unique())
    rmse_metrics = {}
    sum_mse = 0
    for lab in labs:
        y_true_sub = y_true[y_true == lab]
        y_pred_sub = y_pred[y_true == lab]
        mse = mean_squared_error(y_true_sub, y_pred_sub)
        sum_mse += mse
        rmse_metrics[str(lab)] = np.sqrt(mse)
    rmse_metrics["macro avg"] = np.sqrt(sum_mse / len(labs))
    return rmse_metrics


if __name__ == "__main__":
    y_true = self.y
    y_pred = self.y_pred
    pass
