import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 确保输入数组的形状相同
    if y_true.shape != y_pred.shape:
        raise ValueError("The shapes of y_true and y_pred must be the same.")

    # 忽略除零和无效值
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)
        ape[~np.isfinite(ape)] = 0  # 将NaN和inf替换为零

    return np.mean(ape) * 100  # 返回百分比形式的平均绝对误差


import numpy as np


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    temp = 0
    # Convert inputs to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure the shapes of y_true and y_pred match
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred do not match")

    # Get the total number of elements
    total_elements = y_true.size

    # Flatten the arrays to handle both 1D and 2D arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    for i in range(total_elements):
        if np.abs(y_true_flat[i]) + np.abs(y_pred_flat[i]) != 0:
            temp += 100 * np.abs(y_true_flat[i] - y_pred_flat[i]) / (
                        (np.abs(y_true_flat[i]) + np.abs(y_pred_flat[i])) / 2)

    return temp / total_elements

# def mean_absolute_percentage_error(y_true, y_pred):
#     temp = 0
#     for i in range(0, y_true.size):
#         if y_true[i] != 0:
#             temp += np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100
#         else:
#             if y_pred[i] == y_true[i]:
#                 temp += 0
#             else:
#                 temp += np.abs((y_true[i] - y_pred[i]) / y_pred[i]) * 100
#     return temp * 1.0 / y_true.size
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#
#     # Ignore zero division and invalid values
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ape = np.abs((y_true - y_pred) / y_true)
#         ape[~np.isfinite(ape)] = 0  # Replace NaN and inf with zero
#
#     return np.mean(ape) * 100



# def symmetric_mean_absolute_percentage_error(y_true, y_pred):
#     temp = 0
#     for i in range(0, y_true.size):
#         if np.abs(y_true[i]) + np.abs(y_pred[i]) != 0:
#             temp += 100 * np.abs(y_true[i] - y_pred[i]) / ((np.abs(y_true[i]) + np.abs(y_pred[i])) / 2)
#     return temp * 1.0 / y_true.size

