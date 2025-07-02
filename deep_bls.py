import numpy as np
import matplotlib.pyplot as plt
from BLS_Regression import BLS, mean_absolute_percentage_error


class DeepBLS(object):
    """
    深度宽度学习系统（DeepBLS）模型，用于回归任务。
    DeepBLS是一个序贯集成模型，它通过迭代地训练多个宽度学习系统（BLS）来学习并修正前一个BLS的预测残差。
    这类似于梯度提升的思想，旨在逐步提高模型的预测精度。
    """

    def __init__(self,
                 max_iter=50,
                 learn_rate=0.01,
                 new_BLS_max_iter=10,
                 NumFea=20,
                 NumWin=5,
                 NumEnhan=50,
                 s=0.8,
                 C=2 ** -30):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.new_BLS_max_iter = new_BLS_max_iter
        self.NumFea = NumFea
        self.NumWin = NumWin
        self.NumEnhan = NumEnhan
        self.s = s
        self.C = C
        self.dBLSs = []
        self.residual_mean = None
        self.cumulated_pred_score = None

    def fit(self, x_train, y_train):
        """
        训练DeepBLS模型。
        模型通过序贯训练多个BLS，每个BLS学习前一阶段预测的残差（误差），
        并将其贡献（乘以学习率）添加到总预测中。

        Args:
            x_train (np.ndarray): 训练数据的特征，形状为 (样本数, 特征数)。
            y_train (np.ndarray): 训练数据的目标值。可以是形状为 (样本数,) 的一维数组，
                                  或形状为 (样本数, 目标维度) 的二维数组。
        """
        n_samples = x_train.shape[0]
        target_dim = y_train.shape[1] if y_train.ndim > 1 else 1

        self.residual_mean = np.zeros(self.max_iter)
        loss = np.zeros(self.max_iter)

        dBLS_initial = BLS(self.NumFea, self.NumWin, self.NumEnhan)
        dBLS_initial.train(x_train, y_train, self.s, self.C)
        self.dBLSs.append(dBLS_initial)

        f = dBLS_initial.test(x_train)
        if y_train.ndim == 1 and f.ndim > 1 and f.shape[1] == 1:
            f = f.flatten()

        loss[0] = mean_absolute_percentage_error(y_train, f)

        for iter_ in range(1, self.max_iter):
            y_predict_current = f
            residual = y_train - y_predict_current

            dBLS_new = BLS(self.NumFea, self.NumWin, self.NumEnhan)
            dBLS_new.train(x_train, residual * self.learn_rate, self.s, self.C)
            self.dBLSs.append(dBLS_new)

            new_bls_pred = dBLS_new.test(x_train)
            if y_train.ndim == 1 and new_bls_pred.ndim > 1 and new_bls_pred.shape[1] == 1:
                new_bls_pred = new_bls_pred.flatten()
            f += new_bls_pred

            loss[iter_] = mean_absolute_percentage_error(y_train, f)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(loss)), loss, label='训练损失')
        plt.title("DeepBLS模型每层BLS的训练损失")
        plt.xlabel("BLS迭代次数")
        plt.ylabel("损失 (MAPE)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def predict(self, x):
        """
        使用DeepBLS模型进行预测。
        最终预测值是所有训练好的BLS模型对输入数据预测值的累加和。

        Args:
            x (np.ndarray): 需要进行预测的特征数据，形状为 (样本数, 特征数)。

        Returns:
            np.ndarray: 模型的最终预测值，形状与训练时y_train的形状兼容
                        （即 (样本数,) 或 (样本数, 目标维度)）。
        """
        n_samples = x.shape[0]
        first_bls_output = self.dBLSs[0].test(x)

        if first_bls_output.ndim == 1 or first_bls_output.shape[1] == 1:
            y_individual_bls_preds = np.zeros([n_samples, len(self.dBLSs)])
        else:
            y_individual_bls_preds = np.zeros([n_samples, len(self.dBLSs), first_bls_output.shape[1]])

        for iter_ in range(len(self.dBLSs)):
            dBLS = self.dBLSs[iter_]
            current_prediction = dBLS.test(x)

            if current_prediction.ndim == 1 or current_prediction.shape[1] == 1:
                y_individual_bls_preds[:, iter_] = current_prediction.flatten()
            else:
                y_individual_bls_preds[:, iter_, :] = current_prediction

        self.cumulated_pred_score = np.cumsum(y_individual_bls_preds, axis=1)

        final_predictions = np.sum(y_individual_bls_preds, axis=1)

        if self.dBLSs and self.dBLSs[0].test(x).ndim == 1:
            final_predictions = final_predictions.flatten()

        return final_predictions