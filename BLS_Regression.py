import numpy as np
from sklearn import preprocessing
from numpy import random
import time
from data_utils import mean_absolute_percentage_error


def tansig(x):
    """
    Tansig（双曲正切）激活函数。
    将输入值 x 压缩到 -1 到 1 的范围内。
    """
    return (2 / (1 + np.exp(-2 * x))) - 1


def pinv(A, reg):
    """
    计算矩阵 A 的正则化伪逆 (Ridge Regression)。
    用于在宽度学习系统中计算输出层权重。

    Args:
        A (np.ndarray or np.mat): 输入矩阵。
        reg (float): 正则化参数（通常表示为 C 或 lambda）。

    Returns:
        np.mat: 矩阵 A 的正则化伪逆。
    """
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    """
    软阈值函数 (Soft Thresholding Function)。
    在稀疏表示学习中用于引入稀疏性。

    Args:
        a (np.ndarray): 输入数组。
        b (float): 阈值参数。

    Returns:
        np.ndarray: 经过软阈值处理后的数组。
    """
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    """
    用于稀疏化特征映射层权重的迭代算法。
    这是一个基于ADMM (Alternating Direction Method of Multipliers) 的L1范数正则化求解器。

    Args:
        A (np.ndarray): 输入数据矩阵。
        b (np.ndarray): 目标（或残差）矩阵。

    Returns:
        np.ndarray: 稀疏化的权重矩阵 wk。
    """
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]

    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')

    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)

    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk


class BLS:
    """
    宽度学习系统 (Broad Learning System - BLS) 回归模型。
    BLS 通过特征映射层和增强层来构建网络结构，并使用伪逆方法计算输出权重。
    """

    def __init__(self, NumFea, NumWin, NumEnhan):
        """
        初始化BLS模型。

        Args:
            NumFea (int): 每个特征窗口中特征节点的数量。
            NumWin (int): 特征窗口的数量。
            NumEnhan (int): 增强节点的数量。
        """
        self.NumFea = NumFea
        self.NumWin = NumWin
        self.NumEnhan = NumEnhan
        self.WFSparse = []
        self.distOfMaxAndMin = np.zeros(self.NumWin)
        self.meanOfEachWindow = np.zeros(self.NumWin)
        self.WeightEnhan = None
        self.WeightTop = None

    def train(self, train_x, train_y, s, C):
        """
        训练BLS模型。

        Args:
            train_x (np.ndarray): 训练数据的特征，形状为 (样本数, 特征数)。
            train_y (np.ndarray): 训练数据的目标值，形状为 (样本数, 目标维度)。
            s (float): 特征映射层中缩放因子/稀疏化参数。
            C (float): 输出层权重计算中的正则化参数（岭回归）。

        Returns:
            tuple: 包含训练后的模型输出、训练RMSE、训练MAPE和训练时间。
        """
        if train_y.ndim == 1:
            train_y = train_y.reshape(-1, 1)

        WF_initial_list = []
        for i in range(self.NumWin):
            random.seed(i)
            weight_fea = 2 * random.randn(train_x.shape[1] + 1, self.NumFea) - 1
            WF_initial_list.append(weight_fea)

        self.WeightEnhan = 2 * random.randn(self.NumWin * self.NumFea + 1, self.NumEnhan) - 1

        time_start = time.time()

        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])

        y_feature_nodes = np.zeros([train_x.shape[0], self.NumWin * self.NumFea])

        self.WFSparse = []

        for i in range(self.NumWin):
            weight_fea_initial = WF_initial_list[i]
            A1 = H1.dot(weight_fea_initial)

            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1_scaled = scaler1.transform(A1)

            weight_fea_sparse = sparse_bls(A1_scaled, H1).T
            self.WFSparse.append(weight_fea_sparse)

            T1_feature_output = H1.dot(weight_fea_sparse)

            self.meanOfEachWindow[i] = T1_feature_output.mean()
            self.distOfMaxAndMin[i] = T1_feature_output.max() - T1_feature_output.min()

            T1_feature_output_normalized = (T1_feature_output - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]

            y_feature_nodes[:, self.NumFea * i:self.NumFea * (i + 1)] = T1_feature_output_normalized

        # 构建增强层的输入矩阵 H2 (拼接所有特征节点输出 y_feature_nodes 和偏置项)
        H2 = np.hstack([y_feature_nodes, 0.1 * np.ones([y_feature_nodes.shape[0], 1])])
        T2_enhancement_pre_activation = H2.dot(self.WeightEnhan)
        T2_enhancement_output = tansig(T2_enhancement_pre_activation)

        T3_final_input = np.hstack([y_feature_nodes, T2_enhancement_output])

        self.WeightTop = pinv(T3_final_input, C).dot(train_y)

        Training_time = time.time() - time_start

        NetoutTrain = T3_final_input.dot(self.WeightTop)

        RMSE = np.sqrt(np.mean(np.square(NetoutTrain - train_y)))
        MAPE = mean_absolute_percentage_error(train_y, NetoutTrain)

        return NetoutTrain, RMSE, MAPE, Training_time

    def test(self, test_x):
        """
        使用训练好的BLS模型进行预测。

        Args:
            test_x (np.ndarray): 测试数据的特征，形状为 (样本数, 特征数)。

        Returns:
            np.ndarray: 模型的预测输出，形状与训练时的目标维度 train_y 兼容。
        """
        time_start = time.time()

        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])

        yy1_feature_nodes_test = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])

        for i in range(self.NumWin):
            weight_fea_sparse = self.WFSparse[i]
            TT1_feature_output = HH1.dot(weight_fea_sparse)
            TT1_feature_output_normalized = (TT1_feature_output - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1_feature_nodes_test[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1_feature_output_normalized

        HH2 = np.hstack([yy1_feature_nodes_test, 0.1 * np.ones([yy1_feature_nodes_test.shape[0], 1])])
        TT2_enhancement_output_test = tansig(HH2.dot(self.WeightEnhan))

        TT3_final_input_test = np.hstack([yy1_feature_nodes_test, TT2_enhancement_output_test])

        NetoutTest = TT3_final_input_test.dot(self.WeightTop)

        Testing_time = time.time() - time_start

        return NetoutTest