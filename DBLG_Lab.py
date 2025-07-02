# coding=UTF-8
import os
import numpy as np
import pandas as pd
import torch
import time
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import GPy

import statsmodels.api as sm

from data_utils import *
from deep_bls import DeepBLS
from Lab_GAN import build_generator, build_discriminator, build_gan, train_gan

# --- 文件路径配置 ---
# A3C模型的状态和奖励文件路径。
STATE_FILE = "state-200_NN200.mat"
REWARD_FILE = "reward-200_NN200.mat"


# --- 数据加载与预处理辅助函数 ---

def getXlabel():
    """
    生成X坐标标签列表（从1到21的字符串形式）。
    例如：['1', '2', ..., '21']
    """
    return ['%d' % (i + 1) for i in range(21)]


def getYlabel():
    """
    生成Y坐标标签列表（从01到23的字符串形式，个位数会补零）。
    例如：['01', '02', ..., '09', '10', ..., '23']
    """
    return ['0%d' % (j + 1) if j < 9 else '%d' % (j + 1) for j in range(23)]


def getOriginalCSI():
    """
    加载原始CSI（信道状态信息）数据，进行重塑，并收集相应的坐标标签。
    数据从预设路径下的.mat文件中读取。

    Returns:
        tuple: 包含以下元素的元组：
            - originalCSI (np.ndarray): 重塑后的CSI数据，形状为 (样本数, 3 * 30 * 1500)。
            - label (np.ndarray): 每个CSI样本对应的标签（x，y坐标），形状为 (样本数, 2)。
            - count (int): 实际加载的CSI样本数量。
    """
    xLabel = getXlabel()
    yLabel = getYlabel()
    originalCSI = np.zeros((317, 3 * 30 * 1500), dtype=np.float32)
    label = np.empty((0, 2), dtype=np.int32)
    newName = []
    count = 0

    for i in range(21):
        for j in range(23):
            filePath = f"D:/DBLG/47SwapData/coordinate{xLabel[i]}{yLabel[j]}.mat"
            if os.path.isfile(filePath):
                csi = loadmat(filePath)['myData']
                originalCSI[count, :] = np.reshape(csi, (1, -1))
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                newName.append(f"{xLabel[i]}{yLabel[j]}")
                count += 1
    return originalCSI[:count, :], label, count


def generatePilot():
    """
    从原始CSI数据中生成引导（pilot）数据。
    引导数据是原始数据的子集，用于高斯过程回归的训练。

    Returns:
        tuple: 包含以下元素的元组：
            - pilotLabel (np.ndarray): 引导数据的坐标标签。
            - pilotCSI (np.ndarray): 引导CSI数据。
    """
    originalCSI, label, _ = getOriginalCSI()
    originalData = np.array(originalCSI[:, 0:2 * 40 * 800:2000], dtype=np.float32)
    originalData = SimpleImputer(strategy='mean').fit_transform(originalData)

    rng = np.random.RandomState(20)
    labelIndex = np.sort(rng.randint(1, 317, size=32))

    return label[labelIndex], originalData[labelIndex, :]


def findIndex(label, pathPlan):
    """
    根据给定的路径规划点，查找这些点在所有标签中的对应索引。

    Args:
        label (np.ndarray): 所有CSI样本的坐标标签，形状为 (样本数, 2)。
        pathPlan (list): 路径规划中包含坐标点的列表，例如 [[x1, y1], [x2, y2]]。

    Returns:
        list: 包含每个路径点对应CSI样本索引的列表。
    """
    index = []
    for p in pathPlan:
        x_match_indices = np.where(label[:, 0] == p[0])[0]
        y_match_indices = np.where(label[:, 1] == p[1])[0]
        common_indices = list(set(x_match_indices).intersection(y_match_indices))
        if common_indices:
            index.append(common_indices)
    return [idx for sublist in index for idx in sublist if sublist]


def filterProcess(data, n_iter=2):
    """
    对CSI预测数据进行两阶段滤波和平滑处理：
    1. 卡尔曼滤波：用于状态估计和噪声抑制。
    2. 巴特沃斯低通滤波：进一步平滑数据。

    Args:
        data (np.ndarray): 需要滤波的CSI数据。
                          形状为 (样本数, CSI特征数)。
        n_iter (int): 卡尔曼滤波的EM算法迭代次数。

    Returns:
        np.ndarray: 经过滤波和平滑处理后的CSI数据，形状与输入相同。
    """
    from pykalman import KalmanFilter

    bufferCSI = np.zeros_like(data, dtype=np.float32)

    b, a = butter(2, 3 * 2 / 50, 'lowpass')

    for i in range(len(data)):
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        kf = kf.em(data[i], n_iter=n_iter)
        state_means, _ = kf.filter(data[i])
        filtered = filtfilt(b, a, state_means[:, 0])
        bufferCSI[i, :] = filtered
    return bufferCSI


def findPossiblePath(stateFile):
    """
    从A3C训练的状态文件中加载所有可能的路径，并筛选出那些包含特定起点和终点的路径。

    Args:
        stateFile (str): 状态文件 (.mat) 的名称（不含路径，路径在函数内部拼接）。

    Returns:
        tuple: 包含以下元素的元组：
            - possiblePath (list): 包含筛选出的有效路径列表，每个路径是坐标点列表。
            - stateLabel (list): 对应这些路径在原始状态文件中的索引。
    """
    state = loadmat(f"D:/DBLG/Fifth code Lab/{stateFile}")['array']
    stateList = state.reshape((100, 200, 2))
    possiblePath, stateLabel = [], []

    for i in range(100):
        path = [list(pt) for pt in set(tuple(x) for x in stateList[i])]
        path.sort()

        if [1, 1] in path and [21, 23] in path:
            possiblePath.append(path)
            stateLabel.append(i)
    return possiblePath, stateLabel


def OptimalPath(rewardFile):
    """
    根据A3C算法学习到的奖励值，从所有可能的路径中选择具有最高奖励值（即最优）的路径。

    Args:
        rewardFile (str): 奖励文件 (.mat) 的名称（不含路径，路径在函数内部拼接）。

    Returns:
        tuple: 包含以下元素的元组：
            - optimal_path (list): 具有最高奖励值的最优路径。
            - max_reward (float): 最优路径对应的最大奖励值。
    """
    possiblePath, stateLabel = findPossiblePath("state-200_NN200.mat")
    reward = loadmat(f"D:/DBLG/Fifth code Lab/{rewardFile}")['array'][0]

    rewards = [reward[i] for i in stateLabel]
    max_idx = np.argmax(rewards)

    return possiblePath[max_idx], rewards[max_idx]


# --- 主执行流程 ---
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- 1. 数据加载与引导数据生成 ---
    originalCSI, label, _ = getOriginalCSI()
    pilotLabel, pilotCSI = generatePilot()

    # --- 2. 多元高斯过程回归 (GPR) 建模与预测 ---
    mean_pilot_label = np.mean(pilotLabel, axis=1)
    covMatrix_pilot_csi = np.cov(pilotCSI)

    kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)
    omega_kernel = kernelRBF.K(pilotLabel, pilotLabel)
    kernelPilot_combined = covMatrix_pilot_csi @ omega_kernel

    np.random.seed(0)
    mulGauProPrediction = np.random.multivariate_normal(mean_pilot_label, kernelPilot_combined, size=len(label))

    # --- 3. 滤波与平滑处理 ---
    bufferCSI = filterProcess(mulGauProPrediction)

    meanError_pilot_csi = np.mean(pilotCSI, axis=0)
    model_sarimax = sm.tsa.SARIMAX(meanError_pilot_csi, order=(1, 0, 0), trend='c').fit(disp=False)
    errorBand = model_sarimax.get_prediction().conf_int()

    # --- 4. 误差带约束与Savitzky-Golay平滑修正 ---
    for i in range(len(bufferCSI)):
        sw = int(np.clip(np.abs(bufferCSI[i] - errorBand[:, 0]).argmin(), 5, 25))
        sw = sw if sw % 2 == 1 else sw - 1
        bufferCSI[i] = savgol_filter(bufferCSI[i], sw, polyorder=2)

    # --- 5. 数据整合与预处理（DeepBLS和GAN的输入准备） ---
    pathPlan, _ = OptimalPath(REWARD_FILE)

    index_path = np.array(findIndex(label, pathPlan)).flatten()
    index_pilot = np.array(findIndex(label, pilotLabel)).flatten()
    merged_index = np.sort(np.unique(np.append(index_path, index_pilot)))

    secondpilotCSI = originalCSI[merged_index]
    secondpilotCSI = pd.DataFrame(secondpilotCSI).replace([0, np.inf, -np.inf], np.nan).fillna(method='ffill').values

    secondpilotCSI_32 = secondpilotCSI[:, 0:2 * 40 * 800:2000]
    merged_labels = label[merged_index]

    # --- 6. DeepBLS 模型训练与预测 ---
    train_x, test_x, train_y, test_y = train_test_split(
        secondpilotCSI_32, merged_labels, test_size=0.2, random_state=10
    )
    model_deep_bls = DeepBLS(max_iter=50, learn_rate=0.01, new_BLS_max_iter=10,
                             NumFea=20, NumWin=5, NumEnhan=50, s=0.8, C=2 ** -30)
    model_deep_bls.fit(train_x, train_y)
    final_bls_predictions = model_deep_bls.predict(originalCSI[:, 0:2 * 40 * 800:2000])

    # --- 7. GAN生成缺失点CSI进行数据增强 ---
    unique_rows = np.unique(np.concatenate((label, merged_labels), axis=0), axis=0, return_counts=False)
    remainder_label_list = [tuple(row) for row in label.tolist() if
                            tuple(row) not in {tuple(mr) for mr in merged_labels.tolist()}]
    remainder_label = np.array(remainder_label_list)

    latent_dim = 1000

    generator = build_generator(latent_dim, 2, secondpilotCSI.shape[1])
    discriminator = build_discriminator(secondpilotCSI.shape[1], 2)
    gan = build_gan(generator, discriminator, latent_dim, 2)

    train_gan(generator, discriminator, gan, secondpilotCSI, merged_labels,
              latent_dim=latent_dim, epochs=50, batch_size=32)

    noise_for_generation = np.random.normal(0, 1, (len(remainder_label), latent_dim))
    generated_data = generator.predict([noise_for_generation, remainder_label])
    adjusted_generated_data = (
                                          generated_data - generated_data.mean()) / generated_data.std() * secondpilotCSI.std() + secondpilotCSI.mean()

    # --- 8. 合并GAN生成数据与原始数据，并进行最终融合 ---
    total_data_gan_combined = np.concatenate((secondpilotCSI, adjusted_generated_data), axis=0)
    total_labels_gan_combined = np.concatenate((merged_labels, remainder_label), axis=0)

    sorted_idx = np.lexsort((total_labels_gan_combined[:, 1], total_labels_gan_combined[:, 0]))

    sorted_gan_enhanced_csi_reduced = total_data_gan_combined[sorted_idx][:, 0:2 * 40 * 800:2000]

    thirdFinger = 0.4 * sorted_gan_enhanced_csi_reduced + 0.6 * final_bls_predictions

    # --- 9. KNN定位测试与性能评估 ---
    X_train, X_test, y_train, y_test = train_test_split(thirdFinger, label, test_size=0.2, random_state=3)

    KNN = KNeighborsRegressor(n_neighbors=12).fit(X_train, y_train)

    t0 = time.time()
    prediction = KNN.predict(X_test)
    prediction_time = time.time() - t0

    avg_error = np.mean(np.linalg.norm(prediction - y_test, axis=1) * 0.5)
    std_error = np.std(np.linalg.norm(prediction - y_test, axis=1) * 0.5)

    print(f"平均定位误差: {avg_error:.2f} m")
    print(f"定位误差标准差: {std_error:.2f} m")
    print(f"KNN预测所需时间: {prediction_time:.2f} s")