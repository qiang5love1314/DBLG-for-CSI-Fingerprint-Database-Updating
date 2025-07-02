import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():

    "-----Lab------"
    x1, y1 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Predict-Lab-Error.mat')
    x2, y2 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Forest-Lab-Error.mat')
    x3, y3 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/TNSE-Lab-Error.mat')
    x4, y4 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/TMC-Lab-Error.mat')

    figure, ax = plt.subplots()
    plt.step(x1, y1, color='r', marker='o', label='DBLG')
    plt.step(x2, y2, color='green', marker='x', label='ILM-CFBCS')
    plt.step(x3, y3, color='c', marker='p', label='GA-IPP')
    plt.step(x4, y4, color='orange', marker='+', label='A3C-IPP')

    # "-----Meeting room------"
    # x1, y1 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Predict-Meet-Error.mat')
    # x2, y2 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Forest-Meet-Error.mat')
    # x3, y3 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/TNSE-Meet-Error.mat')
    # x4, y4 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/TMC-Meet-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color = 'r', marker ='o', label='DBLG')
    # plt.step(x2, y2, color='green', marker='x', label='ILM-CFBCS')
    # plt.step(x3, y3, color='c', marker='p', label='GA-IPP')
    # plt.step(x4, y4, color='orange', marker='+', label='A3C-IPP')

    # "-----Predict and Original database------"
    # x1, y1 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Predict-Lab-Error.mat')
    # x2, y2 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Original-Lab-Error.mat')
    # x3, y3 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Predict-Meet-Error.mat')
    # x4, y4 = read(r'D:/DBLG/Experiments DBLG/Positioning Performance Analysis and Comparative Experiments/Original-Meet-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color='r', marker='o', label='Area One-Predictive')
    # plt.step(x2, y2, color='b', marker='v', label='Area One-Manual')
    # plt.step(x3, y3, color='green', marker='x', label='Area Two-Predictive')
    # plt.step(x4, y4, color='c', marker='p', label='Area Two-Manual')

    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams.update({'font.size': 15})
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Average Localization Error (m)', size=15)
    plt.ylabel('Cumulative Distribution', size=15)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend( loc = 'lower right')
    # plt.rcParams.update({'font.size': 15})
    plt.tight_layout()
    plt.savefig('Lab-累计误差（多种方法）.pdf', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main()
    pass