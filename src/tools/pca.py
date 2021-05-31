# from sklearn.decomposition import PCA
import cv2
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
from src.tools.exponential_smooth import plotExponentialSmoothing, plotDoubleExponentialSmoothing


class PCA(object):
    def __init__(self, percentage):
        self.percentage = percentage
        self.eigVals = None
        self.eigVects = None
        self.n_eigValIndice = None
        self.n_eigVect = None

    @staticmethod
    def zeroMean(dataMat):
        meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
        newData = dataMat - meanVal
        return newData, meanVal

    def percentage2n(self):
        sortArray = np.sort(self.eigVals)  # 升序
        sortArray = sortArray[-1::-1]  # 逆转，即降序
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum * self.percentage:
                return num

    def fit_transform(self, dataMat):
        # old = dataMat.copy()
        # a = dataMat[:, -1].copy()
        # dataMat[:, -1] = dataMat[:, 0]
        # dataMat[:, 0] = a
        newData, meanVal = self.zeroMean(dataMat)
        print(meanVal)
        covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        self.eigVals, self.eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        print(self.eigVals)
        n = self.percentage2n()  # 要达到percent的方差百分比，需要前n个特征向量
        mask = np.zeros_like(dataMat[0, :])
        self.eigValIndice = np.argsort(self.eigVals)  # 对特征值从小到大排序
        self.n_eigValIndice = self.eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        mask[self.n_eigValIndice] = 1
        print(mask)
        print('\n')
        # self.n_eigVect = self.eigVects[:, self.n_eigValIndice]  # 最大的n个特征值对应的特征向量
        # lowDDataMat = newData * self.n_eigVect  # 低维特征空间的数据
        # reconMat = (lowDDataMat * self.n_eigVect.T) + meanVal  # 重构数据
        dataMat = dataMat[:, [8, 7, 6, 5, 4, 3, 2, 1, 0]]
        newData, meanVal = self.zeroMean(dataMat)
        print(meanVal)
        covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        self.eigVals, self.eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        print(self.eigVals)
        n = self.percentage2n()  # 要达到percent的方差百分比，需要前n个特征向量
        mask = np.zeros_like(dataMat[0, :])
        self.eigValIndice = np.argsort(self.eigVals)  # 对特征值从小到大排序
        self.n_eigValIndice = self.eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        mask[self.n_eigValIndice] = 1
        print(mask)
        return mask

    def transform(self, dataMat):
        newData, meanVal = self.zeroMean(dataMat)
        lowDDataMat = newData * self.n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * self.n_eigVect.T) + meanVal  # 重构数据
        return reconMat

    # def fit_transform(self, dataMat):
    #     newData, meanVal = self.zeroMean(dataMat)
    #     covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    #     eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    #     n = self.percentage2n()  # 要达到percent的方差百分比，需要前n个特征向量
    #     eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    #     n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    #     n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    #     lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    #     reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    #     return lowDDataMat, reconMat

# def get_static_frames_idx_pca(angle_vecs, dim, interval=1, DEBUG=False, video_name=''):
#     estimator = PCA(n_components=dim)
#     static_frames_idx = []
#     meanVals = np.mean(angle_vecs, axis=0)
#     meanRemoved = angle_vecs - meanVals
#     pca_angle_vecs = estimator.fit_transform(meanRemoved)
#     plot_pca_angle_curve(pca_angle_vecs, dim)
#     plotExponentialSmoothing(pca_angle_vecs, [0.5, 0.1], save_path='../../../tmp2.png')
#     plotDoubleExponentialSmoothing(pca_angle_vecs, alphas=[0.5, 0.3], betas=[0.9, 0.3], save_path='../../../tmp1.png')
#     return static_frames_idx
#
#
# def plot_pca_angle_curve(angle_vecs, dim, static_frames_idx=None, save_path='tmp.png'):
#     plt.title('PCA Joint Angle Sequence')
#     for i in range(dim):
#         plt.plot(angle_vecs[:, i])
#     if static_frames_idx is not None:
#         for idx in static_frames_idx:
#             plt.vlines(idx, -np.pi/2, 3*np.pi/2, colors="r", linestyles="dashed")
#     plt.xlabel('Frame ID')
#     plt.ylabel('Angle (rad)')
#     plt.savefig(save_path)
#     plt.clf()
