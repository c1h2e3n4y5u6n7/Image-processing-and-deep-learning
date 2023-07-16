import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
import imgShow as iS
import random
# 去除噪音 - 使用 5x5 的高斯滤波器
def gaussianSmooth(img_gray):
    # 1.生成高斯滤波器/高斯核
    """
    要生成一个 (2k+1)x(2k+1)的高斯滤波器，滤波器的各个元素计算公式如下：2*k+1=5,k=2,
    H[i, j] = (1/(2\*pi\*sigma\*\*2))\*exp(-1/2\*sigma\*\*2((i-k-1)\*\*2 + (j-k-1)\*\*2))
    """
    sigma = 25
    gau_sum = 0
    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp((-1 / (2 * sigma * sigma)) * (np.square(i - 2) + np.square(j - 2)))
            gau_sum = gau_sum + gaussian[i, j]
    # 2.高斯滤波器归一化处理
    gaussian = gaussian / gau_sum
    print(gaussian)

    # 3.高斯滤波
    W, H = img_gray.shape
    new_gray = np.zeros([W , H])
    for i in range(W):
        for j in range(H):
            if(i<W-5 and j<H-5):
                new_gray[i,j] = np.sum(img_gray[i:i + 5, j:j + 5] * gaussian)
            else:
                new_gray[i,j] =img_gray[i,j]
    return new_gray


def gauss_noise(img, mean=0, sigma=25):
    image = np.array(img / 255, dtype=float)  # 将原始图像的像素值进行归一化
    # 创建一个均值为mean，方差为sigma，呈高斯分布的图像矩阵
    noise = np.random.normal(mean, sigma / 255.0, image.shape)
    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    res_img = np.clip(out, 0.0, 1.0)
    res_img = np.uint8(res_img * 255.0)
    return res_img


img = cv2.imread('originImg/PeppersRGB.tif')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noiseImg=gauss_noise(grayImg)
smoothImg = gaussianSmooth(grayImg)
iS.showImagegray(smoothImg, noiseImg, 25, 15, 'smoothImg', 'origin', './GaussianSmooth.jpg')
GaussianBlur_opencv=cv2.GaussianBlur(noiseImg,(5,5),25)
iS.showImagegray(GaussianBlur_opencv,noiseImg , 25, 15, 'GaussianBlur_opencv', 'origin', './GaussianSmooth_Opencv.jpg')