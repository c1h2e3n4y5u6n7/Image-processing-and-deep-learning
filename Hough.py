import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import mpl_toolkits.axisartist as axisartist

def lines_detector_hough(img,ThetaDim=None, DistStep=None, threshold=None, halfThetaWindowSize=2,
                         halfDistWindowSize=None):
    '''

    :param img: 经过边缘检测得到的二值图
    :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
    :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)和对应的索引值,
    '''
    row,col= edge.shape
    if ThetaDim == None:
        ThetaDim = 90
    if DistStep == None:
        DistStep = 1
    # 计算距离分段数量
    MaxDist = np.sqrt(row ** 2 + col ** 2)
    DistDim = int(np.ceil(MaxDist / DistStep))

    if halfDistWindowSize == None:
        halfDistWindowSize = int(DistDim /50)

    # 建立投票
    accumulator = np.zeros((ThetaDim, DistDim))  # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射
    #
    sinTheta = [np.sin(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    #计算距离（rho）
    for i in range(row):
        for j in range(col):
            if not edge[i, j] == 0:
                for k in range(ThetaDim):
                    accumulator[k][int(round((i * cosTheta[k] + j * sinTheta[k]) * DistDim / MaxDist))] += 1
    M = accumulator.max()
#---------------------------------------
    #非极大抑制
    if threshold == None:
        threshold = int(M * 1.369/ 10)
    result = np.array(np.where(accumulator > threshold))  # 阈值化
    #获得对应的索引值
    temp = [[], []]
    for i in range(result.shape[1]):
        eight_neiborhood = accumulator[
                           max(0, result[0, i] - halfThetaWindowSize + 1):min(result[0, i] + halfThetaWindowSize,
                                                                              accumulator.shape[0]),
                           max(0, result[1, i] - halfDistWindowSize + 1):min(result[1, i] + halfDistWindowSize,
                                                                             accumulator.shape[1])]
        if (accumulator[result[0, i], result[1, i]] >= eight_neiborhood).all():
            temp[0].append(result[0, i])
            temp[1].append(result[1, i])
    #记录原图所检测的坐标点（x,y）
    result_temp= np.array(temp)
#-------------------------------------------------------------
    result = result_temp.astype(np.float64)
    result[0] = result[0] * np.pi / ThetaDim
    result[1] = result[1] * MaxDist / DistDim
    return result,result_temp

def drawLines(lines, edge, color=(255, 0, 0), err=3):

    '''
    :param lines: 检测后的直线参数
    :param edge: 原图
    :param color: 直线的颜色
    :param err:检测的可接受的误差值
    :return: 无
    '''

    if len(edge.shape) == 2:
        result = np.dstack((edge, edge, edge))
    else:
        result = edge
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any():
                result[i, j] = color
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()

def data_img(data):

    '''
    :param data: 直线上含有的点（x,y）
    :return: 输出hough空间图像
    '''

    fig = plt.figure()  # 新建画布
    ax = axisartist.Subplot(fig, 111)  # 使用axisartist.Subplot方法创建一个绘图区对象ax
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)  # 隐藏原来的实线矩形
    ax.axis["x"] = ax.new_floating_axis(0, 0, axis_direction="bottom")  # 添加x轴
    ax.axis["y"] = ax.new_floating_axis(1, 0, axis_direction="bottom")  # 添加y轴
    ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加箭头
    ax.axis["y"].set_axisline_style("->", size=1.0)  # 给y坐标轴加箭头
    t = np.arange(-np.pi / 2, np.pi / 2, 0.1)
    ax.annotate(text='x', xy=(2 * math.pi, 0), xytext=(2 * math.pi, 0.1))  # 标注x轴
    ax.annotate(text='y', xy=(0, 1.0), xytext=(-0.5, 1.0))  # 标注y轴
    for i in range(data.shape[1]):
        rho = data[0][i] * np.cos(t) + data[1][i] * np.sin(t)
        plt.plot(t, rho)
    plt.show()


orgImg=cv2.imread('./originImg/line3.jpg')
#去噪
blurred = cv2.GaussianBlur(orgImg, (3, 3), 0)
#灰度化
gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
#canny提取边缘
edge = cv2.Canny(gray, 50, 150)
cv2.imshow("edge",edge)
cv2.waitKey(0)
lines,data = lines_detector_hough(edge)
drawLines(lines, blurred)
data_img(data)


