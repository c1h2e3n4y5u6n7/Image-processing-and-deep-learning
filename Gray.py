import cv2
import numpy as np
import matplotlib.pyplot as plt

#图像灰度化
#最大值法
def grayBymax(img):
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)
    # 图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            # 获取图像R G B最大值
            gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
            grayimg[i, j] = np.uint8(gray)
    return grayimg


def grayByaverage(img):
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)
    # 图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            # 获取图像R G B最大值
            gray = (int(img[i,j][0])+int(img[i,j][1])+int(img[i,j][2]))/3
            grayimg[i, j] = np.uint8(gray)
    return grayimg


def grayByWeightedaverage(img):
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            # 获取图像R G B最大值
            gray = 0.30 * img[i, j][0] + 0.59 * img[i, j][1] + 0.11 * img[i, j][2]
            grayimg[i, j] = np.uint8(gray)
    return grayimg


if __name__ == '__main__':
    img = cv2.imread('originImg/miao.jpg')
    opencv_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayByaverage=grayByaverage(img)
    grayBymax=grayBymax(img)
    grayByWeightedaverage=grayByWeightedaverage(img)
    # # 显示图像
    # cv2.imshow("src", img)
    # cv2.imshow("gray", opencv_gray)
    # # 等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(80, 80))
    plt.rcParams['font.sans-serif'] = [
        'FangSong']  # 支持中文标签
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(imgRGB)
    ax[0,1].imshow(grayByaverage)
    ax[1,0].imshow(grayBymax)
    ax[1,1].imshow(grayByWeightedaverage)
    ax[0,0].title.set_text("原图")
    ax[0,1].title.set_text("平均值法")
    ax[1,0].title.set_text("最大值法")
    ax[1,1].title.set_text("加权平均值法")
    ax[0,0].axis("off")
    ax[0,1].axis("off")
    ax[1,0].axis("off")
    ax[1,1].axis("off")
    fig.tight_layout()
    plt.savefig("gray.jpg")
    plt.show()
