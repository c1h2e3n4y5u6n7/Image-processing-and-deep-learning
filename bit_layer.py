import cv2
import numpy as np
import matplotlib.pyplot as plt


#分层处理过程
def  getBitlayer(img):
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w, 8))
    for i in range(h):
        for j in range(w):
            n = str(np.binary_repr(img[i, j], 8))
            for k in range(8):
                new_img[i, j, k] = n[k]
    return new_img


## 通过plt子图形式显示每层bit图
def showBitlayer(new_img):
    # 调整图像大小 实际大小为w*100,h*100 pixel
    plt.rcParams['figure.figsize'] = (10, 3.6)
    fig, a = plt.subplots(2, 4)
    m = 0
    n = 8
    for i in range(2):
        for j in range(4):

            a[i][j].set_title('Bit plane ' + str(n))
            a[i][j].imshow(new_img[:, :, m], cmap=plt.cm.gray)
            m += 1
            n -= 1
        #去掉边框和刻度
        for ax in a.flat:
            ax.set_axis_off()

    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.5, hspace=-0.2)  # 调整子图间距
    plt.savefig('bitLayer.jpg')
    plt.show()


## bite图像重构
def rebuildImg(bitImags, build_list, img):
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            for m in build_list:
                new_img[i, j] += bitImags[i, j, 7-m] * (2 ** (m))
    return new_img

def showRebuildimgimg(rebuildImag,img):
    plt.figure(figsize=(100, 20))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(rebuildImag, cmap='gray')
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread(r'dollars.tif', 0)
    bit_imgs=getBitlayer(img)
    showBitlayer(bit_imgs)
    rebuildImg=rebuildImg(bit_imgs,[5,6,7],img)
    showRebuildimgimg(rebuildImg,img)

