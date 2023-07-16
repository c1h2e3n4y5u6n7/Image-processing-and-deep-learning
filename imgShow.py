import matplotlib.pyplot as plt
import numpy as np
import math
#图像实际大小为 W*100 * H*100 像素  ,
def showImagegray(newImg,oldImg,W,H,newImgtitle,oldImgtitle,saveImgpath):

    plt.figure(figsize=(W,H))
    plt.subplot(121)
    plt.title(oldImgtitle,fontsize=30)
    plt.axis('off')
    plt.imshow(oldImg, cmap='gray')

    plt.subplot(122)
    plt.title(newImgtitle,fontsize=30)
    plt.axis('off')
    plt.imshow(newImg, cmap='gray')
    # plt.tight_layout()  # 调整整体空白
    plt.savefig(saveImgpath)
    plt.show()

def showMultipleimages(imageList,W,H,saveImgpath):

    imageLength=len(imageList)

    plt.rcParams['figure.figsize'] = (W,H)
    col=row=math.ceil(np.sqrt(imageLength))
    fig, a = plt.subplots(col, row)
    m = 0
    for i in range(col):
        for j in range(row):
            a[i][j].set_title(imageList[m][1])
            a[i][j].imshow(imageList[m][0], cmap=plt.cm.gray)
            m += 1
        #去掉边框和刻度
        for ax in a.flat:
            ax.set_axis_off()

    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.2, hspace=0.2)  # 调整子图间距
    plt.savefig(saveImgpath)
    plt.show()


