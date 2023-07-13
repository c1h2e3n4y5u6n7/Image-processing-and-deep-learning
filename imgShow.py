import matplotlib.pyplot as plt
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
