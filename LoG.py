import numpy as np
import cv2
from matplotlib import pyplot as plt
import imgShow as iS

#定义掩膜
m1 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]]) #LoG算子模板
img = cv2.imread("./originImg/Lena.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#边缘扩充
image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
# image = cv2.GaussianBlur(img,(3,3),4)
rows = image.shape[0]
cols = image.shape[1]
temp = 0
image1 = np.zeros(image.shape)

for i in range(2,rows-2):
    for j in range(2,cols-2):
        temp = np.abs(
            (np.dot(np.array([1, 1, 1, 1, 1]), (m1 * image[i - 2:i + 3, j - 2:j + 3])))
                .dot(np.array([[1], [1], [1], [1], [1]])))
        image1[i,j] = int(temp)
        if image1[i, j] > 255:
            image1[i, j] = 255
        else:
            image1[i, j] = 0
iS.showImagegray(image1,img , 25, 15, 'LoG', 'origin', ' ./ProcessedImg/LoG.jpg')
# cv2.imshow("LoG",image1)
# cv2.waitKey(0)
