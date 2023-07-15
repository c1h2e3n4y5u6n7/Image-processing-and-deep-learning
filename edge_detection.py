import numpy as np
import cv2
def Sobel(img,threshold):
    height = img.shape[0]
    width = img.shape[1]
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    result = np.zeros(img.shape)
    for i in range(0, width - 2):
        for j in range(0, height - 2):
            v = np.sum(G_x * img[i:i + 3, j:j + 3])
            h = np.sum(G_y * img[i:i + 3, j:j + 3])
            result[i,j] = np.sqrt((v ** 2) + (h ** 2))
            if(result[i,j]<threshold):
                result[i,j]=0
    return result
img=cv2.imread("./originImg/Lena.tif")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sobelImg=Sobel(img,90)
cv2.imshow('sobelImg',sobelImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

