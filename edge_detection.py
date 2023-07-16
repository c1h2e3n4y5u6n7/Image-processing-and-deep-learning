import numpy as np
import cv2
import imgShow as iS
def SobelX(img,threshold):
    height = img.shape[0]
    width = img.shape[1]
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    result = np.zeros(img.shape)
    for i in range(0, width - 2):
        for j in range(0, height - 2):
            v = np.sum(G_x * img[i:i + 3, j:j + 3])
            result[i,j] =v
            if(result[i,j]<threshold):
                result[i,j]=0
    return result
def SobelY(img,threshold):
    height = img.shape[0]
    width = img.shape[1]
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    result = np.zeros(img.shape)
    for i in range(0, width - 2):
        for j in range(0, height - 2):
            h = np.sum(G_y * img[i:i + 3, j:j + 3])
            result[i,j] =h
            if(result[i,j]<threshold):
                result[i,j]=0
    return result

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



def Laplacian(img):
    temLaplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    height, width = img.shape[::-1]
    result = np.zeros(img.shape)
    for i in range(0, width - 2):
        for j in range(0, height - 2):
            result[i][j] = np.abs(np.sum(temLaplacian * img[i:i + 3, j:j + 3]))
    return result

img=cv2.imread("./originImg/HorizontalAndVertical.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sobelImg=Sobel(img,56)
# iS.showImagegray(sobelImg, img, 25, 15, 'sobelDetection', 'origin', './ProcessedImg/sobelDetection.jpg')
imageList=[]
origin_img=[img,'origin_img']
imageList.append(origin_img)
sobelx=SobelX(img,0)
sobel2=[sobelx,'Sobel_X']
imageList.append(sobel2)
sobely=SobelY(img,0)
sobel1=[sobely,'Sobel_Y']
imageList.append(sobel1)
sobelImg=Sobel(img,56)
sobel3=[sobelImg,'Sobel']
imageList.append(sobel3)
iS.showMultipleimages(imageList,25,25,'./ProcessedImg/sobelEdge.jpg')
img1=cv2.imread('./originImg/Goldhill.tif')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
LapImg=Laplacian(img1)
iS.showImagegray(LapImg, img1, 25, 15, 'LapImg', 'origin', './ProcessedImg/lapImg.jpg')

# cv2.imshow('sobely',sobely)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

