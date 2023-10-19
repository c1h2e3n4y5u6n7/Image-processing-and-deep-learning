import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_patch(img,patch_size):
    imgs = []
    shape_list = []
    h, w, n = img.shape
    new_h, new_w = patch_size, patch_size
    col = int(w / patch_size) + 1
    row = int(h / patch_size) + 1
    patch_n = col * row
    top = 0
    for r in range(row):
        insert_zeros_row = np.empty((0, 0, 3))
        foot = top + new_h
        if foot > h:
            foot_gap = foot - h
            foot = h
            insert_zeros_row = np.zeros((foot_gap, new_w,3))
        left = 0
        for c in range(col):
            insert_zeros_col = np.empty((0, 0, 3))
            right = left + new_w
            if right > w:
                right_gap = right - w
                right = w
                if r==row-1:
                    insert_zeros_col = np.zeros((new_h-foot_gap, right_gap, 3))
                else:
                    insert_zeros_col = np.zeros((new_h, right_gap, 3))

            img_patch = img[top:foot, left:right]
            shape_list.append(img_patch.shape)
            if insert_zeros_col.shape[0] > 0:
                img_patch = np.hstack((img_patch, insert_zeros_col))
                insert_zeros_col = np.empty((0, 0, 3))
            if insert_zeros_row.shape[0] > 0:
                img_patch = np.vstack((img_patch, insert_zeros_row))
            left = left + new_w
            imgs.append(img_patch)
        top = top + new_h
    return imgs,shape_list, row, col
def jointImage(imgs,shape_list,h_n,w_n,):
    for h in range(h_n):
        # 按行拼接
        for w in range(w_n):
            # 按列拼接img
            if w==0:
               imgs_c=np.array(imgs[h*w_n+w][:shape_list[h*w_n+w][0],:shape_list[h*w_n+w][1]])
            else:
               img_c=np.array(imgs[h*w_n+w][:shape_list[h*w_n+w][0],:shape_list[h*w_n+w][1]])
               imgs_c=np.hstack((imgs_c,img_c))
            # print(imgs[h * w_n + w].shape)
        if h==0:
            imgs_h=imgs_c
        else:
            imgs_h=np.vstack((imgs_h,imgs_c))
    return imgs_h

img=cv.imread('./originImg/lane.png')
imgs,shape_list,h_n,w_n=get_patch(img,128)
# print(imgs)
# print(shape_list)
plt.rcParams["figure.figsize"] = [10, 8]
fig, axes = plt.subplots(nrows=h_n, ncols=w_n)
num=0
for r in range(h_n):
    for c in range(w_n):
        axes[r, c].imshow(imgs[num]/255)
        num+=1
# plt.tight_layout()
plt.subplots_adjust(bottom=-.1, right=0.5, top=.8)
plt.show()
full_img=jointImage(imgs,shape_list,h_n,w_n)
plt.title('full_img')
plt.imshow(full_img/255)
plt.show()
# print('origh',img.shape)
# print('joint',full_img.shape)