
import os
import time
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

from thinplatespline.batch import TPS
from thinplatespline.tps import tps_warp
from thinplatespline.utils import (
    TOTEN, TOPIL, grid_points_2d, noisy_grid, grid_to_img)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


cur_path = os.path.abspath(os.path.dirname(__file__))
img = Image.open(cur_path + "/boris_johnson.jpg")
w, h = img.size                             #[512, 738]

dense_grid = grid_points_2d(w, h, DEVICE)   #[377856, 2]    [wh, 2]
X = grid_points_2d(7, 11, DEVICE)           #[77, 2]    X:是规则的点，即target(归一化后)
Y = noisy_grid(7, 11, 0.15, DEVICE)         #[77, 2]    Y:是加入扰动的点, 即source

x1, y1 = grid_to_img(X, w, h)               #target在图片中对应的像素位置
x2, y2 = grid_to_img(Y, w, h)               #source在图片中对应的像素位置


#-----------------------opencv2-----------------------------

targetpoints = torch.stack([torch.from_numpy(x1), torch.from_numpy(y1)], dim=-1)
sourcepoints = torch.stack([torch.from_numpy(x2), torch.from_numpy(y2)], dim=-1)

tps = cv2.createThinPlateSplineShapeTransformer()

matches =[]
for i in range(1, sourcepoints.shape[0]+1):
    matches.append(cv2.DMatch(i,i,0))

sourcepoints = sourcepoints.numpy().reshape(1,-1,2)
targetpoints = targetpoints.numpy().reshape(1,-1,2)

tps.estimateTransformation(targetpoints, sourcepoints, matches)
img_cv=tps.warpImage(np.array(img))
img_cv = Image.fromarray(img_cv)
img_cv.save(cur_path + '/img_result/cv.png')

#-----------------------pytorch-----------------------------

tpsb = TPS(size=(h, w), device=DEVICE)

warped_grid_b = tpsb(X[None, ...], Y[None, ...])    #[1, 738, 512, 2]（归一化）   根据XY得到的仿射函数，处理图片
ten_img = TOTEN(img).to(DEVICE)                     #[3, 738, 512]
ten_wrp_b = torch.grid_sampler_2d(ten_img[None, ...], warped_grid_b, 0, 0, False)   #[1, 738, 512, 2]
img_torch = TOPIL(ten_wrp_b[0].cpu())

img_torch.save(cur_path + "/img_result/ptorch.png")


#绘制图像
fig, ax = plt.subplots(1, 3, figsize=[13, 7], sharey=True)
ax[0].imshow(img)
ax[0].plot(x1, y1, "+g", ms=15, mew=2, label="origin")
ax[0].legend(loc=1)
ax[1].plot(x2, y2, "+r", ms=15, mew=2, label="cv2")
ax[1].imshow(img_cv)
ax[1].legend(loc=1)
ax[2].plot(x2, y2, "+r", ms=15, mew=2, label="pytorch")
ax[2].imshow(img_torch)
ax[2].legend(loc=1)
plt.tight_layout()
fig.savefig(cur_path + "/img_result/plot.jpg", bbox_inches="tight")
