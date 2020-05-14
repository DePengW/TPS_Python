import cv2
import numpy as np
import math
import pdb
import os
import argparse
import sys
import torch
import time
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from thinplatespline.batch import TPS
from thinplatespline.tps import tps_warp


def tps_cv2(source, target, img):
	"""
	使用cv2自带的tps处理曲文
    """
	tps = cv2.createThinPlateSplineShapeTransformer()
	
	source_cv2 = source.reshape(1, -1, 2)
	target_cv2 = target.reshape(1, -1, 2)

	matches = list()
	for i in range(0, len(source_cv2[0])):
		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target_cv2, source_cv2, matches)
	new_img_cv2 = tps.warpImage(img)
	
	return new_img_cv2

def tps_torch(source, target, img, DEVICE):
	"""
	使用pyotrch实现的tps处理曲文
    """
	ten_img = ToTensor()(img).to(DEVICE)
	h, w = ten_img.shape[1], ten_img.shape[2]

	ten_source = norm(source, w, h)
	ten_target = norm(target, w, h)
	tpsb = TPS(size=(h, w), device=DEVICE)

	warped_grid = tpsb(ten_target[None, ...], ten_source[None, ...])    #[bs, h, w, 2]（相对）   根据source、target得到的仿射函数，处理图片
	ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, False)
	new_img_torch = np.array(ToPILImage()(ten_wrp[0].cpu()))
	return new_img_torch


def norm(points_int, width, height):
	"""
	将像素点坐标归一化至 -1 ~ 1
    """
	points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
	x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
	y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
	return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)



def Warp_tps_demo(pts, img, output, segment=2):
	"""
	将曲文拉直并裁剪
    """
	pts_up = pts[:len(pts)//2]				
	pts_down = pts[len(pts)//2:][::-1]		
	warp = []
	target_up = []
	target_down = []
	min_distance = []	
	for i in range(len(pts_up)):
		p1 = pts_up[i]
		p2 = pts_down[i]
		delta_p =  np.array(p2)-np.array(p1)
		min_distance.append(math.hypot(delta_p[0], delta_p[1]))

	height = int(min(min_distance))		
	L = 0
	for i in range(len(pts_up)-1):	#通过原点、右点、下点计算目标点
		p1 = np.array(pts_up[i])
		p2 = np.array(pts_up[i+1])
		p3 = np.array(pts_down[i])
		delta_p12 = p2-p1
		delta_p13 = p3-p1
		ratio = math.hypot(delta_p12[0], delta_p12[1])/math.hypot(delta_p13[0], delta_p13[1]) # ratio = horizontal/vertical
		target_up.append((int(L), 0))
		target_down.append((int(L), height))
		L = L + height*ratio
	target_up.append((int(L), 0))
	target_down.append((int(L), height))

	pts_up_ = []
	pts_down_ = []
	target_up_ = []
	target_down_ = []
	for i in range(len(pts_up)-1):
		for j in range(segment):
			x_up = pts_up[i][0] + j*(pts_up[i+1][0]-pts_up[i][0])/segment	#不包括结尾点
			y_up = pts_up[i][1] + j*(pts_up[i+1][1]-pts_up[i][1])/segment
			x_down = pts_down[i][0] + j*(pts_down[i+1][0]-pts_down[i][0])/segment
			y_down = pts_down[i][1] + j*(pts_down[i+1][1]-pts_down[i][1])/segment
			pts_up_.append((int(x_up), int(y_up)))
			pts_down_.append((int(x_down), int(y_down)))

			target_x_up = target_up[i][0] + j*(target_up[i+1][0]-target_up[i][0])/segment
			target_y_up = target_up[i][1] + j*(target_up[i+1][1]-target_up[i][1])/segment
			target_x_down = target_down[i][0] + j*(target_down[i+1][0]-target_down[i][0])/segment
			target_y_down = target_down[i][1] + j*(target_down[i+1][1]-target_down[i][1])/segment
			target_up_.append((int(target_x_up), int(target_y_up)))
			target_down_.append((int(target_x_down), int(target_y_down)))

	pts_up_.append(pts_up[-1])	
	pts_down_.append(pts_down[-1])
	target_up_.append(target_up[-1])
	target_down_.append(target_down[-1])

	pts_up = pts_up_	
	pts_down = pts_down_

	target_up = target_up_
	target_down = target_down_

	pts_up.extend(pts_down)
	target_up.extend(target_down)

	source = np.array(pts_up)	#len(source) = [segment * (len(pts)/2 - 1) + 1] * 2
	target = np.array(target_up)

#--------------------------   cv2  ------------------------

	cv2_img_cut_save_path = output + '/tps_cv2.jpg'
	new_img_cv2 = tps_cv2(source, target, img)
	cv2.imwrite(cv2_img_cut_save_path, new_img_cv2[:height,:int(L),:])

#--------------------------  torch  ------------------------

	torch_img_cut_save_path = output + '/tps_torch.jpg'
	new_img_torch = tps_torch(source, target, img, DEVICE)
	cv2.imwrite(torch_img_cut_save_path, new_img_torch[:height,:int(L),:])

#--------------------------  test time  ------------------------

	# nums = 10
	# t0_cv2 = time.time()
	# for i in range(nums):
	# 	new_img_cv2 = tps_cv2(source, target, img)
	# t1_cv2 = time.time()
	# print("cv2.time : {:.3f}".format((t1_cv2- t0_cv2) / nums))

	# t0_wdp = time.time()
	# for i in range(nums):
	# 	new_img_cv2 = tps_torch(source, target, img, DEVICE)
	# t1_wdp = time.time()
	# print("pytorch_cpu.time : {:.3f}".format((t1_wdp- t0_wdp) / nums))



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Process file and Save result')
	parser.add_argument('--input', default='/demo_data/', type=str, help="image folder path")
	parser.add_argument('--output_path', default='/result', type=str, help="output folder path")

	args = parser.parse_args()


	#-----------------------  处理文件夹下多张图片代码  -----------------------
	# cur_path = os.path.abspath(os.path.dirname(__file__))
	# input_fold = cur_path  + args.input
	# output_path = cur_path + args.output_path

	# input_list = sorted(os.listdir(input_fold))
	# input_img = []
	# input_ctt = []
	# for i in range(len(input_list)):
	# 	if input_list[i].split('.')[-1] == 'jpg':
	# 		input_img.append(input_list[i])
	# 	if input_list[i].split('.')[-1] == 'polyctt':
	# 		input_ctt.append(input_list[i])
	# input_img = sorted(input_img)	#存储img_id
	# input_ctt = sorted(input_ctt)	#存储gt_id

	# for i in range(len(input_img)):
	# 	input_img_path = os.path.join(input_fold, input_img[i])		##存储img_path
	# 	input_ctt_path = os.path.join(input_fold, input_ctt[i])		##存储gt_path

	# 	img = cv2.imread(input_img_path)	#存储img的信息

	# 	data = []	#存储gt的信息
	# 	for line in open(input_ctt_path, "r"):
	# 		temp = line.split(' ')[0]
	# 		tmp = temp.split(',')
	# 		tmp = list(map(int, tmp))
	# 		data.append(tmp)
	# 	points = []		#存储point的信息（n, 4, 2）
	# 	for j in range(len(data)):
	# 		pts = []
	# 		for k in range(len(data[j])//2):
	# 			pts.append((data[j][2*k], data[j][2*k+1]))
	# 		points.append(pts)

	# 	for j in range(len(points)):
	# 		output_file_name = os.path.join(output_path, input_img[i].split('.')[0]+'_%d'%j+'.jpg')
	# 		Warp_tps_demo(points[j], img, output_file_name)
	# 		print(output_file_name)


	#-------------------------处理一张图片代码-------------------------

	# DEVICE = torch.device("cpu")
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	cur_path = os.path.abspath(os.path.dirname(__file__))
	img = cv2.imread(cur_path + '/demo_data/gt_1.jpg')
	txt_path = cur_path + '/demo_data/gt_1.polyctt'

	tmp = ToTensor()(img).to(DEVICE)		#用于提前占显存

	data = []
	for line in open(txt_path, "r"):
		temp = line.split(' ')[0]
		tmp = temp.split(',')
		tmp = list(map(int, tmp))
		data.append(tmp)

	points = []
	for j in range(len(data)):
		pts = []
		for k in range(len(data[j])//2):
			pts.append((data[j][2*k], data[j][2*k+1]))
		points.append(pts)

	j = 2	#选取曲文对应的标注idx

	#图上画出标注点
	for i in range(len(points[j])):
		cv2.circle(img, points[j][i], 3, (0, 255, 0), 2)

	#对曲文做tps并保存结果
	Warp_tps_demo(points[j], img, cur_path + '/result', 4)




