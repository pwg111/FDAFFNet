# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#==========================dataset load==========================

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image_a, image_b,label = sample['image_a'],sample['image_b'],sample['label']

		h, w = image_a.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img_a = transform.resize(image_a,(self.output_size,self.output_size),mode='constant')
		img_b = transform.resize(image_b, (self.output_size, self.output_size), mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'image_a':img_a,'image_b':img_b,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label = sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl}

class CenterCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
		assert((h >= new_h) and (w >= new_w))

		h_offset = int(math.floor((h - new_h)/2))
		w_offset = int(math.floor((w - new_w)/2))

		image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

		return {'image': image, 'label': label}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'image': image, 'label': label}

# class ToTensor(object):
# 	"""Convert ndarrays in sample to Tensors."""

# 	def __call__(self, sample):

# 		image, label = sample['image'], sample['label']

# 		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
# 		tmpLbl = np.zeros(label.shape)

# 		image = image/np.max(image)
# 		if(np.max(label)<1e-6):
# 			label = label
# 		else:
# 			label = label/np.max(label)

# 		if image.shape[2]==1:
# 			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
# 			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
# 			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
# 		else:
# 			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
# 			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
# 			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

# 		tmpLbl[:,:,0] = label[:,:,0]

# 		# change the r,g,b to b,r,g from [0,255] to [0,1]
# 		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
# 		tmpImg = tmpImg.transpose((2, 0, 1))
# 		tmpLbl = label.transpose((2, 0, 1))

# 		return {'image': torch.from_numpy(tmpImg),
# 			'label': torch.from_numpy(tmpLbl)}

import cv2
def extractEdge(ori_img):
	edge_img = np.zeros(ori_img.shape, dtype='uint8')
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
	eroded = cv2.erode(ori_img.astype(np.uint8)[:,:,0],kernel)        #腐蚀图像
	edge_img[:,:,0] = ori_img.astype(np.uint8)[:,:,0]-eroded
	# edge_img[edge_img>125] = 255
	# edge_img[edge_img<=125] = 255
	return edge_img

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label = sample['image'], sample['label']

		edge_label = extractEdge(label)

		tmpLbl = np.zeros(label.shape)
		tmpEdge = np.zeros(edge_label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if(np.max(edge_label)<1e-6):
			edge_label = edge_label
		else:
			edge_label = edge_label/np.max(edge_label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
#				tmpImg[:,:,0] = (image[:,:,0]-image[:,:,0].mean())
#				tmpImg[:,:,1] = (image[:,:,1]-image[:,:,1].mean())
#				tmpImg[:,:,2] = (image[:,:,2]-image[:,:,2].mean())
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225



		tmpLbl[:,:,0] = label[:,:,0]
		tmpEdge[:,:,0] = edge_label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg),
			'label': torch.from_numpy(tmpLbl),
			'edge':torch.from_numpy(tmpEdge)}
#csx
class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
## 不减均值除方差
	def __call__(self, sample):

		image_a, image_b, label = sample['image_a'], sample['image_b'], sample['label']
		tmpImg_a = np.zeros((image_a.shape[0],image_a.shape[1],3))
		tmpImg_b = np.zeros((image_b.shape[0], image_b.shape[1], 3))
		tmpLbl = np.zeros(label.shape)

		# tmpImg = (image / 255).transpose((2, 0, 1))
		# tmpLbl = label.transpose((2, 0, 1))

		image_a = image_a/np.max(image_a)
		image_b = image_b / np.max(image_b)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# if image.shape[2]==1:
		# 	tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
		# 	tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
		# 	tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		# else:
		# 	tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
		# 	tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
		# 	tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		# tmpLbl[:,:,0] = label[:,:,0]

		# # change the r,g,b to b,r,g from [0,255] to [0,1]
		# #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg_a = image_a.transpose((2, 0, 1))
		tmpImg_b = image_b.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'image_a': torch.from_numpy(tmpImg_a),
				'image_b': torch.from_numpy(tmpImg_b),
			'label': torch.from_numpy(tmpLbl)}
class SalObjDataset(Dataset):
	def __init__(self,img_a_list,img_b_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_a_list = img_a_list
		self.image_b_list = img_b_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_a_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image_a = io.imread(self.image_a_list[idx], plugin='pil')
		image_b = io.imread(self.image_b_list[idx], plugin='pil')


		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image_a.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx], plugin='pil')

		#print("len of label3")
		#print(len(label_3.shape))
		#print(label_3.shape)

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image_a.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image_a.shape) and 2==len(label.shape)):
			image_a = image_a[:,:,np.newaxis]
			image_b = image_b[:, :, np.newaxis]
			label = label[:,:,np.newaxis]

		# #vertical flipping
		# # fliph = np.random.randn(1)
		# flipv = np.random.randn(1)
		#
		# if flipv>0:
		# 	image = image[::-1,:,:]
		# 	label = label[::-1,:,:]
		# #vertical flip

		sample = {'image_a':image_a, 'image_b':image_b,'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample
