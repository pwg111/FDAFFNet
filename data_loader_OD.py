from __future__ import print_function, division
import glob
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from skimage import io, transform, color
import numpy as np
import math
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import os
import shutil



def packBs_both(inputs_A_test,inputs_B_test, labels_test, size):
    labelans = []
    imageans_A = []
    imageans_B = []
    for i, labelpath in enumerate(labels_test[0]):
        image_A = cv2.imread(inputs_A_test[i])
        image_B = cv2.imread(inputs_B_test[i])
        w, h = image_A.shape[1], image_A.shape[0]
        if w == h:
            image_A = cv2.resize(image_A, (size, size))
            image_B = cv2.resize(image_B, (size, size))
            image_A = image_A.transpose(2, 0, 1).astype(np.float32)
            image_B = image_B.transpose(2, 0, 1).astype(np.float32)
            for n in range(0,image_A.shape[0]):
                image_A[n,:,:] = image_A[n,:,:] / np.max(image_A[n,:,:])
                image_B[n, :, :] = image_B[n, :, :] / np.max(image_B[n, :, :])
            imageans_A.append(image_A)
            imageans_B.append(image_B)
            f = open(labelpath)
            lines = f.read()
            labellines = lines.split("\n")
            for labelline in labellines:
                labellinesans = [0, 0, 0.0, 0.0, 0.0, 0.0]
                if labelline != "":
                    labellinesans[0] = i
                    labellinesans[1:6] = labelline.split(" ")
                    labelans.append(labellinesans)
            f.close()
            del f
        elif w > h:
            image0_A = np.zeros((size, size, image_A.shape[2]), dtype=np.float32)
            image0_B = np.zeros((size, size, image_B.shape[2]), dtype=np.float32)
            image_A = cv2.resize(image_A, (size, int(h * size / w)))
            image_B = cv2.resize(image_B, (size, int(h * size / w)))
            image0_A[0:int(h * size / w), :, :] = image_A
            image0_B[0:int(h * size / w), :, :] = image_B
            image_A = image_A.transpose(2, 0, 1)
            image_B = image_B.transpose(2, 0, 1)
            for n in range(0, image_A.shape[0]):
                image_A[n, :, :] = image_A[n, :, :] / np.max(image_A[n, :, :])
                image_B[n, :, :] = image_B[n, :, :] / np.max(image_B[n, :, :])
            imageans_A.append(image_A)
            imageans_B.append(image_B)
            f = open(labelpath)
            lines = f.read()
            labellines = lines.split("\n")
            for labelline in labellines:
                labellinesans = [0, 0, 0.0, 0.0, 0.0, 0.0]
                if labelline != "":
                    labellinesans[0] = i
                    labellinesans[1:6] = labelline.split(" ")
                    labelans.append(labellinesans)
            f.close()

                # labelans.append(Variable(torch.from_numpy(np.array(labellinesans).astype(float))).cuda())
        elif h > w:
            image0_A = np.zeros((size, size, image_A.shape[2]), dtype=np.float32)
            image0_B = np.zeros((size, size, image_B.shape[2]), dtype=np.float32)
            image_A = cv2.resize(image_A, (int(w * size / h), size))
            image_B = cv2.resize(image_B, (int(w * size / h), size))
            image0_A[:, 0:int(w * size / h), :] = image_A
            image0_B[:, 0:int(w * size / h), :] = image_B
            image_A = image_A.transpose(2, 0, 1)
            image_B = image_B.transpose(2, 0, 1)
            for n in range(0, image_A.shape[0]):
                image_A[n, :, :] = image_A[n, :, :] / np.max(image_A[n, :, :])
                image_B[n, :, :] = image_B[n, :, :] / np.max(image_B[n, :, :])
            imageans_A.append(image_A)
            imageans_B.append(image_B)
            f = open(labelpath)
            lines = f.read()
            labellines = lines.split("\n")
            for labelline in labellines:
                labellinesans = [0, 0, 0.0, 0.0, 0.0, 0.0]
                if labelline != "":
                    labellinesans[0] = i
                    labellinesans[1:6] = labelline.split(" ")
                    labelans.append(labellinesans)
            f.close()

    # return torch.from_numpy(np.array(labelans).astype(float))
    return torch.from_numpy(np.array(imageans_A).astype(float)),torch.from_numpy(np.array(imageans_A).astype(float)), torch.from_numpy(np.array(labelans).astype(float))


def packBs(labelpaths):
    labelans = []
    for i, labelpath in enumerate(labelpaths):
        f = open(labelpath)
        lines = f.read()
        labellines = lines.split("\n")
        for labelline in labellines:
            labellinesans = [0, 0.0, 0.0, 0.0, 0.0, ""]
            if labelline != "":
                labellinesans[0] = i
                labellinesans[1:6] = labelline.split(" ")
                labelans.append(labellinesans)
                # labelans.append(Variable(torch.from_numpy(np.array(labellinesans).astype(float))).cuda())
    # return torch.from_numpy(np.array(labelans).astype(float))
    return np.array(labelans).astype(float)
def generateATlabel(label_v,batchsize,imgsize):
    ans = np.zeros((batchsize,1,imgsize,imgsize),dtype=np.float32)
    for label in label_v:
        ans[int(label[0]),0,int(imgsize * (label[3] - 0.5*label[5])) :int(imgsize *  (label[3] + 0.5*label[5])),int(imgsize * (label[2] - 0.5*label[4])) : int(imgsize * (label[2] + 0.5*label[4]))] = 1
    return ans

class SalObjDataset(Dataset):
    def __init__(self, img_a_name_list,img_b_name_list, lbl_name_list, transform=None):
        self.A_name_list = img_a_name_list
        self.B_name_list = img_b_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.A_name_list)

    def __getitem__(self, idx):
        image_A = self.A_name_list[idx]
        image_B = self.B_name_list[idx]
        label = [self.label_name_list[idx]]
        # sample = {'image': torch.from_numpy(image), 'label': label}
        sample = {'image_A': image_A,'image_B': image_B, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
class SalObjDataset_yolox(Dataset):
    def __init__(self, img_a_name_list,img_b_name_list, lbl_name_list, transform=None):
        self.A_name_list = img_a_name_list
        self.B_name_list = img_b_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.A_name_list)

    def __getitem__(self, idx):
        image_a = io.imread(self.A_name_list[idx], plugin='pil')
        image_b = io.imread(self.B_name_list[idx], plugin='pil')
        atlabel = np.zeros((1,image_a.shape[0],image_a.shape[1]),dtype=np.float32)
        label = np.zeros((120,5),dtype=np.float32)
        image_a = image_a / np.max(image_a)
        image_b = image_b / np.max(image_b)

        n = 0
        f = open(self.label_name_list[idx])
        lines = f.read()
        labellines = lines.split("\n")
        for labelline in labellines:
            labellinesans = [0.0, 0.0, 0.0, 0.0, 0.0]
            if labelline != "":
                labellinesans = np.array(labelline.split(" ")).astype(np.float32)
                labellinesans[1:5] = labellinesans[1:5] * image_a.shape[0]
                label[n,:] = labellinesans
                atlabel[0,int((label[n,2] - 0.5 * label[n,4])):int((label[n,2] + 0.5 * label[n,4])),int((label[n,1] - 0.5 * label[n,3])): int((label[n,1] + 0.5 * label[n,3]))] = 1
                n = n + 1
        # sample = {'image': torch.from_numpy(image), 'label': label}
        image_a = image_a.transpose((2, 0, 1))
        image_b = image_b.transpose((2, 0, 1))
        sample = {'image_A':  torch.from_numpy(image_a)
            ,'image_B':  torch.from_numpy(image_b), "img_name" : self.B_name_list[idx],'label':  torch.from_numpy(label), 'atlabel':  torch.from_numpy(atlabel)}
        if self.transform:
            sample = self.transform(sample)
        return sample

# image flip
def GridMask(image, mask, num, p=0.5):
    seed_f = np.random.random()
    # print(seed_f)
    if seed_f > p:
        img1, lbl = image, mask
        gMI1, gML = img1.copy(), lbl.copy()

        img_h, img_w, _ = img1.shape
        x = np.random.randint(0, 160)
        y = np.random.randint(0, 200)
        w = np.random.randint(int(np.min(lbl[..., 2]) * image.shape[0] / num / 2),
                              int(np.min(lbl[..., 2]) * image.shape[0] / num))
        h = np.random.randint(int(np.min(lbl[..., 3]) * image.shape[0] / 2), int(np.min(lbl[..., 3]) * image.shape[0]))
        l = np.random.randint(int(image.shape[0] / 10), int(image.shape[0] / 5))

        temp = x
        while y < img_h:
            x = temp
            while x < img_w:
                # print('3')
                if x + w >= img_w or y + h > img_h: break
                # gML[x: x + w, y: y + h] = 0
                gMI1[x: x + w, y: y + h, :] = 0
                x = x + l + w
            y = y + l + h

        return gMI1
    else:
        return image


def flip(image1,image2, mask, mode, p=0.5):
    seed_f = np.random.random()
    if seed_f > p:
        if mode == 'x':
            image1_ = cv2.flip(image1, 0)
            image2_ = cv2.flip(image2, 0)
            mask[..., 1] = 1 - mask[..., 1]
        elif mode == 'y':
            image1_ = cv2.flip(image1, 1)
            image2_ = cv2.flip(image2, 0)
            mask[..., 0] = 1 - mask[..., 0]
        else:
            image1_, mask_ = image1, mask
        return image1_, image2_, mask
    else:
        return image1, image2, mask
def mixup(im_A,im_B, labels, im2_A, im2_B,labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im_A = (im_A * r + im2_A * (1 - r)).astype(np.uint8)
    im_B = (im_B * r + im2_B * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im_A,im_B, labels


def augment_hsv(im_A,im_B, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im_A, cv2.COLOR_BGR2HSV))
        dtype = im_A.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im_A)

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im_B, cv2.COLOR_BGR2HSV))
        dtype = im_B.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im_B)
        return im_A,im_B


def add_noise(image1, p=0.5):
    seed_f = np.random.random()
    if seed_f > p:
        h, w, _ = image1.shape
        for it in range(200):
            offset_x = np.random.randint(0, h)
            offset_y = np.random.randint(0, w)
            image1[offset_x, offset_y, :] = 255
        return image1
    else:
        return image1


def getRotate(image1, mask, angle, p=0.5):
    seed_f = np.random.random()
    if seed_f > p:
        h, w, _ = image1.shape
        M_rotate = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        image1_ = cv2.warpAffine(image1, M_rotate, (h, w))
        mask_ = cv2.warpAffine(mask, M_rotate, (h, w))
        return image1_, mask_
    else:
        return image1, mask

def packBs_one(labelpath,num):
    labelans = []
    f = open(labelpath,errors="ignore")
    lines = f.read()
    labellines = lines.split("\n")
    for labelline in labellines:
        labellinesans = [0.0, 0.0, 0.0, 0.0, 0.0]
        if labelline != '':
            labellinesans[0:5] = labelline.split(" ")
            labelans.append(labellinesans[0:5])
    return labelans

class Augment:
    def __init__(self, num=7, Path="./", outPath="./",Randomaffine=True, Mixup=True):
        self.num = num
        self.Randomaffine = Randomaffine
        self.Mixup = Mixup
        self.Path = Path
        self.outPath = outPath
        if not os.path.exists(self.outPath):
            os.mkdir(self.outPath)
        else:
            shutil.rmtree(self.outPath)
            os.mkdir(self.outPath)
        if not os.path.exists(self.outPath + r"\A"):
            os.mkdir(self.outPath+ r"\A")
        if not os.path.exists(self.outPath + r"\B"):
            os.mkdir(self.outPath + r"\B")
        if not os.path.exists(self.outPath + r"\label"):
            os.mkdir(self.outPath + r"\label")

    def readimageandlabel(self, A, B,mask):
        Aoutput = cv2.imread(A)
        Boutput = cv2.imread(B)
        maskoutput = packBs_one(mask, self.num)
        return Aoutput, Boutput,maskoutput

    def __call__(self, times, Rr=0.5, Mr=0.1, Hr=0.1):
        images_A_loader = []
        images_B_loader = []
        labels_loader = []
        image_A = glob.glob(self.Path + "\\A\\*.png")
        image_B = glob.glob(self.Path + "\\A\\*.png")
        mask = glob.glob(self.Path + "\\txt_label\\*.txt")
        if self.Randomaffine:
            for n in range(0, times):
                for i in range(len(image_A)):
                    Aimage, Bimage,mask = self.readimageandlabel(image_A[i],image_B[i], mask[i])
                    #newimage = GridMask(Aimage, Bimage, mask, self.num, p=Rr)
                    Aimage = add_noise(Aimage, p=Rr)
                    Bimage = add_noise(Bimage, p=Rr)
                    Aimage,Bimage, newlabel = flip(Aimage, Bimage, mask, mode='x', p=Rr)
                    Aimage,Bimage, newlabel = flip(Aimage,Bimage, mask, mode='y', p=Rr)
                    Aimage,Bimage = augment_hsv(Aimage,Bimage, hgain=0.5, sgain=0.5, vgain=0.5)
                    cv2.imwrite(self.outPath + '/A/' + str(i) + "_" + str(n) + ".png", Aimage)
                    cv2.imwrite(self.outPath + '/B/' + str(i) + "_" + str(n) + ".png", Bimage)
                    with open(self.outPath + '/label/' + str(i) + "_" + str(n) + ".txt", "a+") as f:
                        for labelline in newlabel:
                            f.write(" ".join(str(r) for r in labelline.tolist()) + "\n")
                        f.close()
                    images_A_loader.append(Aimage)
                    images_B_loader.append(Bimage)
                    labels_loader.append(newlabel)
        if self.Mixup:
            if len(images_A_loader) == 0:
                mixnum = 0
                for i in range(len(image_A)):
                    seed_f_1 = int(np.random.random() * len(image_A))
                    seed_f_2 = int(np.random.random() * len(image_A))
                    seed_r = np.random.random()
                    if seed_r < Mr:
                        Aimage_1, Bimage_1,mask_1 = self.readimageandlabel(image_A[seed_f_1],image_B[seed_f_1], mask[seed_f_1])
                        Aimage_2, Bimage_2,mask_1 = self.readimageandlabel(image_A[seed_f_2],image_B[seed_f_2], mask[seed_f_2])
                        Aimage, Bimage, newlabel = mixup(Aimage_1, Bimage_1,mask_1, Aimage_2, Bimage_2,mask_1)
                        cv2.imwrite(self.outPath + '/A/' + str(mixnum) + "_Mixup" + ".png", Aimage)
                        cv2.imwrite(self.outPath + '/B/' + str(mixnum) + "_Mixup" + ".png", Bimage)
                        with open(self.outPath + '/labels/' + str(mixnum) + "_Mixup" + ".txt", "a+") as f:
                            for labelline in newlabel:
                                f.write(" ".join(str(r) for r in labelline.tolist()) + "\n")
                            f.close()
                        images_A_loader.append(Aimage)
                        images_B_loader.append(Bimage)
                        labels_loader.append(newlabel)
                        mixnum += 1

            else:
                mixnum = 0
                for i in range(len(images_A_loader)):
                    seed_f_1 = int(np.random.random() * len(images_A_loader))
                    seed_f_2 = int(np.random.random() * len(images_A_loader))
                    seed_r = np.random.random()
                    if seed_r < Mr:
                        Aimage_1,Bimage_1, mask_1 = images_A_loader[seed_f_1], images_B_loader[seed_f_1],labels_loader[seed_f_1]
                        Aimage_2,Bimage_2, mask_2 = images_A_loader[seed_f_2], images_B_loader[seed_f_2],labels_loader[seed_f_2]
                        Aimage,Bimage, newlabel = mixup(Aimage_1,Bimage_1, mask_1,Aimage_2,Bimage_2, mask_2 )
                        cv2.imwrite(self.outPath + '/A/' + str(mixnum) + "_Mixup" + ".png", Aimage)
                        cv2.imwrite(self.outPath + '/B/' + str(mixnum) + "_Mixup" + ".png", Bimage)
                        with open(self.outPath + '/label/' + str(mixnum) + "_Mixup" + ".txt", "a+") as f:
                            for labelline in newlabel:
                                f.write(" ".join(str(r) for r in labelline.tolist()) + "\n")
                            f.close()
                        images_A_loader.append(Aimage)
                        images_B_loader.append(Bimage)
                        labels_loader.append(newlabel)
                        mixnum += 1
        return images_A_loader,images_B_loader, labels_loader
