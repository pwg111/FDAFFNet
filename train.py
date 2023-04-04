import os

# https://blog.csdn.net/qq_45266796/article/details/109028605
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import time
# import os
import math

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.FDDAFFNet import FDDAFFNet_Seg_resblock
import matplotlib.pyplot as plt

'''
bs = 4
60 epoch
'''

import pytorch_ssim
import pytorch_iou

from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def structure_loss_edge(pred, mask, radius):
    ks = 2 * radius + 1
    filt1 = torch.ones(1, 1, ks, ks)
    filt1[:, :, radius:2 * radius, radius:2 * radius] = -8
    filt1.requires_grad = False
    filt1 = filt1.cuda()
    # label = label.unsqueeze(1)
    mask = F.conv2d(mask.float(), filt1, bias=None, stride=1, padding=radius)
    mask = 1 - torch.eq(mask, 0).float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    ssim_out = 1 - ssim_loss(pred, mask)
    return (wbce + wiou + ssim_out).mean()


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    ssim_out = 1 - ssim_loss(pred, mask)
    return (wbce + wiou + ssim_out).mean()

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        #predict = F.softmax(predict, dim=1)
        predict = torch.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
dice =DiceLoss().to(device, dtype=torch.float)
def focal_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    get_focal_loss = Focal_Loss()

    pred = torch.sigmoid(pred)
    wbce = 5 * get_focal_loss(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    ssim_out = 0.0
    return (wbce + wiou + ssim_out).mean()


def muti_structure_loss_fusion_v7(at1, at2, at3, d1, d2, d3, d4,dout, labels_v):
    lossa1 = structure_loss(at1, labels_v)
    lossa2 = structure_loss(at2, labels_v)
    lossa3 = structure_loss(at3, labels_v)
    loss1 = structure_loss(d1, labels_v)
    loss2 = structure_loss(d2, labels_v)
    loss3 = structure_loss(d3, labels_v)
    loss4 = structure_loss(d4, labels_v)
    loss5 = structure_loss(dout,labels_v)
    loss = loss5 + loss4 +loss3 + loss2 + loss1 + lossa1 + lossa2 + lossa3
    return loss
def muti_structure_loss_fusion_v7_2(at1, at2, at3, d1, d2, d3, d4,dout, labels_v):
    lossa1 = structure_loss(at1, labels_v)
    lossa2 = structure_loss(at2, labels_v)
    lossa3 = structure_loss(at3, labels_v)
    loss1 = structure_loss(d1, labels_v)
    loss2 = structure_loss(d2, labels_v)
    loss3 = structure_loss(d3, labels_v)
    loss4 = structure_loss(d4, labels_v)
    loss5 = structure_loss(dout,labels_v)
    loss = loss5 + 0.2 * (loss4 +loss3 + loss2 + loss1 )+ 0.2*(lossa1 + lossa2 + lossa3)
    loss2 = loss5 + loss4 + loss3 + loss2 + loss1 + lossa1 + lossa2 + lossa3
    return loss,loss2

summary_name = r"weights\polyu\0330\v7_epr\log"
model_dir = "weights/polyu/0330/v7_epr"
os.makedirs(model_dir, exist_ok=True)
writer = SummaryWriter(summary_name)
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp
def hist(gt_data, pre_data):
    # show_numpydata(gt_data)
    # show_numpydata(pre_data)
    gt_data[gt_data > 0.5] = 1
    gt_data[gt_data < 1] = 0
    pre_data[pre_data > 0.5] = 1
    pre_data[pre_data < 1] = 0
    hist = np.zeros((2, 2))
    # tp
    tp = np.count_nonzero((gt_data == pre_data) & (gt_data > 0))
    # tn
    tn = np.count_nonzero((gt_data == pre_data) & (gt_data == 0))
    # fp
    fp = np.count_nonzero(gt_data < pre_data)
    # fn
    fn = np.count_nonzero(gt_data > pre_data)
    hist[0, 0] = hist[0, 0] + tp
    hist[1, 1] = hist[1, 1] + tn
    hist[0, 1] = hist[0, 1] + fp
    hist[1, 0] = hist[1, 0] + fn
    return hist


# ------- 2. set the directory of training dataset --------


data_dir = r"dataset\train"
tra_a_dir = '\A'
tra_b_dir = '\B'
tra_label_dir = '\label'

val_a_dir = r"dataset\test\A"
val_b_dir = r"dataset\test\B"
val_label_dir = r"dataset\test\label"

image_ext = '.png'
label_ext = '.png'

# ------------set hyper parameters
epoch_num = 60
batch_size_train = 4
batch_size_val = 4
train_num = 0
val_num = 0
# -------------load dataset
print("preparing datasets and dataloaders......")
# tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_a_name_list = glob.glob(data_dir + tra_a_dir + '\\*' + image_ext)
tra_b_name_list = glob.glob(data_dir + tra_b_dir + '\\*' + image_ext)
tra_lbl_name_list = []
for img_path in tra_a_name_list:
    img_name = img_path.split("\\")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + "\\" + imidx + label_ext)

val_a_name_list = glob.glob(val_a_dir + '\\*' + image_ext)
val_b_name_list = glob.glob(val_b_dir + '\\*' + image_ext)

val_lbl_name_list = []
for img_path in val_a_name_list:
    img_name = img_path.split("\\")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    val_lbl_name_list.append(val_label_dir + "\\" + imidx + label_ext)

# tra_img_name_list = tra_img_name_list[:3]
# tra_lbl_name_list = tra_lbl_name_list[:3]
# val_img_name_list = val_img_name_list[:3]
# val_lbl_name_list = val_lbl_name_list[:3]
print("---")
print("train images: ", len(tra_a_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("valid images: ", len(val_a_name_list))
print("valid labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_a_name_list)

train_dataset = SalObjDataset(
    img_a_list=tra_a_name_list,
    img_b_list=tra_b_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        # RandomCrop(224),
        ToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)

val_dataset = SalObjDataset(
    img_a_list=val_a_name_list,
    img_b_list=val_b_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        # RandomCrop(224),
        ToTensor()]))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=0, drop_last=True)

dataloader = {
    'train': train_dataloader,
    'valid': val_dataloader
}

dataset_size = {
    'train': len(train_dataset),
    'valid': len(val_dataset)
}
# ------- 3. define model --------
# define the net


net = FDDAFFNet_Seg_resblock()
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
'''
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
'''
''''''
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)
# ------- 5. training process --------
print("---start training...")

def train_model(model, optimizer, num_epoches):
    since = time.time()
    lowest_loss = 1000.0
    best_iou = 60.0
    ite_num = 0
    for epoch in range(0, num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches - 1))
        print('-' * 10)
        # 每轮都有训练和验证阶段
        for phase in ['train', 'valid']:

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            # running_tar_loss = 0.0
            hists = [[0, 0], [0, 0]]
            iou = 0.0

            ite_num4val = 0
            i = 0
            inter, unin = 0, 0
            for data in dataloader[phase]:
                i = i + 1
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs_a, inputs_b, labels = data['image_a'],data['image_b'], data['label']

                inputs_a = inputs_a.type(torch.FloatTensor)
                inputs_b = inputs_b.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_va,inputs_vb, labels_v = Variable(inputs_a.cuda(), requires_grad=False), Variable(inputs_b.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_va, inputs_vb, labels_v = Variable(inputs_a, requires_grad=False), Variable(inputs_b, requires_grad=False), Variable(labels,requires_grad=False)

                optimizer.zero_grad()
                '''
                d1, d2, d3, d4 = net(inputs_va,inputs_va)
                loss = muti_structure_loss_fusion(d1, d2, d3, d4, labels_v)
                '''
                at1, at2, at3, d1, d2, d3, d4,dout = net(inputs_va,inputs_vb)
                if num_epoches - epoch >20:
                    loss = muti_structure_loss_fusion_v7(at1, at2, at3, d1, d2, d3, d4, dout, labels_v)
                    sum_loss = loss
                else:
                    loss,loss2 = muti_structure_loss_fusion_v7_2(at1, at2, at3, d1, d2, d3, d4, dout, labels_v)
                    sum_loss = loss2
                # # 前向传播

                # 只在训练阶段反向优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # loss.backward()
                # optimizer.step()

                # # print statistics
                running_loss += sum_loss.item()
                # running_tar_loss += loss2.item()
                y_pb = torch.ge(torch.sigmoid(dout), 0.5).float()
                pred = y_pb.cpu().detach().numpy()
                intr, unn = calMetric_iou(labels.numpy(), pred)
                inter = inter + intr
                unin = unin + unn
                hist_t = hist(labels.numpy(), pred)
                hists = hists + hist_t
                # if ite_num % 1000 ==0:
                #     if phase =='train':
                #         writer.add_images('Train/Img', inputs_v, ite_num)
                #         writer.add_images('Train/Label', labels_v, ite_num)
                #         writer.add_images('Train/d1', torch.sigmoid(d1)*255, ite_num)
                #         writer.add_images('Train/d2', torch.sigmoid(d2)*255, ite_num)
                #         writer.add_images('Train/d3', torch.sigmoid(d3)*255, ite_num)
                #         writer.add_images('Train/d4', torch.sigmoid(d4)*255, ite_num)
                #         # writer.add_images('Train/d5', torch.sigmoid(d5)*255, ite_num)
                #         writer.add_images('Train/d6', torch.sigmoid(d6)*255, ite_num)
                #         # writer.add_images('Valid/d6', d6, epoch + 1)
                #         writer.flush()
                #     else:
                #         writer.add_images('Valid/Img', inputs_v, ite_num)
                #         writer.add_images('Valid/Label', labels_v, ite_num)
                #         writer.add_images('Valid/d1', torch.sigmoid(d1)*255, ite_num)
                #         writer.add_images('Valid/d2', torch.sigmoid(d2)*255, ite_num)
                #         writer.add_images('Valid/d3', torch.sigmoid(d3)*255, ite_num)
                #         writer.add_images('Valid/d4', torch.sigmoid(d4)*255, ite_num)
                #         # writer.add_images('Valid/d5', torch.sigmoid(d5)*255, ite_num)
                #         writer.add_images('Valid/d6', torch.sigmoid(d6)*255, ite_num)
                #         # writer.add_images('Valid/d6', d6, epoch + 1)
                #         writer.flush()
                #     # del d0, d1, d2, loss2, loss1,loss0,loss,y_pb, pred,hist_t
                del d1, d2, d3, d4,dout,  loss,  sum_loss, y_pb, pred, hist_t,at1, at2, at3
                #del d1, d2, d3, d4,  loss,  sum_loss, y_pb, pred, hist_t

                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))
            if  phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            tp = hists[0, 0]
            fp = hists[0, 1]
            fn = hists[1, 0]
            iou = tp * 100.0 / (tp + fp + fn)
            print('{} Loss: {:.6f}'.format(phase, epoch_loss))
            print('{} iou: {:.6f}'.format(phase, (inter * 1.0 / unin)))
            if phase == 'valid':
                writer.add_scalar('Valid/Loss', running_loss / dataset_size[phase], epoch + 1)
                writer.add_scalar('Valid/IoU', iou, epoch + 1)
                writer.flush()
            else:
                writer.add_scalar('Train/Loss', running_loss / dataset_size[phase], epoch + 1)
                writer.add_scalar('Train/IoU', iou, epoch + 1)
                writer.flush()

            # 深度复制模型
            if phase == 'valid':
                if epoch_loss < lowest_loss:
                    if iou > best_iou:
                        best_iou = iou
                    lowest_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, model_dir + "epoch_%3d_best_"
                                                           ""
                                                           "loss_loss_%3f_iou_%3f.pth" % (epoch + 1, epoch_loss, iou))
                elif iou > best_iou:
                    best_iou = iou
                    best_model_wts_acc = model.state_dict()
                    torch.save(best_model_wts_acc,
                               model_dir + "epoch_%3d_best_iou_loss_%3f_iou_%3f.pth" % (epoch + 1, epoch_loss, iou))

    time_elapsed = time.time() - since
    writer.close()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val loss: {:.4f}'.format(lowest_loss))
    print('Best val iou: {:.4f}'.format(best_iou))


train_model(net, optimizer, epoch_num)

print('-------------Congratulations! Training Done!!!-------------')