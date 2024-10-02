import os
from pathlib import Path
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
import argparse
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import RandomCrop,Randomflip,Randomcov,RandomCut
from model.FDAFFNet import FDAFFNet_l
from data_loader import SalObjDataset
'''
bs = 4
60 epoch
'''

import pytorch_ssim

from tensorboardX import SummaryWriter

def dice(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    
    return 1 - dice.mean()

# IoU Loss
def Iou_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    total = (pred + target).sum(dim=(2, 3))
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return 1 - iou.mean()
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
def getloss( d1, d2, d3, d4, dout, labels_v,aloss):
    if aloss == "bce":
        loss1 = F.binary_cross_entropy_with_logits(d1, labels_v, reduce='none')
        loss2 = F.binary_cross_entropy_with_logits(d2, labels_v, reduce='none')
        loss3 = F.binary_cross_entropy_with_logits(d3, labels_v, reduce='none')
        loss4 = F.binary_cross_entropy_with_logits(d4, labels_v, reduce='none')
        loss_out = F.binary_cross_entropy_with_logits(dout, labels_v, reduce='none')
        return loss_out+0.4*loss4+0.3*loss3+0.2*loss2+0.1*loss1
    elif aloss == "iou":
        loss1 = Iou_loss(d1, labels_v)
        loss2 = Iou_loss(d1, labels_v)
        loss3 = Iou_loss(d1, labels_v)
        loss4 = Iou_loss(d1, labels_v)
        loss_out = Iou_loss(d1, labels_v)
        return loss_out+0.4*loss4+0.3*loss3+0.2*loss2+0.1*loss1
    elif aloss == "dice":
        loss1 = dice(d1, labels_v)
        loss2 = dice(d1, labels_v)
        loss3 = dice(d1, labels_v)
        loss4 = dice(d1, labels_v)
        loss_out = dice(d1, labels_v)
        return loss_out+0.4*loss4+0.3*loss3+0.2*loss2+0.1*loss1
    elif aloss == "ssim":
        loss1 = 1 - ssim_loss(d1, labels_v)
        loss2 = 1 - ssim_loss(d1, labels_v)
        loss3 = 1 - ssim_loss(d1, labels_v)
        loss4 = 1 - ssim_loss(d1, labels_v)
        loss_out = dice(d1, labels_v)
        return loss_out+0.4*loss4+0.3*loss3+0.2*loss2+0.1*loss1
    else:
        return torch.tensor(0.0, requires_grad=True)
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

    
def run(data_dir = "",
        image_ext = ".png",
        pre_weights = '',
        epochs = 100,
        batch_size = 4,
        imgsz = 256,
        cache = "ram",
        device = "",
        project = "output",
        name = "demo",
        loss_fc = ["BCE"]):
    
    model_dir = str(Path(project) / name)
    summary_name = Path(model_dir) / "log"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    Path(summary_name).mkdir(parents=True, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(summary_name)
    tra_a_name_list = list((Path(data_dir) / "train/A").rglob(f'*{image_ext }'))
    tra_a_name_list = [str(f) for f in tra_a_name_list]
    tra_b_name_list = list((Path(data_dir) / "train/B").rglob(f'*{image_ext }'))
    tra_b_name_list = [str(f) for f in tra_b_name_list]
    tra_lbl_name_list = [f.replace("\\A\\" ,"\\label\\") for f in tra_a_name_list ]

    val_a_name_list = list((Path(data_dir) / "test/A").rglob(f'*{image_ext }'))
    val_a_name_list = [str(f) for f in val_a_name_list]
    val_b_name_list = list((Path(data_dir) / "test/B").rglob(f'*{image_ext }'))
    val_b_name_list = [str(f) for f in val_b_name_list]

    val_lbl_name_list = [str(f).replace("\\A\\" ,"\\label\\") for f in val_a_name_list]
    
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
        RescaleT(imgsz),
        RandomCrop(224),
        #Randomcov(0.6),
        #Randomflip(0.6),
        #RandomCut(0.6,5),
        ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    val_dataset = SalObjDataset(
        img_a_list=val_a_name_list,
        img_b_list=val_b_name_list,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(imgsz),
            #RandomCrop(224),
            #Randomcov(0.6),
            #Randomflip(0.6),
            #RandomCut(0.6,5),
            ToTensor()]))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    dataloader = {
        'train': train_dataloader,
        'valid': val_dataloader
    }

    dataset_size = {
        'train': len(train_dataset),
        'valid': len(val_dataset)
    }
    net = FDAFFNet_l()
    if len(pre_weights) != 0:
        net.load_state_dict(torch.load(pre_weights))
    if torch.cuda.is_available() and device!="cpu":
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    since = time.time()
    lowest_loss = 1000.0
    best_iou = 60.0
    ite_num = 0
    for epoch in range(0, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # 每轮都有训练和验证阶段
        for phase in ['train', 'valid']:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

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
                d1, d2, d3, d4, dout = net(inputs_va,inputs_vb)
                loss = torch.tensor(.0, requires_grad=True)
                for aloss in loss_fc:
                    loss = loss + getloss( d1, d2, d3, d4, dout, labels_v,aloss)
                sum_loss = loss
                # # 前向传播

                # 只在训练阶段反向优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += sum_loss.item()
                # running_tar_loss += loss2.item()
                y_pb = torch.ge(torch.sigmoid(dout), 0.5).float()
                pred = y_pb.cpu().detach().numpy()
                intr, unn = calMetric_iou(labels.numpy(), pred)
                inter = inter + intr
                unin = unin + unn
                hist_t = hist(labels.numpy(), pred)
                hists = hists + hist_t
                del d1, d2, d3, d4,dout,  loss,  sum_loss, y_pb, pred, hist_t
                
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
                    epoch + 1, epochs, (i + 1) * batch_size, train_num, ite_num, running_loss / ite_num4val))
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
                    best_model_wts = net.state_dict()
                    torch.save(best_model_wts, model_dir + "//epoch_%3d_best_"
                                                           ""
                                                           "loss_loss_%3f_iou_%3f.pth" % (epoch + 1, epoch_loss, iou))
                elif iou > best_iou:
                    best_iou = iou
                    best_model_wts_acc = net.state_dict()
                    torch.save(best_model_wts_acc,
                               model_dir + "//epoch_%3d_best_iou_loss_%3f_iou_%3f.pth" % (epoch + 1, epoch_loss, iou))

    time_elapsed = time.time() - since
    writer.close()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val loss: {:.4f}'.format(lowest_loss))
    print('Best val iou: {:.4f}'.format(best_iou))

    # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)

    
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='LEVIR_256', help='input dataset')
    parser.add_argument('--image-ext', type=str, default='.png', help='image extention')
    parser.add_argument('--pre-weights', type=str, default='runs/train/0929epoch_  8_best_iou_loss_0.104475_iou_85.345496.pth', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=80, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=256, help='train, val image size (pixels)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='0929', help='save to project/name')
    # Logger arguments
    parser.add_argument('--loss-fc', default=['bce'], help='loss function')
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt):
    run(**vars(opt))

if __name__ =="__main__":
    opt = parse_opt()
    main(opt)

print('-------------Congratulations! Training Done!!!-------------')