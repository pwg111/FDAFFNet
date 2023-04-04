import os
# https://blog.csdn.net/qq_45266796/article/details/109028605
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
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
from model.FDDAFFNet_yolox import FDDAFFNet_yolox
from data_loader_OD import SalObjDataset_yolox,packBs,Augment,packBs_both
from tqdm import tqdm
import cv2
import pytorch_ssim
import pytorch_iou
#anchors pip = ([10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326])
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
def nums(pred,predscores,iou_threshold):
    predoutput = []
    for i,predimg in enumerate(pred):
        boxes_wx = predimg[:, 0:4]
        boxes_xy = torch.zeros(boxes_wx.shape, dtype=float)
        boxes_xy[:, 0] = boxes_wx[:, 0] - boxes_wx[:, 2] / 2
        boxes_xy[:, 1] = boxes_wx[:, 1] - boxes_wx[:, 3] / 2
        boxes_xy[:, 2] = boxes_wx[:, 0] + boxes_wx[:, 2] / 2
        boxes_xy[:, 3] = boxes_wx[:, 1] + boxes_wx[:, 3] / 2
        scores = predscores[i]
        output = torchvision.ops.nms(boxes_xy.cuda(), scores, iou_threshold)
        predoutput.append(output)
    return predoutput

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
def muti_structure_loss_fusion_a(at1, at2, at3, at_labels_v):
    lossa1 = structure_loss(at1, at_labels_v)
    lossa2 = structure_loss(at2, at_labels_v)
    lossa3 = structure_loss(at3, at_labels_v)
    # loss5 = structure_loss(d5,labels_v)
    loss = (lossa1 + lossa2 + lossa3).mean()
    return loss
data_dir = r"dataset\train"
tra_A_image_dir = '\A'
tra_B_image_dir = '\B'
tra_label_dir = r'\txt_label'



val_A_image_dir = r"dataset\test\A"
val_B_image_dir = r"dataset\test\B"
val_label_dir = r"dataset\test\txt_label"


image_ext = '.png'
label_ext = '.txt'
model_dir = r"weights\polyu\03_23\obj"
os.makedirs(model_dir, exist_ok=True)
outputpath = r"E:\PWG\MCDONet\dataset\test\pred"
tra_A_img_name_list = glob.glob(data_dir +  tra_A_image_dir + '\\*' + image_ext)
tra_B_img_name_list = glob.glob(data_dir +  tra_B_image_dir + '\\*' + image_ext)
tra_lbl_name_list = []

for img_path in tra_A_img_name_list:
    img_name = img_path.split("\\")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + "\\"+imidx + label_ext)


val_A_img_name_list = glob.glob(val_A_image_dir + '\\*' + image_ext)
val_B_img_name_list = glob.glob(val_B_image_dir + '\\*' + image_ext)
val_lbl_name_list = []
for img_path in val_A_img_name_list:
    img_name = img_path.split("\\")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    val_lbl_name_list.append(val_label_dir + "\\"+imidx + label_ext)

#DataAugment = Augment(Path = r"LP2\trainaug")
print("---")
print("train images: ", len(tra_A_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("valid images: ", len(val_A_img_name_list))
print("valid labels: ", len(val_lbl_name_list))
print("---")
epoch_num = 30
batch_size_train = 4
batch_size_val = 4
train_num = 0
val_num = len(val_A_img_name_list)
conf = 0.9
iou = 0.5
train_num = len(tra_A_img_name_list)



train_dataset = SalObjDataset_yolox(
    img_a_name_list=tra_A_img_name_list,
    img_b_name_list = tra_B_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
      #  packBs(),
     #   encoder(7)
    ]))

val_dataset = SalObjDataset_yolox(
    img_a_name_list=val_A_img_name_list,
    img_b_name_list=val_B_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        #packBs(),
       # encoder(7)
    ]))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=0,drop_last=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0,drop_last=True)

dataloader={
    'train':train_dataloader,
    'valid':val_dataloader
    }

dataset_size={
    'train':len(train_dataset),
    'valid':len(val_dataset)
    }
print(dataset_size)
net = FDDAFFNet_yolox()


pre_model = torch.load(r"weights\polyu\03_23\obj\epoch_ 30_loss_1.240071.pth")
net.load_state_dict(pre_model)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
net.eval()
# ------- 5. training process --------
print("---start training...")
min_loss = 999999999
for epoch in range(0, epoch_num):
    print('Epoch {}/{}'.format(epoch, epoch_num - 1))
    print('-' * 10)

    net.train(False)
    i = 0
    meanloss = 0
    # 训练模块
    for data_test in val_dataloader:
        i = i + 1
        inputs_a_test = data_test['image_A']
        inputs_b_test = data_test['image_B']
        labels_test = data_test['label']
        at_label_v = data_test['atlabel']
        input_path = data_test['img_name']
        inputs_a_test  = inputs_a_test.type(torch.FloatTensor)
        inputs_b_test = inputs_b_test.type(torch.FloatTensor)
        labels_test = labels_test.type(torch.FloatTensor)
        at_label_v = at_label_v.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_a_test = Variable(inputs_a_test.cuda())
            inputs_b_test = Variable(inputs_b_test.cuda())
            labels_test = Variable(labels_test.cuda())
            at_label_v = Variable(at_label_v.cuda())
        else:
            inputs_a_test = Variable(inputs_a_test)
            inputs_b_test = Variable(inputs_b_test)
            labels_test = Variable(labels_test)
            at_label_v = Variable(at_label_v)
        optimizer.zero_grad()
        at_out1,at_out2,at_out3,d1 = net(inputs_a_test,inputs_b_test,labels_test)
        ans = d1.cpu().detach().numpy()
        flagans = np.zeros((ans.shape[0], ans.shape[1]), dtype=np.uint8)
        scoresbox = ans[:, :, 4]
        flagans[scoresbox > 0.5] = 1
        scoresbox2 = []
        ans_2 = []
        for i, scorebox in enumerate(scoresbox):
            scoresbox2.append(torch.tensor(np.delete(scorebox, np.where(scorebox < 0.5))).cuda())
        for i, Aans in enumerate(ans):
            ans_2.append(torch.tensor(np.delete(Aans, np.where(flagans[i] == 0), axis=0)).cuda())
        output = nums(ans_2, scoresbox2, iou)
        targets = []
        for n, preds in enumerate(output):
            preds = preds.cpu().detach().numpy()
            img = cv2.imread(input_path[n])
            for ts in preds:
                ing = ans_2[n][ts].cpu().detach().numpy()
                targets.append(ing)
                img = cv2.rectangle(img, (int(ing[0] - 0.5 * ing[2]), int(ing[1] - 0.5 * ing[3])),
                                    (int(ing[0] + 0.5 * ing[2]), int(ing[1] + 0.5 * ing[3])), (0, 0, 255), 3)
            cv2.imwrite(outputpath + "/" + input_path[n].split("\\")[-1], img)
        ''''''
        loss = d1[0]
        at_loss = 0.2 * muti_structure_loss_fusion_a(at_out1,at_out2,at_out3, at_label_v)
        sum_loss = loss + at_loss
        print("[epoch: %3d/%3d, batch: %5d/%5d] loss: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, sum_loss
        ))
        meanloss = meanloss +(float(sum_loss)/len(train_dataset))
        y_pb = torch.ge(torch.sigmoid(at_out3), 0.5).float()
        pred = y_pb.cpu().detach().numpy()
        #sum_loss.backward()
        #optimizer.step()
        del at_out1,at_out2,at_out3,d1,loss,sum_loss,at_loss
    targetsums = 0
    '''
    net.train(False)
    i = 0
    for data_test in val_dataloader:
        i = i + 1
        inputs_a_test = data_test['image_A']
        inputs_b_test = data_test['image_B']
        input_path = inputs_b_test
        labels_test = data_test['label']
        inputs_a_test, inputs_b_test, labels_test = packBs_both(inputs_a_test, inputs_b_test, labels_test, 256
                                                                )
        inputs_a_test = inputs_a_test.type(torch.FloatTensor)
        inputs_b_test = inputs_b_test.type(torch.FloatTensor)
        labels_test = labels_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_a_test = Variable(inputs_a_test.cuda())
            inputs_b_test = Variable(inputs_b_test.cuda())
            labels_test = Variable(labels_test.cuda())
        else:
            inputs_a_test = Variable(inputs_a_test.cuda())
            inputs_b_test = Variable(inputs_b_test.cuda())
            labels_test = Variable(labels_test.cuda())
        optimizer.zero_grad()
        at_out1,at_out2,at_out3,d1 = net(inputs_a_test, inputs_b_test)
        ans = d1[0].cpu().detach().numpy()
        flagans = np.zeros((ans.shape[0], ans.shape[1]), dtype=np.uint8)
        scoresbox = ans[:, :, 4]
        flagans[scoresbox > 0.5] = 1
        scoresbox2 = []
        ans_2 = []
        for i, scorebox in enumerate(scoresbox):
            scoresbox2.append(torch.tensor(np.delete(scorebox, np.where(scorebox < 0.5))).cuda())
        for i, Aans in enumerate(ans):
            ans_2.append(torch.tensor(np.delete(Aans, np.where(flagans[i] == 0), axis=0)).cuda())
        output = nums(ans_2, scoresbox2, iou)
        targets = []
        for n, preds in enumerate(output):
            preds = preds.cpu().detach().numpy()
            img = cv2.imread(input_path[n])
            for ts in preds:
                ing = ans_2[n][ts].cpu().detach().numpy()
                targets.append(ing)
                img = cv2.rectangle(img,(int(ing[0] - 0.5 * ing[2]),int(ing[1] - 0.5 * ing[3])),(int(ing[0] + 0.5 * ing[2]),int(ing[1] + 0.5 * ing[3])),(0,0,255),3)
            cv2.imwrite(outputpath + "/" +input_path[n].split("\\")[-1],img)
        print("[epoch: %3d/%3d, batch: %5d/%5d] val" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_val, val_num))
        # ans[np.max(ans[:,:,:,:,]) > conf]
        del d1, ans, flagans, scoresbox, output
    '''
    if meanloss < min_loss or epoch_num == 0:
        min_loss = meanloss
        best_model_wts = net.state_dict()
        torch.save(best_model_wts,
                   model_dir + "\epoch_%3d_loss_%3f.pth" % (epoch + 1, meanloss))