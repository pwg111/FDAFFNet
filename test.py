import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

import glob
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import SalObjDataset

import time
from tqdm import tqdm
import acc
from othermodels.MSCD_v4 import MSCDNet
from model.FDAFFNet import FDAFFNet,FDDAFFNet_X
from othermodels.DSIFN_CD import DSIFN
from othermodels.siamunet_diff import SiamUnet_diff
from othermodels.siamunet_conc import SiamUnet_conc
# --------- 1. get image path and name ---------
'''
img_a_dirs = ["BCD_\\val\\A"]
img_b_dirs = ["BCD_\\val\\B"]
label_dirs = ["BCD_\\val\\label"]
output_dir = ["BCD_\\val\\pred"]

'''
img_a_dirs = ["Levir_CD_256\\test\\A"]
img_b_dirs = ["Levir_CD_256\\test\\B"]
label_dirs = ["Levir_CD_256\\test\\label"]
output_dir = ["Levir_CD_256\\test\\DSIFNpred"]

img_num = len(img_a_dirs)
image_ext = ".png"
label_ext = ".png"
# save_dir = "C:\\Users\\john\\Desktop\\PWG\\COCOME_S_A\\testresult\\"
# if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
for i in range(img_num):
    image_a_dir = img_a_dirs[i]
    image_b_dir = img_b_dirs[i]
    label_dir = label_dirs[i]
    output_height = 256
    output_width = output_height

    test_a_name_list = glob.glob(image_a_dir + '\\*' + image_ext)
    test_b_name_list = glob.glob(image_b_dir  + '\\*' + image_ext)
    test_lbl_name_list = []
    imdixs = []
    for img_path in test_a_name_list:
        img_name = img_path.split("\\")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        imdixs.append(imidx)
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
            imdixs.append(imidx)
        test_lbl_name_list.append(label_dir + "\\" + imidx + '.png')

    # --------- 2. dataloader ---------
    # 1. dataload
    test_salobj_dataset = SalObjDataset(img_a_list=test_a_name_list,
                                        img_b_list=test_b_name_list,
                                        lbl_name_list=test_lbl_name_list,
                                        transform=transforms.Compose([RescaleT(256), ToTensor()]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --------- 3. model define ---------
    print("...load Model...")

    net = FDDAFFNet_X()
    #net = DSIFN()
    #model_dir = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\Mine_l_C3\epoch_ 77_best_loss_loss_0.293974_iou_0.852902.pth'
    model_dir = r'weights\polyu\Levir_cd\ourx\epoch_ 46_best_loss_loss_0.429331_iou_85.724471.pth'
    '''
    model_dir = r'E:\pwg\FDDAFFNet\weights\polyu\Levir_cd\Mine_2\epoch_ 25_best_loss_loss_2.322033_iou_84.540904.pth'
    pre_model = torch.load(model_dir)
    model2_dict = net.state_dict()
    state_dict = {k: v for k, v in pre_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    net.load_state_dict(model2_dict)
    '''
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    hists = [[0, 0], [0, 0]]
    time_start_all = time.time()
    # --------- 4. inference for each image ---------
    num = 0
    for data_test in tqdm(test_salobj_dataloader):
        inputs_a, inputs_b, labels = data_test['image_a'], data_test['image_b'], data_test['label']

        inputs_a = inputs_a.type(torch.FloatTensor)
        inputs_b = inputs_b.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # # wrap them in Variable
        if torch.cuda.is_available():
            inputs_va, inputs_vb, labels_v = Variable(inputs_a.cuda(), requires_grad=False), Variable(inputs_b.cuda(),
                                                                                                      requires_grad=False), Variable(
                labels.cuda(),
                requires_grad=False)
        else:
            inputs_va, inputs_vb, labels_v = Variable(inputs_a, requires_grad=False), Variable(inputs_b,
                                                                                               requires_grad=False), Variable(
                labels, requires_grad=False)
        '''
        d1, d2, d3, d4 = net(inputs_va,inputs_va)
        loss = muti_structure_loss_fusion(d1, d2, d3, d4, labels_v)
        '''
        #----------DSIFN-------
        '''
        d1, d2, d3, d4, d5 = net(inputs_va, inputs_vb)
        y_pb = d1[:, 0, :, :]'''
        #----------FDDAFFNET--------

        #at1,at2,at3,at4,d1,d2,d3,d4,dout = net(inputs_va, inputs_vb)
        d1, d2, d3,d4, dout = net(inputs_va, inputs_vb)
        y_pb = dout[:, 0, :, :]

        # ----------MSCDNet---------
        '''
        CD_final, d1, d2, d3, d4, d5 = net(inputs_va, inputs_vb)
        y_pb = CD_final[:, 0, :, :]'''
        sigmoid = torch.nn.Sigmoid()
        y_pb = torch.ge(sigmoid(y_pb), 0.39).float()
        pred = y_pb.cpu().detach().numpy()
        ans = pred[0]
        #cv2.imwrite(test_a_name_list[num].replace("\\A\\", "\\MINEpred\\"), ans * 255)
        hist_t = acc.hist(labels_v.cpu().detach().numpy(), pred)
        hists = hists + hist_t
        num = num + 1
        del d1, d2, d3,d4, dout, y_pb, pred, hist_t
        #del d1, d2, d3, d4, y_pb,CD_final, pred, hist_t

    recall, precision, iou, f1measure, accuray = acc.show_cd_rpqf1_pixel(hists)
    print("finished predicting ", image_a_dir)