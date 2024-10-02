import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils
from pathlib import Path
import glob
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import SalObjDataset
import argparse
import time
from tqdm import tqdm
import acc
from model.FDAFFNet import FDAFFNet_l
import os
def run(data_dir = "",
        image_ext = ".png",
        weights = '',
        imgsz = 256,
        device = "",
        project = "output",
        name = "demo",
        ifsave = False):
    
    model_dir = str(Path(project) / name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    os.makedirs(model_dir, exist_ok=True)
    test_a_name_list = list((Path(data_dir) / "test/A").rglob(f'*{image_ext }'))
    test_a_name_list = [str(f) for f in test_a_name_list]
    test_b_name_list = list((Path(data_dir) / "test/B").rglob(f'*{image_ext }'))
    test_b_name_list = [str(f) for f in test_b_name_list]
    test_lbl_name_list = [f.replace("\\A\\" ,"\\label\\") for f in test_a_name_list ]
   
    print("---")
    print("test images: ", len(test_a_name_list))
    print("test labels: ", len(test_lbl_name_list))
    print("---")
    
    test_dataset = SalObjDataset(
    img_a_list=test_a_name_list,
    img_b_list=test_b_name_list,
    lbl_name_list=test_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(imgsz),
        ToTensor()]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    net = FDAFFNet_l()
    net.load_state_dict(torch.load(weights))
    if torch.cuda.is_available() and device!="cpu":
        net.cuda()
    net.eval()

    hists = [[0, 0], [0, 0]]
    # --------- 4. inference for each image ---------
    num = 0
    for data_test in tqdm(test_dataloader):
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

        d1, d2, d3,d4, dout = net(inputs_va, inputs_vb)
        y_pb = dout[:, 0, :, :]
        sigmoid = torch.nn.Sigmoid()
        y_pb = torch.ge(sigmoid(y_pb), 0.5).float()
        pred = y_pb.cpu().detach().numpy()
        ans = pred[0]
        if ifsave:
            image_name = Path(test_a_name_list[num]).name
            cv2.imwrite(str(Path(model_dir)/image_name), ans * 255)
        hist_t = acc.hist(labels_v.cpu().detach().numpy(), pred)
        hists = hists + hist_t
        num = num + 1
        del d1, d2, d3,d4, dout, y_pb, pred, hist_t
        #del d1, d2, d3, d4, y_pb,CD_final, pred, hist_t

    recall, precision, iou, f1measure, accuray = acc.show_cd_rpqf1_pixel(hists)
    print("finished predicting ")
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='C:/Users/Admins/Desktop/pwg/CD_dataset/LEVIR_256', help='input dataset')
    parser.add_argument('--image-ext', type=str, default='.png', help='image extention')
    parser.add_argument('--weights', type=str, default='epoch_  7_best_loss_loss_0.064433_iou_86.059515.pth', help='initial weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=256, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='0930', help='save to project/name')
    # Logger arguments
    parser.add_argument('--ifsave', default=True, help='if save the predict image')
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt):
    run(**vars(opt))

if __name__ =="__main__":
    opt = parse_opt()
    main(opt)