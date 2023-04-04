import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import warnings
from .resnet_model import *

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,d = 1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p)+d-1, groups=g,dilation=d ,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.PReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True,d = 1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=c2 if c2<c1 else c1, act=act,d = 1)
class BottleNeck1(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c1, 3, 1,1, g=g)
    def forward(self, x):
        return self.cv2(self.cv1(x)) + x
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
class BottleNeck2(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c1, 3, 1,1, g=g)
    def forward(self, x):
        return self.cv2(self.cv1(x))
class C3_1(nn.Module):
    def __init__(self, c1,c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[BottleNeck1(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3_2(nn.Module):
    def __init__(self, c1,c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1*2, c1, 1, 1)
        self.cv2 = Conv(c1*2, c1, 1, 1)
        self.cv3 = Conv(c1*2, c2, 1, 1)
        self.m = nn.Sequential(*[BottleNeck2(c2, c2, shortcut, g, e=1.0) for _ in range(n)])
    def forward(self, x):
        x = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)
        return self.cv3(x)
class C3_3(nn.Module):
    def __init__(self, c1,c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv(c1, c1, 1, 1)
        self.cv3 = Conv(c1*2, c2, 1, 1)
        self.m = nn.Sequential(*[BottleNeck2(c2, c2, shortcut, g, e=1.0) for _ in range(n)])
    def forward(self, x):
        x = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)
        return self.cv3(x)
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class Detect(nn.Module):
    stride = (4,8,16) # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=2, anchors=([5,7, 8,15, 17,12],[15,31, 31,13, 30,60],[58,45, 78,99, 187,163]), ch=(256,512,1024), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = 5 + nc  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio = 8 , kernel_size = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class Resblock(nn.Module):
    def __init__(self, c1,k =3,s=1,p=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.cv1 = ConvBnRelu(c1, c1, k, s,p)
        self.cv2 = Conv(c1, c1, k, s,p)
        self.bn1 = nn.BatchNorm2d(c1, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.cv2(self.cv1(x))
        return self.relu(self.bn1(x))

class Resblock_2(nn.Module):
    def __init__(self, c1,c2,k =3,s=1,p=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.cv1 = ConvBnRelu(c1, c2, k, s,p)
        self.cv2 = Conv(c1, c2, k, s,p )
        self.cv3 = Conv(c2, c2, k, s,p)
        self.bn1 = nn.BatchNorm2d(c2, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.cv2(x) + self.cv3(self.cv1(x))
        return self.relu(self.bn1(x))

class mpuunit(nn.Module):
    def __init__(self, c1, c2, a = (1, 2, 4)):
        super().__init__()
        self.in_channel = c1
        self.out_channel = c2
        self.mpu0 = Conv(c1,c2,1,1)
        self.mpu1 = DWConv(c1, c2,3,1,d = a[0])
        self.mpu2 = DWConv(c1, c2,3,1,d = a[1])
        self.mpu3 = DWConv(c1, c2,3,1,d = a[2])
        self.conv1 = Conv(4*c2,c2,1,1)


    def forward(self, x):
        x0 = self.mpu0(x)
        x1 = self.mpu1(x)
        x2 = self.mpu2(x)
        x3 = self.mpu3(x)
        return self.conv1(torch.cat([x0, x1, x2, x3], 1))
class CRBMoudule(nn.Module):
    def __init__(self, c1, c2,s =1):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.convbnrelu1 = Conv(c2,c2,3,s,act=None)
        self.conv2 = Conv(c2,c2,3,s,act=None)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.PReLU()
        self.conv3 = Conv(c2,1,3,s,act=None)
    def forward(self, x):
        identity = self.conv1(x)
        out = self.bn2(self.conv2(self.convbnrelu1(identity)))
        out = out + identity
        out = self.conv3(self.act(out))
        return out
class EPRmoudul(nn.Module):
    def __init__(self, c1, c2,a = (1, 2, 4),Down_sample = None):
        super().__init__()
        s = 2 if Down_sample else 1
        self.conv1 = Conv(c1,c2,1,s,act=None)
        self.mpu = mpuunit(c1,c2,a)
        self.conv2 = Conv(c2,c2,3,s,act=None)
        self.act = nn.PReLU()
    def forward(self, x):
        identity = x
        out = self.mpu(x)
        out = self.conv2(out)
        if self.conv1 is not None:
            identity = self.conv1(identity)
        out = out + identity
        return self.act(out)
#EPRmodel
class FDDAFFNet_Seg_EPR(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_Seg_EPR, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet18(pretrained=True)
        ## -------------Feature Etraction--------------

        self.p3 = Conv(3, 64,3, 1, 1)
        self.p4 = Conv(64, 64, 3, 2, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,5)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.cbam_T1_1 = CBAM(256)
        self.cbam_T2_1 = CBAM(256)
        self.conv1_diff = ConvBnRelu(256, 256, 3, 1,1)
        self.conv1_and = ConvBnRelu(256, 256, 3, 1,1)
        self.cbam_T1_2 = CBAM(128)
        self.cbam_T2_2 = CBAM(128)
        self.conv2_diff = ConvBnRelu(128, 128, 3, 1,1)
        self.conv2_and = ConvBnRelu(128, 128, 3, 1,1)
        self.cbam_T1_3 = CBAM(128)
        self.cbam_T2_3 = CBAM(128)
        self.conv3_diff = ConvBnRelu(128, 128, 3, 1,1)
        self.conv3_and = ConvBnRelu(128, 128, 3, 1,1)

        ## -------------FPN--------------
        self.p6 = ConvBnRelu(512, 256, 3, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat = Concat(1)
        self.C3_NUM5 = EPRmoudul(512, 256)
        self.p7 = ConvBnRelu(256,128,3,1,1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.C3_NUM6 = EPRmoudul(256,128)
        ## -------------PAN--------------

        self.p8 = ConvBnRelu(256, 256, 3, 2, 1)
        self.C3_NUM7 = EPRmoudul(512,512)
        self.p9 = ConvBnRelu(512, 512, 3, 2, 1)
        self.C3_NUM8 = EPRmoudul(1024,1024)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512,3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512,3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512,3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024, 256,3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(256, 256,3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256,3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(512, 128,3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(128, 128,3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128,3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128, 64,3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_out = RefUnet(1,64)
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = Conv(512, 1, 1, 1, 0)
        self.outside_2 = Conv(256, 1, 1, 1, 0)
        self.outside_3 = Conv(128, 1, 1, 1, 0)
        self.outside_4 = Conv(64, 1, 1, 1, 0)

    def forward(self, x1,x2):
        ##-----------X1------------
        x1_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x1))))
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.SPPF_1(self.encoder4(x1_MAP2))
        x1_FPN_feature_1 = self.p6(x1_MAP3)
        x1_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x1_MAP2, self.upsample2(x1_FPN_feature_1)])))
        x1_FPN_feature_3 = self.C3_NUM6(self.concat([x1_MAP1, self.upsample2(x1_FPN_feature_2)]))
        ##-----------X2------------
        x2_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x2))))
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.SPPF_1(self.encoder4(x2_MAP2))
        x2_FPN_feature_1 = self.p6(x2_MAP3)
        x2_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x2_MAP2, self.upsample2(x2_FPN_feature_1)])))
        x2_FPN_feature_3 = self.C3_NUM6(self.concat([x2_MAP1, self.upsample2(x2_FPN_feature_2)]))
        # -----PAN-----
        Converse_Attention_1 = F.pairwise_distance(self.cbam_T1_1(x1_FPN_feature_1), self.cbam_T2_1(x2_FPN_feature_1), keepdim=True)
        at_out1 = self.upscore16(Converse_Attention_1)
        FPN_feature_1 = self.concat([self.conv1_diff(self.sigmoid(Converse_Attention_1) * torch.abs(x1_FPN_feature_1 - x2_FPN_feature_1))
                                    , self.conv1_and(self.sigmoid(Converse_Attention_1) * (x1_FPN_feature_1 + x2_FPN_feature_1))])
        Converse_Attention_2 = F.pairwise_distance(self.cbam_T1_2(x1_FPN_feature_2), self.cbam_T2_2(x2_FPN_feature_2), keepdim=True)
        at_out2 = self.upscore8(Converse_Attention_2)
        FPN_feature_2 = self.concat([self.conv2_diff(self.sigmoid(Converse_Attention_2) * torch.abs(x1_FPN_feature_2 - x2_FPN_feature_2))
                                    , self.conv2_and(self.sigmoid(Converse_Attention_2) * (x1_FPN_feature_2 + x2_FPN_feature_2))])
        Converse_Attention_3 = F.pairwise_distance(self.cbam_T1_3(x1_FPN_feature_3), self.cbam_T2_3(x2_FPN_feature_3), keepdim=True)
        at_out3 = self.upscore4(Converse_Attention_3)
        FPN_feature_3 = self.concat([self.conv3_diff(self.sigmoid(Converse_Attention_3) * torch.abs(x1_FPN_feature_3 - x2_FPN_feature_3))
                                    , self.conv3_and(self.sigmoid(Converse_Attention_3) * (x1_FPN_feature_3 + x2_FPN_feature_3))])
        PAN_feature_1 = FPN_feature_3
        PAN_feature_2 = self.C3_NUM7(self.concat([FPN_feature_2,self.p8(FPN_feature_3)]))
        PAN_feature_3 = self.C3_NUM8(self.concat([FPN_feature_1, self.p9(PAN_feature_2)]))

        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(PAN_feature_3)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), PAN_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), PAN_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.upscore2(d3))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(d4_out)
        return at_out1,at_out2,at_out3,d1_out,d2_out,d3_out,d4_out,dout
class FDDAFFNet_Seg_EPR_v2(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_Seg_EPR_v2, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        #resnet = models.resnet18(pretrained=True)
        ## -------------Feature Etraction--------------

        self.p3 = Conv(3, 64,3, 1, 1)
        self.p4 = Conv(64, 64, 3, 2, 1)
        self.encoder1 = nn.Sequential(
            EPRmoudul(64, 64, a=(1, 2, 4)),
            EPRmoudul(64, 64, a=(1, 2, 4), Down_sample=True)
        )
        self.encoder2 = nn.Sequential(
            EPRmoudul(64,128, a=(2, 4, 6)),
            EPRmoudul(128,128, a=(2, 4, 6)),

        )
        self.encoder3 = nn.Sequential(
            EPRmoudul(128,256, a=(3, 6, 9)),
            EPRmoudul(256, 256, a=(3, 6, 9), Down_sample=True)
        )
        self.encoder4 = nn.Sequential(
            EPRmoudul(256, 512, a=(4, 8, 12)),
            EPRmoudul(512, 512, a=(4, 8, 12), Down_sample=True)
        )
        self.SPPF_1 = SPPF(512,512,5)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.cbam_T1_1 = CBAM(256)
        self.cbam_T2_1 = CBAM(256)
        self.conv1_diff = CBAM(256)
        self.conv1_and = CBAM(256)
        self.cbam_T1_2 = CBAM(128)
        self.cbam_T2_2 = CBAM(128)
        self.conv2_diff = CBAM(128)
        self.conv2_and = CBAM(128)
        self.cbam_T1_3 = CBAM(128)
        self.cbam_T2_3 = CBAM(128)
        self.conv3_diff = CBAM(128)
        self.conv3_and = CBAM(128)

        ## -------------FPN--------------
        self.p6 = ConvBnRelu(512, 256, 3, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat = Concat(1)
        self.C3_NUM5 = EPRmoudul(512, 256)
        self.p7 = ConvBnRelu(256,128,3,1,1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.C3_NUM6 = EPRmoudul(256,128)
        ## -------------PAN--------------

        self.p8 = ConvBnRelu(256, 256, 3, 2, 1)
        self.C3_NUM7 = EPRmoudul(512,512)
        self.p9 = ConvBnRelu(512, 512, 3, 2, 1)
        self.C3_NUM8 = EPRmoudul(1024,1024)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512,3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512,3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512,3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024, 256,3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(256, 256,3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256,3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(512, 128,3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(128, 128,3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128,3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128, 64,3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_out = RefUnet(1,64)
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = Conv(512, 1, 1, 1, 0)
        self.outside_2 = Conv(256, 1, 1, 1, 0)
        self.outside_3 = Conv(128, 1, 1, 1, 0)
        self.outside_4 = Conv(64, 1, 1, 1, 0)

    def forward(self, x1,x2):
        ##-----------X1------------
        x1_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x1))))
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.SPPF_1(self.encoder4(x1_MAP2))
        x1_FPN_feature_1 = self.p6(x1_MAP3)
        x1_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x1_MAP2, self.upsample2(x1_FPN_feature_1)])))
        x1_FPN_feature_3 = self.C3_NUM6(self.concat([x1_MAP1, self.upsample2(x1_FPN_feature_2)]))
        ##-----------X2------------
        x2_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x2))))
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.SPPF_1(self.encoder4(x2_MAP2))
        x2_FPN_feature_1 = self.p6(x2_MAP3)
        x2_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x2_MAP2, self.upsample2(x2_FPN_feature_1)])))
        x2_FPN_feature_3 = self.C3_NUM6(self.concat([x2_MAP1, self.upsample2(x2_FPN_feature_2)]))
        # -----PAN-----
        Converse_Attention_1 = F.pairwise_distance(self.cbam_T1_1(x1_FPN_feature_1), self.cbam_T2_1(x2_FPN_feature_1),
                                                   keepdim=True)
        at_out1 = self.upscore16(Converse_Attention_1)
        FPN_feature_1 = self.concat(
            [self.conv1_diff(self.sigmoid(Converse_Attention_1) * torch.abs(x1_FPN_feature_1 - x2_FPN_feature_1))
                , self.conv1_and(self.sigmoid(Converse_Attention_1) * (x1_FPN_feature_1 + x2_FPN_feature_1))])
        Converse_Attention_2 = F.pairwise_distance(self.cbam_T1_2(x1_FPN_feature_2), self.cbam_T2_2(x2_FPN_feature_2),
                                                   keepdim=True)
        at_out2 = self.upscore8(Converse_Attention_2)
        FPN_feature_2 = self.concat(
            [self.conv2_diff(self.sigmoid(Converse_Attention_2) * torch.abs(x1_FPN_feature_2 - x2_FPN_feature_2))
                , self.conv2_and(self.sigmoid(Converse_Attention_2) * (x1_FPN_feature_2 + x2_FPN_feature_2))])
        Converse_Attention_3 = F.pairwise_distance(self.cbam_T1_3(x1_FPN_feature_3), self.cbam_T2_3(x2_FPN_feature_3),
                                                   keepdim=True)
        at_out3 = self.upscore4(Converse_Attention_3)
        FPN_feature_3 = self.concat(
            [self.conv3_diff(self.sigmoid(Converse_Attention_3) * torch.abs(x1_FPN_feature_3 - x2_FPN_feature_3))
                , self.conv3_and(self.sigmoid(Converse_Attention_3) * (x1_FPN_feature_3 + x2_FPN_feature_3))])
        PAN_feature_1 = FPN_feature_3
        PAN_feature_2 = self.C3_NUM7(self.concat([FPN_feature_2, self.p8(FPN_feature_3)]))
        PAN_feature_3 = self.C3_NUM8(self.concat([FPN_feature_1, self.p9(PAN_feature_2)]))

        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(PAN_feature_3)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), PAN_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), PAN_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.upscore2(d3))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(d4_out)
        return at_out1,at_out2,at_out3,d1_out,d2_out,d3_out,d4_out,dout
#Resblock
class FDDAFFNet_Seg_resblock(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_Seg_resblock, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.p3 = Conv(3, 64,3, 1, 1)
        self.p4 = Conv(64, 64, 3, 2, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,5)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.cbam_T1_1 = CBAM(256)
        self.cbam_T2_1 = CBAM(256)
        self.conv1_diff = CBAM(256)
        self.conv1_and = CBAM(256)
        self.cbam_T1_2 = CBAM(128)
        self.cbam_T2_2 = CBAM(128)
        self.conv2_diff = CBAM(128)
        self.conv2_and = CBAM(128)
        self.cbam_T1_3 = CBAM(128)
        self.cbam_T2_3 = CBAM(128)
        self.conv3_diff = CBAM(128)
        self.conv3_and = CBAM(128)

        ## -------------FPN--------------
        self.p6 = ConvBnRelu(512, 256, 3, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat = Concat(1)
        self.C3_NUM5 = Resblock_2(512, 256,3,1,1)
        self.p7 = ConvBnRelu(256,128,3,1,1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.C3_NUM6 = Resblock_2(256,128,3,1,1)
        ## -------------PAN--------------

        self.p8 = ConvBnRelu(256, 256, 3, 2, 1)
        self.C3_NUM7 = Resblock(512,3,1,1)
        self.p9 = ConvBnRelu(512, 512, 3, 2, 1)
        self.C3_NUM8 = Resblock(1024,3,1,1)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512,3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512,3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512,3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024, 256,3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(256, 256,3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256,3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(512, 128,3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(128, 128,3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128,3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128, 64,3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_out = RefUnet(1,64)
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = Conv(512, 1, 1, 1, 0)
        self.outside_2 = Conv(256, 1, 1, 1, 0)
        self.outside_3 = Conv(128, 1, 1, 1, 0)
        self.outside_4 = Conv(64, 1, 1, 1, 0)

    def forward(self, x1,x2):
        ##-----------X1------------
        x1_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x1))))
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.SPPF_1(self.encoder4(x1_MAP2))
        x1_FPN_feature_1 = self.p6(x1_MAP3)
        x1_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x1_MAP2, self.upsample2(x1_FPN_feature_1)])))
        x1_FPN_feature_3 = self.C3_NUM6(self.concat([x1_MAP1, self.upsample2(x1_FPN_feature_2)]))
        ##-----------X2------------
        x2_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x2))))
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.SPPF_1(self.encoder4(x2_MAP2))
        x2_FPN_feature_1 = self.p6(x2_MAP3)
        x2_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x2_MAP2, self.upsample2(x2_FPN_feature_1)])))
        x2_FPN_feature_3 = self.C3_NUM6(self.concat([x2_MAP1, self.upsample2(x2_FPN_feature_2)]))
        # -----PAN-----
        Converse_Attention_1 = F.pairwise_distance(self.cbam_T1_1(x1_FPN_feature_1), self.cbam_T2_1(x2_FPN_feature_1), keepdim=True)
        at_out1 = self.upscore16(Converse_Attention_1)
        FPN_feature_1 = self.concat([self.conv1_diff(self.sigmoid(Converse_Attention_1) * torch.abs(x1_FPN_feature_1 - x2_FPN_feature_1))
                                    , self.conv1_and(self.sigmoid(Converse_Attention_1) * (x1_FPN_feature_1 + x2_FPN_feature_1))])
        Converse_Attention_2 = F.pairwise_distance(self.cbam_T1_2(x1_FPN_feature_2), self.cbam_T2_2(x2_FPN_feature_2), keepdim=True)
        at_out2 = self.upscore8(Converse_Attention_2)
        FPN_feature_2 = self.concat([self.conv2_diff(self.sigmoid(Converse_Attention_2) * torch.abs(x1_FPN_feature_2 - x2_FPN_feature_2))
                                    , self.conv2_and(self.sigmoid(Converse_Attention_2) * (x1_FPN_feature_2 + x2_FPN_feature_2))])
        Converse_Attention_3 = F.pairwise_distance(self.cbam_T1_3(x1_FPN_feature_3), self.cbam_T2_3(x2_FPN_feature_3), keepdim=True)
        at_out3 = self.upscore4(Converse_Attention_3)
        FPN_feature_3 = self.concat([self.conv3_diff(self.sigmoid(Converse_Attention_3) * torch.abs(x1_FPN_feature_3 - x2_FPN_feature_3))
                                    , self.conv3_and(self.sigmoid(Converse_Attention_3) * (x1_FPN_feature_3 + x2_FPN_feature_3))])
        PAN_feature_1 = FPN_feature_3
        PAN_feature_2 = self.C3_NUM7(self.concat([FPN_feature_2,self.p8(FPN_feature_3)]))
        PAN_feature_3 = self.C3_NUM8(self.concat([FPN_feature_1, self.p9(PAN_feature_2)]))

        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(PAN_feature_3)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), PAN_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), PAN_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.upscore2(d3))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(d4_out)
        return at_out1,at_out2,at_out3,d1_out,d2_out,d3_out,d4_out,dout
class FDDAFFNet_Seg_resblock_2(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_Seg_resblock_2, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------


        self.p3 = Conv(3, 64,3, 1, 1)
        self.p4 = Conv(64, 64, 3, 2, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,5)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.cbam_T1_1 = CBAM(256)
        self.cbam_T2_1 = CBAM(256)
        self.conv1_diff = CBAM(256)
        self.conv1_and = CBAM(256)
        self.cbam_T1_2 = CBAM(128)
        self.cbam_T2_2 = CBAM(128)
        self.conv2_diff = CBAM(128)
        self.conv2_and = CBAM(128)
        self.cbam_T1_3 = CBAM(128)
        self.cbam_T2_3 = CBAM(128)
        self.conv3_diff = CBAM(128)
        self.conv3_and = CBAM(128)

        ## -------------FPN--------------
        self.p6 = ConvBnRelu(512, 256, 3, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat = Concat(1)
        self.C3_NUM5 = Resblock_2(512, 256,3,1,1)
        self.p7 = ConvBnRelu(256,128,3,1,1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.C3_NUM6 = Resblock_2(256,128,3,1,1)
        ## -------------PAN--------------

        self.p8 = ConvBnRelu(256, 256, 3, 2, 1)
        self.C3_NUM7 = Resblock(512,3,1,1)
        self.p9 = ConvBnRelu(512, 512, 3, 2, 1)
        self.C3_NUM8 = Resblock(1024,3,1,1)
        ## -------------decoder--------------
        self.decoder_t1_1 = Resblock_2(1024, 512,3, 1, 1)
        self.decoder_t1_2 = Resblock(512,3, 1, 1)
        self.decoder_t1_3 = Resblock(512,3, 1, 1)

        self.decoder_t2_1 = Resblock_2(1024, 256,3, 1, 1)
        self.decoder_t2_2 = Resblock(256,3, 1, 1)
        self.decoder_t2_3 = Resblock(256,3, 1, 1)

        self.decoder_t3_1 = Resblock_2(512, 128,3, 1, 1)
        self.decoder_t3_2 = Resblock(128,3, 1, 1)
        self.decoder_t3_3 = Resblock(128,3, 1, 1)

        self.decoder_t4_1 = Resblock_2(128, 64,3, 1, 1)
        self.decoder_t4_2 = Resblock(64,3, 1, 1)
        self.decoder_t4_3 = Resblock(64,3, 1, 1)
        self.decoder_out = RefUnet(1,64)
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = Conv(512, 1, 1, 1, 0)
        self.outside_2 = Conv(256, 1, 1, 1, 0)
        self.outside_3 = Conv(128, 1, 1, 1, 0)
        self.outside_4 = Conv(64, 1, 1, 1, 0)

    def forward(self, x1,x2):
        ##-----------X1------------
        x1_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x1))))
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.SPPF_1(self.encoder4(x1_MAP2))
        x1_FPN_feature_1 = self.p6(x1_MAP3)
        x1_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x1_MAP2, self.upsample2(x1_FPN_feature_1)])))
        x1_FPN_feature_3 = self.C3_NUM6(self.concat([x1_MAP1, self.upsample2(x1_FPN_feature_2)]))
        ##-----------X2------------
        x2_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x2))))
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.SPPF_1(self.encoder4(x2_MAP2))
        x2_FPN_feature_1 = self.p6(x2_MAP3)
        x2_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x2_MAP2, self.upsample2(x2_FPN_feature_1)])))
        x2_FPN_feature_3 = self.C3_NUM6(self.concat([x2_MAP1, self.upsample2(x2_FPN_feature_2)]))
        # -----PAN-----
        Converse_Attention_1 = F.pairwise_distance(self.cbam_T1_1(x1_FPN_feature_1), self.cbam_T2_1(x2_FPN_feature_1), keepdim=True)
        at_out1 = self.upscore16(Converse_Attention_1)
        FPN_feature_1 = self.concat([self.conv1_diff(self.sigmoid(Converse_Attention_1) * torch.abs(x1_FPN_feature_1 - x2_FPN_feature_1))
                                    , self.conv1_and(self.sigmoid(Converse_Attention_1) * (x1_FPN_feature_1 + x2_FPN_feature_1))])
        Converse_Attention_2 = F.pairwise_distance(self.cbam_T1_2(x1_FPN_feature_2), self.cbam_T2_2(x2_FPN_feature_2), keepdim=True)
        at_out2 = self.upscore8(Converse_Attention_2)
        FPN_feature_2 = self.concat([self.conv2_diff(self.sigmoid(Converse_Attention_2) * torch.abs(x1_FPN_feature_2 - x2_FPN_feature_2))
                                    , self.conv2_and(self.sigmoid(Converse_Attention_2) * (x1_FPN_feature_2 + x2_FPN_feature_2))])
        Converse_Attention_3 = F.pairwise_distance(self.cbam_T1_3(x1_FPN_feature_3), self.cbam_T2_3(x2_FPN_feature_3), keepdim=True)
        at_out3 = self.upscore4(Converse_Attention_3)
        FPN_feature_3 = self.concat([self.conv3_diff(self.sigmoid(Converse_Attention_3) * torch.abs(x1_FPN_feature_3 - x2_FPN_feature_3))
                                    , self.conv3_and(self.sigmoid(Converse_Attention_3) * (x1_FPN_feature_3 + x2_FPN_feature_3))])
        PAN_feature_1 = FPN_feature_3
        PAN_feature_2 = self.C3_NUM7(self.concat([FPN_feature_2,self.p8(FPN_feature_3)]))
        PAN_feature_3 = self.C3_NUM8(self.concat([FPN_feature_1, self.p9(PAN_feature_2)]))

        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(PAN_feature_3)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), PAN_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), PAN_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.upscore2(d3))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(d4_out)
        return at_out1,at_out2,at_out3,d1_out,d2_out,d3_out,d4_out,dout
#C3net
class FDDAFFNet_Seg_C3net(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_Seg_C3net, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet18(pretrained=True)
        ## -------------Feature Etraction--------------

        self.p3 = Conv(3, 64, 3, 1, 1)
        self.p4 = Conv(64, 64, 3, 2, 0)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,5)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        self.cbam_T1_1 = CBAM(256)
        self.cbam_T2_1 = CBAM(256)
        self.conv1_diff = ConvBnRelu(256, 256, 3, 1,1)
        self.conv1_and = ConvBnRelu(256, 256, 3, 1,1)
        self.cbam_T1_2 = CBAM(128)
        self.cbam_T2_2 = CBAM(128)
        self.conv2_diff = ConvBnRelu(128, 128, 3, 1,1)
        self.conv2_and = ConvBnRelu(128, 128, 3, 1,1)
        self.cbam_T1_3 = CBAM(128)
        self.cbam_T2_3 = CBAM(128)
        self.conv3_diff = ConvBnRelu(128, 128, 3, 1,1)
        self.conv3_and = ConvBnRelu(128, 128, 3, 1,1)

        ## -------------FPN--------------
        self.p6 = ConvBnRelu(512, 256, 1, 1, 0)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat = Concat(1)
        self.C3_NUM5 = C3_2(256, 256)
        self.p7 = ConvBnRelu(256,128, 1, 1, 0)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.C3_NUM6 = C3_2(128,128)
        ## -------------PAN--------------

        self.p8 = ConvBnRelu(256, 256, 3, 2, 1)
        self.C3_NUM7 = C3_3(512, 512)
        self.p9 = ConvBnRelu(512, 512, 3, 2, 1)
        self.C3_NUM8 = C3_3(1024, 1024)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512,3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512,3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512,3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024, 256,3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(256, 256,3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256,3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(512, 128,3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(128, 128,3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128,3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128, 64,3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64,3, 1, 1)
        self.decoder_out = RefUnet(1,64)
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = Conv(512, 1, 1, 1, 0)
        self.outside_2 = Conv(256, 1, 1, 1, 0)
        self.outside_3 = Conv(128, 1, 1, 1, 0)
        self.outside_4 = Conv(64, 1, 3, 1, 1)

    def forward(self, x1,x2):
        ##-----------X1------------
        x1_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x1))))
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.SPPF_1(self.encoder4(x1_MAP2))
        x1_FPN_feature_1 = self.p6(x1_MAP3)
        x1_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x1_MAP2, self.upsample2(x1_FPN_feature_1)])))
        x1_FPN_feature_3 = self.C3_NUM6(self.concat([x1_MAP1, self.upsample2(x1_FPN_feature_2)]))
        ##-----------X2------------
        x2_MAP1 = self.encoder2(self.encoder1(self.p4(self.p3(x2))))
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.SPPF_1(self.encoder4(x2_MAP2))
        x2_FPN_feature_1 = self.p6(x2_MAP3)
        x2_FPN_feature_2 = self.p7(self.C3_NUM5(self.concat([x2_MAP2, self.upsample2(x2_FPN_feature_1)])))
        x2_FPN_feature_3 = self.C3_NUM6(self.concat([x2_MAP1, self.upsample2(x2_FPN_feature_2)]))
        # -----PAN-----
        Converse_Attention_1 = F.pairwise_distance(self.cbam_T1_1(x1_FPN_feature_1), self.cbam_T2_1(x2_FPN_feature_1), keepdim=True)
        at_out1 = self.upscore16(Converse_Attention_1)
        FPN_feature_1 = self.concat([self.conv1_diff(self.sigmoid(Converse_Attention_1) * torch.abs(x1_FPN_feature_1 - x2_FPN_feature_1))
                                    , self.conv1_and(self.sigmoid(Converse_Attention_1) * (x1_FPN_feature_1 + x2_FPN_feature_1))])
        Converse_Attention_2 = F.pairwise_distance(self.cbam_T1_2(x1_FPN_feature_2), self.cbam_T2_2(x2_FPN_feature_2), keepdim=True)
        at_out2 = self.upscore8(Converse_Attention_2)
        FPN_feature_2 = self.concat([self.conv2_diff(self.sigmoid(Converse_Attention_2) * torch.abs(x1_FPN_feature_2 - x2_FPN_feature_2))
                                    , self.conv2_and(self.sigmoid(Converse_Attention_2) * (x1_FPN_feature_2 + x2_FPN_feature_2))])
        Converse_Attention_3 = F.pairwise_distance(self.cbam_T1_3(x1_FPN_feature_3), self.cbam_T2_3(x2_FPN_feature_3), keepdim=True)
        at_out3 = self.upscore4(Converse_Attention_3)
        FPN_feature_3 = self.concat([self.conv3_diff(self.sigmoid(Converse_Attention_3) * torch.abs(x1_FPN_feature_3 - x2_FPN_feature_3))
                                    , self.conv3_and(self.sigmoid(Converse_Attention_3) * (x1_FPN_feature_3 + x2_FPN_feature_3))])
        PAN_feature_1 = FPN_feature_3
        PAN_feature_2 = self.C3_NUM7(self.concat([FPN_feature_2,self.p8(FPN_feature_3)]))
        PAN_feature_3 = self.C3_NUM8(self.concat([FPN_feature_1, self.p9(PAN_feature_2)]))

        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(PAN_feature_3)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), PAN_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), PAN_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.upscore2(d3))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(d4_out)
        return at_out1,at_out2,at_out3,d1_out,d2_out,d3_out,d4_out,dout
