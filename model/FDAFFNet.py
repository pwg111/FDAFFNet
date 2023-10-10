import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import warnings
from .mobilenet import *
from .MobileNetV2 import *
import time

#----base-----
def getdist(feature_1,feature_2 ,ca):
    ing = feature_1 - feature_2 + 0.00001
    ing = torch.pow(ing,2) * ca
    ing = torch.sqrt(torch.sum(ing,dim=1,keepdim=True)) / feature_1.shape[1]
    return ing
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
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_drop=False, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        self.has_drop = has_drop
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.PReLU()
        if self.has_drop:
            self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        if self.has_drop:
            x = self.drop(x)

        return x
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
class BnReluConv(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_drop=False, has_bias=False):
        super(BnReluConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        self.has_drop = has_drop
        if self.has_bn:
            self.bn = norm_layer(in_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.PReLU()
        if self.has_drop:
            self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        x = self.conv(x)
        if self.has_drop:
            x = self.drop(x)

        return x
class Resblock(nn.Module):
    def __init__(self, c1,k =3,s=1,p=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.cv1 = ConvBnRelu(c1, c1, k, s, p)
        self.cv2 = nn.Conv2d(c1, c1,kernel_size=3,
                              stride=1, padding=1,)
        self.bn1 = nn.BatchNorm2d(c1, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.bn1(self.cv2(self.cv1(x)))
        return self.relu(x)
class Resblock_2(nn.Module):
    def __init__(self, c1,c2,k =3,s=1,p=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.cv1 = ConvBnRelu(c1, c2, k, s,p)
        self.cv2 = nn.Conv2d(c2, c2,kernel_size=3,
                              stride=1, padding=1,)
        #self.cv3 = ConvBnRelu(c1, c2, k, s,p)
        self.cv3 = ConvBnRelu(c1, c2, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(c2, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.cv3(x) + self.bn1(self.cv2(self.cv1(x)))
        return self.relu(x)

#----spp-----
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=3):  # equivalent to SPP(k=(5, 9, 13))
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
class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
class ASPPF(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super(ASPPF, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(atrous_block1)
        atrous_block12 = self.atrous_block12(atrous_block6)
        atrous_block18 = self.atrous_block18(atrous_block12)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
class SPPFCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

#----CSAM-----
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x1) + self.avg_pool(x2))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x1) + self.max_pool(x2))))
        out = avg_out + max_out
        return self.sigmoid(out)
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

#----FDAFFM-----
class FDAFFM(nn.Module):
    def __init__(self, in_planes, ratio = 4 , kernel_size = 3):
        super(FDAFFM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.concat = Concat(1)
        self.atres = Resblock(1, 3, 1, 1)
        self.conv1 = ConvBnRelu(in_planes*2, in_planes, 1, 1, 0)
        self.conv2 = ConvBnRelu(in_planes, in_planes, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_planes, in_planes*2,kernel_size=1,
                              stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_planes*2, eps=1e-5)
        self.relu = nn.PReLU()
        self.sigmoid = torch.tanh
    def forward(self, x1,x2):
        ca = self.ca(x1,x2)
        da = self.atres(self.sigmoid(getdist(x1, x2, ca)))
        #x = self.concat(
        #    [da * torch.abs(x1 - x2)
        #        ,  da * (x1 + x2)])
        x = da * self.concat([x1,x2])
        x = self.relu(x + self.conv3(self.conv2(self.conv1(self.relu(self.bn1(x))))))
        x = self.sa(x) * x
        return x
class Decoder(nn.Module):
    def __init__(self, in_planes):
        super(Decoder, self).__init__()
        self.decoder_t1_1 = ConvBnRelu(in_planes, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t1_3 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(in_planes+in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t2_3 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(in_planes, in_planes//2, 3, 1, 1, has_drop=True)
        self.decoder_t3_2 = ConvBnRelu( in_planes//2,  in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t3_3 = ConvBnRelu(in_planes//4, in_planes//4, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(in_planes//2, in_planes//2, 3, 1, 1, has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(in_planes//2, in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t4_3 = ConvBnRelu(in_planes//4, in_planes//8, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(in_planes//4, in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(in_planes//4, 64, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        self.concat = Concat(1)
        self.trans_conv1 = nn.ConvTranspose2d(in_planes // 2, in_planes // 2, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(in_planes // 8, in_planes // 8, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size=2, stride=2)
        self.trans_conv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore32 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(in_planes // 2, 1, 3, 1, 1)
        self.outside_2 = nn.Conv2d(in_planes // 2, 1, 3, 1, 1)
        self.outside_3 = nn.Conv2d(in_planes // 4, 1, 3, 1, 1)
        self.outside_4 = nn.Conv2d(in_planes // 8, 1, 3, 1, 1)

    def forward(self, out):
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(out[4])))
        d1_out = self.outside_1(self.upscore32(d1))
        d2 = self.decoder_t2_3(
            self.decoder_t2_2(self.decoder_t2_1(self.concat([d1, out[3]]))))
        d2_out = self.outside_2(self.upscore32(d2))
        d3 = self.decoder_t3_3(
            self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv1(d2), out[2]]))))
        d3_out = self.outside_3(self.upscore16(d3))
        d4 =  self.decoder_t4_3(
            self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv2(d3), out[1]]))))
        d4_out = self.outside_4(self.upscore8(d4))
        dout = self.decoder_out(self.decoder_t5_3(
            self.trans_conv5(self.decoder_t5_2(self.trans_conv4(self.decoder_t5_1(self.concat([self.trans_conv3(d4), out[0]])))))))
        return d1_out,d2_out,d3_out,d4_out,dout
class Decoder_2(nn.Module):
    def __init__(self, in_planes):
        super(Decoder_2, self).__init__()
        self.decoder_t1_1 = ConvBnRelu(in_planes, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t1_3 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(in_planes+in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1, has_drop=True)
        self.decoder_t2_3 = ConvBnRelu(in_planes // 2, in_planes // 2, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(in_planes, in_planes//2, 3, 1, 1, has_drop=True)
        self.decoder_t3_2 = ConvBnRelu( in_planes//2,  in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t3_3 = ConvBnRelu(in_planes//4, in_planes//4, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(in_planes//2, in_planes//2, 3, 1, 1, has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(in_planes//2, in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t4_3 = ConvBnRelu(in_planes//4, in_planes//8, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(in_planes//4, in_planes//4, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(in_planes//4, 64, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        self.concat = Concat(1)
        self.trans_conv1 = nn.ConvTranspose2d(in_planes // 2, in_planes // 2, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(in_planes // 8, in_planes // 8, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(in_planes // 2, 1, 3, 1, 1)
        self.outside_2 = nn.Conv2d(in_planes // 2, 1, 3, 1, 1)
        self.outside_3 = nn.Conv2d(in_planes // 4, 1, 3, 1, 1)
        self.outside_4 = nn.Conv2d(in_planes // 8, 1, 3, 1, 1)

    def forward(self, out):
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(out[4])))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(
            self.decoder_t2_2(self.decoder_t2_1(self.concat([d1, out[3]]))))
        d2_out = self.outside_2(self.upscore16(d2))
        d3 = self.decoder_t3_3(
            self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv1(d2), out[2]]))))
        d3_out = self.outside_3(self.upscore8(d3))
        d4 =  self.decoder_t4_3(
            self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv2(d3), out[1]]))))
        d4_out = self.outside_4(self.upscore4(d4))
        dout = self.decoder_out(self.trans_conv4(self.decoder_t5_3(
            self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv3(d4), out[0]]))))))
        return d1_out,d2_out,d3_out,d4_out,dout
class FDAFFM_sig(nn.Module):
    def __init__(self, in_planes, ratio = 4 , kernel_size = 3):
        super(FDAFFM_sig, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.concat = Concat(1)
        self.atres = Resblock(1, 3, 1, 1)
        self.conv1 = ConvBnRelu(in_planes*2, in_planes, 1, 1, 0)
        self.conv2 = ConvBnRelu(in_planes, in_planes, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_planes, in_planes*2,kernel_size=1,
                              stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_planes*2, eps=1e-5)
        self.relu = nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1,x2):
        ca = self.ca(x1,x2)
        da = self.atres(self.sigmoid(getdist(x1, x2, ca)))
        #x = self.concat(
        #    [da * torch.abs(x1 - x2)
        #        ,  da * (x1 + x2)])
        x = da * self.concat([x1,x2])
        x = self.relu(x + self.conv3(self.conv2(self.conv1(self.relu(self.bn1(x))))))
        x = self.sa(x) * x
        return x
class FDAFFM_resBeforeconv(nn.Module):
    def __init__(self, in_planes, ratio = 4 , kernel_size = 3):
        super(FDAFFM_resBeforeconv, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.concat = Concat(1)
        self.atres = Resblock(1, 3, 1, 1)
        self.conv1 = BnReluConv(in_planes*2, in_planes*2, 3, 1, 1)
        self.sigmoid = torch.tanh
    def forward(self, x1,x2):
        ca = self.ca(x1,x2)
        da = self.atres(self.sigmoid(getdist(x1, x2, ca)))
        x = da * self.concat([x1,x2])
        x = x + self.conv1(x)
        x = self.sa(x) * x
        return x
#----DSAM----
class DeepAttention(nn.Module):
    def __init__(self, inchannel,Down_sample = 2):
        super().__init__()
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=Down_sample)
        self.maskconv = nn.Conv2d(1, 1, 3, 1, 1)
        #self.afterconv = BnReluConv(inchannel, inchannel, 3, 1, 1)
        #self.outconv = BnReluConv(inchannel, inchannel, 3, 1, 1)
        self.afterconv = BnReluConv(inchannel, inchannel, 3, 1, 1)
        self.outconv = ConvBnRelu(inchannel, inchannel, 3, 1, 1)
    def forward(self, d1, d1_out):
        deep_attention = self.maskconv(self.maxpool(self.relu(d1_out)))
        out = self.outconv(self.afterconv(deep_attention * d1) + d1)
        return out

#----model----
class FDAFFNet(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,3)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)


        '''
        self.CDS0 = CDSwithoutsa(64)
        self.CDS1 = CDSwithoutsa(128)
        self.CDS2 = CDSwithoutsa(256)
        self.CDS3 = CDSwithoutsa(512)
        self.CDS4 = CDSwithoutsa(512)'''
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1,has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1,has_drop=True)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1,has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1,has_drop=True)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1,has_drop=True)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1,has_drop=True)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1,has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1,has_drop=True)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1,has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1,has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)

        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

        ## -------------Mask guidance--------------
        self.DeepAttention1 = DeepAttention(512, 16)
        self.DeepAttention2 = DeepAttention(256, 8)
        self.DeepAttention3 = DeepAttention(128, 4)
        self.DeepAttention4 = DeepAttention(64, 2)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d1_mask = self.DeepAttention1(d1, d1_out)
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1_mask), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d2_mask = self.DeepAttention2(d2, d2_out)
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2_mask), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d3_mask = self.DeepAttention3(d3, d3_out)
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3_mask), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        d4_mask = self.DeepAttention4(d4, d4_out)
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4_mask), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return d1_out, d2_out, d3_out, d4_out, dout
class FDAFFNet_l(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_l, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,3)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        '''
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        '''

        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout
class FDAFFNet_l_CSPC(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_l_CSPC, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPFCSPC(512,512)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        '''
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        '''

        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout
class FDAFFNet_l_nospp(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_l_nospp, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        #self.SPPF_1 = SPPFCSPC(512,512)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(x1_MAP3)

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(x2_MAP3)

        # -----Fusion-----
        '''
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        '''

        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout
class FDAFFNet_l_ASPP(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_l_ASPP, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.ASPP= SPPF(512,512)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        t1 = time.time()
        x1_MAP4 = self.ASPP(x1_MAP3)
        t2 = time.time()
        x1_MAP4 = self.d(x1_MAP4)

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        t3 = time.time()
        x2_MAP4 = self.ASPP(x2_MAP3)
        t4 = time.time()
        x2_MAP4 = self.d(x2_MAP4)
        # -----Fusion-----
        '''
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        '''

        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout,t4+t2-t3-t1
class FDDAFFNet_X(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_X, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(512,512,3)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()
        '''
        self.CDS0 = Concat(1)
        self.CDS1 = Concat(1)
        self.CDS2 = Concat(1)
        self.CDS3 = Concat(1)
        self.CDS4 = Concat(1)
        '''
        '''
        self.CDS0 = CDS_oringal(64)
        self.CDS1 = CDS_oringal(128)
        self.CDS2 = CDS_oringal(256)
        self.CDS3 = CDS_oringal(512)
        self.CDS4 = CDS_oringal(512)
        '''
        '''
        self.CDS0 = CDS_res_D(64)
        self.CDS1 = CDS_res_D(128)
        self.CDS2 = CDS_res_D(256)
        self.CDS3 = CDS_res_D(512)
        self.CDS4 = CDS_res_D(512)
        '''
        self.CDS0 = FDAFFM_resBeforeconv(64)
        self.CDS1 = FDAFFM_resBeforeconv(128)
        self.CDS2 = FDAFFM_resBeforeconv(256)
        self.CDS3 = FDAFFM_resBeforeconv(512)
        self.CDS4 = FDAFFM_resBeforeconv(512)


        '''
        self.CDS0 = CDSwithoutsa(64)
        self.CDS1 = CDSwithoutsa(128)
        self.CDS2 = CDSwithoutsa(256)
        self.CDS3 = CDSwithoutsa(512)
        self.CDS4 = CDSwithoutsa(512)'''
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1,has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1,has_drop=True)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1,has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1,has_drop=True)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1,has_drop=True)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1,has_drop=True)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1,has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1,has_drop=True)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1,has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1,has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)

        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

        ## -------------Mask guidance--------------
        self.DeepAttention1 = DeepAttention(512, 16)
        self.DeepAttention2 = DeepAttention(256, 8)
        self.DeepAttention3 = DeepAttention(128, 4)
        self.DeepAttention4 = DeepAttention(64, 2)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d1_mask = self.DeepAttention1(d1, d1_out)
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1_mask), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d2_mask = self.DeepAttention2(d2, d2_out)
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2_mask), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d3_mask = self.DeepAttention3(d3, d3_out)
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3_mask), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        d4_mask = self.DeepAttention4(d4, d4_out)
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4_mask), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return d1_out, d2_out, d3_out, d4_out, dout

#----other----
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

class BottleNeck1(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c1, 3, 1,1, g=g)
    def forward(self, x):
        return self.cv2(self.cv1(x)) + x
class C3_1(nn.Module):
    def __init__(self, c1,c2, n=3, shortcut=True, g=1, e=0.5, Down_sample = False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[BottleNeck1(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.Down_sample = Down_sample
        if Down_sample:
            self.down = Conv(c2, c2, 3,2,1)
    def forward(self, x):
        if self.Down_sample:
            return self.down(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)))
        else:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


#----other decoder----

class FDAFFNet_EPR(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_EPR, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        ## -------------Feature Etraction--------------

        self.e1 = ConvBnRelu(3, 16, 3, 1,1)
        self.e2 = ConvBnRelu(16, 16, 3, 2,1)
        self.encoder1 = nn.Sequential(
            EPRmoudul(16, 32, a=(1, 2, 4)),
            EPRmoudul(32, 32, a=(1, 2, 4), Down_sample=True)
        )
        self.encoder2 = nn.Sequential(
            EPRmoudul(32, 64, a=(2, 4, 6)),
            EPRmoudul(64, 64, a=(2, 4, 6)),
            EPRmoudul(64, 64, a=(2, 4, 6)),
            EPRmoudul(64, 64, a=(2, 4, 6), Down_sample=True)
        )
        self.encoder3 = nn.Sequential(
            EPRmoudul(64, 128, a=(3, 6, 9)),
            EPRmoudul(128, 128, a=(9, 15, 21), Down_sample=True)
        )
        #self.stern_conv1 = Conv(128, 128, 3, 1,1)
        self.stern_conv1 = SPPF(128, 128, 3)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(16)
        self.CDS1 = FDAFFM(32)
        self.CDS2 = FDAFFM(64)
        self.CDS3 = FDAFFM(128)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(256, 128, 3, 1, 1,has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(256, 64, 3, 1, 1,has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(128, 32, 3, 1, 1,has_drop=True)
        self.decoder_t3_2 = ConvBnRelu(32, 32, 3, 1, 1,has_drop=True)

        self.decoder_t4_1 = ConvBnRelu(64, 16, 3, 1, 1,has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(16, 16, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(16, 16, 3, 1, 1,has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(16, 16, 3, 1, 1)

        self.decoder_out = nn.Conv2d(16, 1, 3, 1, 1)
        #
        ## -------------Bilinear Upsampling--------------
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(64, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(32, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(16, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.e2(self.e1(x1))
        x1_MAP1 = self.encoder1(x1_MAP0)
        x1_MAP2 = self.encoder2(x1_MAP1)
        x1_MAP3 = self.stern_conv1(self.encoder3(x1_MAP2))


        ##-----------X2------------
        x2_MAP0 = self.e2(self.e1(x2))
        x2_MAP1 = self.encoder1(x2_MAP0)
        x2_MAP2 = self.encoder2(x2_MAP1)
        x2_MAP3 = self.stern_conv1(self.encoder3(x2_MAP2))

        # -----Fusion-----
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        # ------decoder-tranconv------
        d1 = self.decoder_t1_2(self.decoder_t1_1(fusion_feature_3))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_2(self.decoder_t2_1(self.concat([self.upscore2(d1), fusion_feature_2])))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_2(self.decoder_t3_1(self.concat([self.upscore2(d2), fusion_feature_1])))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_2(self.decoder_t4_1(self.concat([self.upscore2(d3), fusion_feature_0])))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_2(self.decoder_t5_1(self.upscore2(d4))))
        return  d1_out, d2_out, d3_out, d4_out, dout

class FDAFFNet_Mobilev2(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_Mobilev2, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        ## -------------Feature Etraction--------------
        self.mobilenet = mobilenet_v2(pretrained=True)
        # stage 1
        self.SPPF_1 = SPPF(320,320,3)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(16)
        self.CDS1 = FDAFFM(24)
        self.CDS2 = FDAFFM(32)
        self.CDS3 = FDAFFM(96)
        self.CDS4 = FDAFFM(320)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(640, 192, 3, 1, 1,has_drop=True)
        self.decoder_t1_2 = ConvBnRelu(192, 192, 3, 1, 1,has_drop=True)
        self.decoder_t1_3 = ConvBnRelu(192, 192, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(192 +192, 192, 3, 1, 1,has_drop=True)
        self.decoder_t2_2 = ConvBnRelu(192, 64, 3, 1, 1,has_drop=True)
        self.decoder_t2_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(64 + 64, 64, 3, 1, 1,has_drop=True)
        self.decoder_t3_2 = ConvBnRelu(64, 48, 3, 1, 1,has_drop=True)
        self.decoder_t3_3 = ConvBnRelu(48, 48, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(48 + 48, 48, 3, 1, 1,has_drop=True)
        self.decoder_t4_2 = ConvBnRelu(48, 32, 3, 1, 1,has_drop=True)
        self.decoder_t4_3 = ConvBnRelu(32, 32, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1,has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1,has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)

        self.decoder_t6_1 = ConvBnRelu(32, 16, 3, 1, 1,has_drop=True)
        self.decoder_t6_2 = ConvBnRelu(16, 16, 3, 1, 1,has_drop=True)
        self.decoder_t6_3 = ConvBnRelu(16, 16, 3, 1, 1)

        self.decoder_out = nn.Conv2d(16, 1, 3, 1, 1)

        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.trans_conv5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.upscore32 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(64, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(48, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(32, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(32, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0, x1_MAP1, x1_MAP2, x1_MAP3, x1_MAP4= self.mobilenet(x1)
        x1_MAP4 = self.SPPF_1(x1_MAP4)
        ##-----------X2------------
        x2_MAP0, x2_MAP1, x2_MAP2, x2_MAP3, x2_MAP4 = self.mobilenet(x2)
        x2_MAP4 = self.SPPF_1(x2_MAP4)
        # -----Fusion-----
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d1_out = self.outside_1(self.upscore16(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d2_out = self.outside_2(self.upscore8(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d3_out = self.outside_3(self.upscore4(d4))
        d5 = self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0]))))
        d4_out = self.outside_4(self.upscore2(d5))
        dout = self.decoder_out(self.decoder_t6_3(self.decoder_t6_2(self.decoder_t6_1(self.trans_conv5(d5)))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout

class FDAFFNet_res50(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_res50, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet50(pretrained=True)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = resnet.layer1  # 256^2*64
        # stage 2
        self.encoder2 = resnet.layer2  # 128^2*128
        # stage 3
        self.encoder3 = resnet.layer3  # 64^2*256
        # stage 4
        self.encoder4 = resnet.layer4  # 32^2*512

        self.SPPF_1 = SPPF(2048,2048,3)
        self.d = Conv(2048, 2048, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(256)
        self.CDS1 = FDAFFM(512)
        self.CDS2 = FDAFFM(1024)
        self.CDS3 = FDAFFM(2048)
        self.CDS4 = FDAFFM(2048)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(4096, 2048, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(2048, 2048, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(2048, 2048, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(4096 +2048, 2048, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(2048, 1024, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(1024, 1024, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(1024 + 2048, 1024, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(512 + 1024, 512, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(256 +512, 256, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(256, 256, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(256, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------

        self.trans_conv1 = nn.ConvTranspose2d(2048, 2048, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(2048, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(1024, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(256, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        '''
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1,x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2,x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3,x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4,x2_MAP4)
        '''

        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout

class FDDAFFNet_C3Net(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDDAFFNet_C3Net, self).__init__()
        ## -------------Feature Etraction--------------

        self.e1 = Conv(n_channels, 64, 3, 1, 1)
        # stage 1
        self.encoder1 = C3_1(64,64)  # 256^2*64
        # stage 2
        self.encoder2 = C3_1(64,128,Down_sample = True)  # 128^2*128
        # stage 3
        self.encoder3 = C3_1(128,256,Down_sample = True)  # 64^2*256
        # stage 4
        self.encoder4 = C3_1(256,512,Down_sample = True)  # 32^2*512

        self.SPPF_1 = SPPF(512,512,3)
        self.d = Conv(512, 512, 3, 2, 1)
        ## -----------diffusion-------------
        self.concat = Concat(1)
        self.sigmoid = nn.Sigmoid()

        self.CDS0 = FDAFFM(64)
        self.CDS1 = FDAFFM(128)
        self.CDS2 = FDAFFM(256)
        self.CDS3 = FDAFFM(512)
        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.decoder_t1_1 = ConvBnRelu(1024, 512, 3, 1, 1)
        self.decoder_t1_2 = ConvBnRelu(512, 512, 3, 1, 1)
        self.decoder_t1_3 = ConvBnRelu(512, 512, 3, 1, 1)

        self.decoder_t2_1 = ConvBnRelu(1024 +512, 512, 3, 1, 1)
        self.decoder_t2_2 = ConvBnRelu(512, 256, 3, 1, 1)
        self.decoder_t2_3 = ConvBnRelu(256, 256, 3, 1, 1)

        self.decoder_t3_1 = ConvBnRelu(256 + 512, 256, 3, 1, 1)
        self.decoder_t3_2 = ConvBnRelu(256, 128, 3, 1, 1)
        self.decoder_t3_3 = ConvBnRelu(128, 128, 3, 1, 1)

        self.decoder_t4_1 = ConvBnRelu(128 + 256, 128, 3, 1, 1)
        self.decoder_t4_2 = ConvBnRelu(128, 64, 3, 1, 1)
        self.decoder_t4_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_t5_1 = ConvBnRelu(64 +128, 64, 3, 1, 1)
        self.decoder_t5_2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.decoder_t5_3 = ConvBnRelu(64, 64, 3, 1, 1)

        self.decoder_out = nn.Conv2d(64, 1, 3, 1, 1)
        '''
        self.decoder_t5_1 = ConvBnRelu(64, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_2 = ConvBnRelu(32, 32, 3, 1, 1, has_drop=True)
        self.decoder_t5_3 = ConvBnRelu(32, 32, 3, 1, 1)
        self.decoder_out = nn.Conv2d(32, 1, 3, 1, 1)'''


        #
        ## -------------Bilinear Upsampling--------------
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # --------mutil-supervise--------------
        self.outside_1 = nn.Conv2d(512, 1, 3,1,1)
        self.outside_2 = nn.Conv2d(256, 1, 3,1,1)
        self.outside_3 = nn.Conv2d(128, 1, 3,1,1)
        self.outside_4 = nn.Conv2d(64, 1, 3,1,1)

    def forward(self, x1, x2):
        ##-----------X1------------
        x1_MAP0 = self.encoder1(self.e1(x1))
        x1_MAP1 = self.encoder2(x1_MAP0)
        x1_MAP2 = self.encoder3(x1_MAP1)
        x1_MAP3 = self.encoder4(x1_MAP2)
        x1_MAP4 = self.d(self.SPPF_1(x1_MAP3))

        ##-----------X2------------
        x2_MAP0 = self.encoder1(self.e1(x2))
        x2_MAP1 = self.encoder2(x2_MAP0)
        x2_MAP2 = self.encoder3(x2_MAP1)
        x2_MAP3 = self.encoder4(x2_MAP2)
        x2_MAP4 = self.d(self.SPPF_1(x2_MAP3))

        # -----Fusion-----
        fusion_feature_0 = self.CDS0(x1_MAP0, x2_MAP0)

        fusion_feature_1 = self.CDS1(x1_MAP1, x2_MAP1)

        fusion_feature_2 = self.CDS2(x1_MAP2, x2_MAP2)

        fusion_feature_3 = self.CDS3(x1_MAP3, x2_MAP3)

        fusion_feature_4 = self.CDS4(x1_MAP4, x2_MAP4)
        # ------decoder-tranconv------
        d1 = self.decoder_t1_3(self.decoder_t1_2(self.decoder_t1_1(fusion_feature_4)))
        d1_out = self.outside_1(self.upscore16(d1))
        d2 = self.decoder_t2_3(self.decoder_t2_2(self.decoder_t2_1(self.concat([self.trans_conv1(d1), fusion_feature_3]))))
        d2_out = self.outside_2(self.upscore8(d2))
        d3 = self.decoder_t3_3(self.decoder_t3_2(self.decoder_t3_1(self.concat([self.trans_conv2(d2), fusion_feature_2]))))
        d3_out = self.outside_3(self.upscore4(d3))
        d4 = self.decoder_t4_3(self.decoder_t4_2(self.decoder_t4_1(self.concat([self.trans_conv3(d3), fusion_feature_1]))))
        d4_out = self.outside_4(self.upscore2(d4))
        dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.concat([self.trans_conv4(d4), fusion_feature_0])))))
        #dout = self.decoder_out(self.decoder_t5_3(self.decoder_t5_2(self.decoder_t5_1(self.trans_conv4(d4)))))
        return  d1_out, d2_out, d3_out, d4_out, dout

class FDAFFNet_test(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels=3, norm_layer=nn.BatchNorm2d):
        super(FDAFFNet_test, self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        ## -------------Feature Etraction--------------

        self.e1 = Conv(3, 512, 3, 1, 1)
        #self.SPPF_1 = SPPF(2048,2048,3)

        self.CDS4 = FDAFFM(512)
        ## -------------decoder--------------
        self.output = Conv(512, 1, 3, 1, 1)


    def forward(self, x1, x2):
        f1 = self.e1(x1)
        f2 = self.e1(x2)
        o1 = self.CDS4(f1,f2)
        return self.output(o1)
        ##-----------X1-----------

