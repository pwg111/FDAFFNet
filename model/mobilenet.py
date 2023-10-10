from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):#卷积、BN、激活函数组合
    def __init__(self,
                 in_planes: int,#输入channel数
                 out_planes: int,#输出channel数
                 kernel_size: int = 3,#卷积核大小
                 stride: int = 1,#步距
                 groups: int = 1,#group conv参数
                 norm_layer: Optional[Callable[..., nn.Module]] = None,#norm层用什么
                 activation_layer: Optional[Callable[..., nn.Module]] = None):#激活函数用什么
        padding = (kernel_size - 1) // 2#计算padding
        if norm_layer is None:#如果没传norm，默认为BN
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:#如果没传激活函数，默认为Relu6
            activation_layer = nn.ReLU6
            #调用父类初始化
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):#SE层
    def __init__(self, input_c: int, squeeze_factor: int = 4):#输入通道数，超参r（默认为4）
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)#获取第一个全连接层输出通道数
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)#第一个全连接层，由于HxW=1x1，所以这里的1x1卷积与全连接层一样
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)#第二个全连接层

    def forward(self, x: Tensor) -> Tensor:#前向传播过程
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))#采用自适应平均池化层来充当全局平均池化
        #以下为激活操作，先过一个全连接层降维在过非线性激活函数，接着是第二个全连接层还原channel数，最后与输入对应相乘
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:#v3的基本模块配置信息
    def __init__(self,
                 input_c: int,#输入的channel数
                 kernel: int,#depthwise conv卷积核大小
                 expanded_c: int,#第一个1x1卷积要升到的维数
                 out_c: int,#输出channel
                 use_se: bool,#是否使用SE模块
                 activation: str,#激活函数类型
                 stride: int,#depth wise conv步距
                 width_multi: float):#α超参，调整channel数
        #下面为属性赋值
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod#静态方法，不用产生实例就能用
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):#v3基本模块
    def __init__(self,
                 cnf: InvertedResidualConfig,#配置信息
                 norm_layer: Callable[..., nn.Module]):#norm层
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:#stride只能是1或2
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)#检查能否用short cut

        layers: List[nn.Module] = []#模块集合
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU#获取激活函数

        # expand
        if cnf.expanded_c != cnf.input_c:#如果没有升维就不加入下面的1x1卷积
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:#添加SE模块
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project，降维
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)#获得模块集合
        self.out_channels = cnf.out_c#给出输出的通道数
        self.is_strided = cnf.stride > 1#给出是否进行了了分辨率降低操作

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:#需要则shortcut连接
            result += x

        return result


class MobileNetV3(nn.Module):#整体网络
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],#各层的配置信息
                 last_channel: int,#分类器前的是输出通道数
                 num_classes: int = 1000,#类个数
                 block: Optional[Callable[..., nn.Module]] = None,#基本模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None):#norm
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:#没有传入配置信息，报错
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and#配置信息不是列表，报错
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:#没有传入基本模块就默认为v3的
            block = InvertedResidual

        if norm_layer is None:#没传入norm就默认用BN
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)#传入两个默认参数

        layers: List[nn.Module] = []#模块集合

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c#获取最后一层的输出通道数
        lastconv_output_c = 6 * lastconv_input_c#获取输出通道数，为上面的6倍
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)#特征提取部分网络
        self.avgpool = nn.AdaptiveAvgPool2d(1)#池化
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)#展平，从batch后面一个维度开始展平
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:#前向传播过程
        return self._forward_impl(x)

class MobileNetV3_Encoder(nn.Module):#整体网络
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],#各层的配置信息
                 last_channel: int,#分类器前的是输出通道数
                 block: Optional[Callable[..., nn.Module]] = None,#基本模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None):#norm
        super(MobileNetV3_Encoder, self).__init__()

        if not inverted_residual_setting:#没有传入配置信息，报错
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and#配置信息不是列表，报错
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:#没有传入基本模块就默认为v3的
            block = InvertedResidual

        if norm_layer is None:#没传入norm就默认用BN
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)#传入两个默认参数

        layers: List[nn.Module] = []#模块集合

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c#获取最后一层的输出通道数
        lastconv_output_c = lastconv_input_c#获取输出通道数，为上面的6倍
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features_0 = nn.Sequential(*layers[0:2])#特征提取部分网络
        self.features_1 = nn.Sequential(*layers[2:4])  # 特征提取部分网络
        self.features_2 = nn.Sequential(*layers[4:7])  # 特征提取部分网络
        self.features_3 = nn.Sequential(*layers[7:13])  # 特征提取部分网络
        self.features_4 = nn.Sequential(*layers[13:])  # 特征提取部分网络
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x1 = self.features_0(x)
        x2 = self.features_1(x1)
        x3 = self.features_2(x2)
        x4 = self.features_3(x3)
        x5 = self.features_4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x: Tensor) -> Tensor:#前向传播过程
        return self._forward_impl(x)
def mobilenet_v3_large(num_classes: int = 1000,#类个数
                       reduced_tail: bool = False) -> MobileNetV3:#尾部是否缩减
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0#α
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)#传入默认参数，后面就不用传了
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)#传入默认参数

    reduce_divider = 2 if reduced_tail else 1#是否缩减尾部

    inverted_residual_setting = [#各层配置列表
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5，分类器前的维度

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,#构建模型
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
def mobilenet_v3_Ecoder(num_classes: int = 1000,#类个数
                       reduced_tail: bool = False) -> MobileNetV3:#尾部是否缩减
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0#α
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)#传入默认参数，后面就不用传了
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)#传入默认参数

    reduce_divider = 2 if reduced_tail else 1#是否缩减尾部

    inverted_residual_setting = [#各层配置列表
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(160)  # C5，分类器前的维度

    return MobileNetV3_Encoder(inverted_residual_setting=inverted_residual_setting,#构建模型
                       last_channel=last_channel)
if __name__ == '__main__':
    net = mobilenet_v3_small()
    print('seccess')