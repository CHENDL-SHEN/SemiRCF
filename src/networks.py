import torch
import torch.nn as nn
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# this is scale5 (deconv 32!!)
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        needing_down=False,
        norm_layer: Optional[Callable[..., nn.Module]] = None ) -> None:

        super(Bottleneck, self).__init__()
        if needing_down:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
            )
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.conv2 = conv3x3(width, width, groups = groups, dilation = dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.SideSeq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 21, kernel_size=1, bias=False),
        )
    def forward(self, x):
        return self.SideSeq(x)

'RESNET50'
class EdgeGeneratorRES50(nn.Module):
    def __init__(self, num_block = [3, 4, 6, 3]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = Bottleneck(inplanes = 64, planes = 64, stride = 1, needing_down = True)
        self.conv2b = Bottleneck(inplanes = 256, planes = 64, stride = 1)
        self.conv2c = Bottleneck(inplanes = 256, planes = 64, stride = 1)

        self.conv3a = Bottleneck(inplanes = 256, planes = 128, stride = 2, needing_down = True)
        self.conv3b = Bottleneck(inplanes = 512, planes = 128, stride = 1)
        self.conv3c = Bottleneck(inplanes = 512, planes = 128, stride = 1)
        self.conv3d = Bottleneck(inplanes = 512, planes = 128, stride = 1)

        self.conv4a = Bottleneck(inplanes = 512, planes = 256, stride = 2, needing_down = True)
        self.conv4b = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4c = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4d = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4e = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4f = Bottleneck(inplanes = 1024, planes = 256, stride = 1)

        self.conv5a = Bottleneck(inplanes = 1024, planes = 512, dilation = 2, needing_down = True)
        self.conv5b = Bottleneck(inplanes = 2048, planes = 512, stride = 1)
        self.conv5c = Bottleneck(inplanes = 2048, planes = 512, stride = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.conv1_down = DownBlock(in_channels = 64, out_channels = 32)
        self.conv2a_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2b_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2c_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv3a_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3b_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3c_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3d_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv4a_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4b_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4c_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4d_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4e_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4f_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv5a_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5b_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5c_down = DownBlock(in_channels = 2048, out_channels = 128)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)

        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)


    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv1_maxpool = self.maxpool(conv1)
        conv2a = self.conv2a(conv1_maxpool)
        conv2b = self.conv2b(conv2a)
        conv2c = self.conv2c(conv2b)
        conv3a = self.conv3a(conv2c)
        conv3b = self.conv3b(conv3a)
        conv3c = self.conv3c(conv3b)
        conv3d = self.conv3d(conv3c)
        conv4a = self.conv4a(conv3d)
        conv4b = self.conv4b(conv4a)
        conv4c = self.conv4c(conv4b)
        conv4d = self.conv4d(conv4c)
        conv4e = self.conv4e(conv4d)
        conv4f = self.conv4f(conv4e)
        conv5a = self.conv5a(conv4f)
        conv5b = self.conv5b(conv5a)
        conv5c = self.conv5c(conv5b)

        conv1_down = self.conv1_down(conv1)
        conv2a_down = self.conv2a_down(conv2a)
        conv2b_down = self.conv2b_down(conv2b)
        conv2c_down = self.conv2c_down(conv2c)
        conv3a_down = self.conv3a_down(conv3a)
        conv3b_down = self.conv3b_down(conv3b)
        conv3c_down = self.conv3c_down(conv3c)
        conv3d_down = self.conv3d_down(conv3d)
        conv4a_down = self.conv4a_down(conv4a)
        conv4b_down = self.conv4b_down(conv4b)
        conv4c_down = self.conv4c_down(conv4c)
        conv4d_down = self.conv4d_down(conv4d)
        conv4e_down = self.conv4e_down(conv4e)
        conv4f_down = self.conv4f_down(conv4f)
        conv5a_down = self.conv5a_down(conv5a)
        conv5b_down = self.conv5b_down(conv5b)
        conv5c_down = self.conv5c_down(conv5c)

        so1_out = self.score_dsn1(conv1_down)
        so2_out = self.score_dsn2(conv2a_down + conv2b_down + conv2c_down)
        so3_out = self.score_dsn3(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_out = self.score_dsn4(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down)
        so5_out = self.score_dsn5(conv5a_down + conv5b_down + conv5c_down)

        so1_outs = self.score_dsn1s(conv1_down)
        so2_outs = self.score_dsn2s(conv2a_down + conv2b_down + conv2c_down)
        so3_outs = self.score_dsn3s(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_outs = self.score_dsn4s(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down)
        so5_outs = self.score_dsn5s(conv5a_down + conv5b_down + conv5c_down)

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2a_down + conv2b_down + conv2c_down, f_weight_deconv2, stride=2 )
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3a_down + conv3b_down + conv3c_down + conv3d_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down, f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5a_down + conv5b_down + conv5c_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s =torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]

        return results,resultss,f_so_fuse

'VGG16'
class EdgeGenerator_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 =nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 =nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]


        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        so1_outs = self.score_dsn1s(conv1_1_down + conv1_2_down)
        so2_outs = self.score_dsn2s(conv2_1_down + conv2_2_down)
        so3_outs = self.score_dsn3s(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_outs = self.score_dsn4s(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_outs = self.score_dsn5s(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_1_down + conv1_2_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2_1_down + conv2_2_down, f_weight_deconv2, stride=2)
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3_1_down + conv3_2_down + conv3_3_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4_1_down + conv4_2_down + conv4_3_down, f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5_1_down + conv5_2_down + conv5_3_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)


        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s = torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]
        return results,resultss,f_so_fuse

'RESNEXT50'
class EdgeGeneratorRESNEXT50(nn.Module):
    def __init__(self, num_block = [3, 4, 6, 3]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = Bottleneck(inplanes = 64, planes = 64, stride = 1, needing_down = True, groups = 32, base_width = 4)
        self.conv2b = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 4)
        self.conv2c = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 4)

        self.conv3a = Bottleneck(inplanes = 256, planes = 128, stride = 2, needing_down = True, groups = 32, base_width = 4)
        self.conv3b = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 4)
        self.conv3c = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 4)
        self.conv3d = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 4)

        self.conv4a = Bottleneck(inplanes = 512, planes = 256, stride = 2, needing_down = True, groups = 32, base_width = 4)
        self.conv4b = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 4)
        self.conv4c = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 4)
        self.conv4d = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 4)
        self.conv4e = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 4)
        self.conv4f = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 4)

        self.conv5a = Bottleneck(inplanes = 1024, planes = 512, dilation = 2, needing_down = True, groups = 32, base_width = 4)
        self.conv5b = Bottleneck(inplanes = 2048, planes = 512, stride = 1, groups = 32, base_width = 4)
        self.conv5c = Bottleneck(inplanes = 2048, planes = 512, stride = 1, groups = 32, base_width = 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.conv1_down = DownBlock(in_channels = 64, out_channels = 32)
        self.conv2a_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2b_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2c_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv3a_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3b_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3c_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3d_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv4a_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4b_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4c_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4d_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4e_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4f_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv5a_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5b_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5c_down = DownBlock(in_channels = 2048, out_channels = 128)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)

        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv1_maxpool = self.maxpool(conv1)
        conv2a = self.conv2a(conv1_maxpool)
        conv2b = self.conv2b(conv2a)
        conv2c = self.conv2c(conv2b)
        conv3a = self.conv3a(conv2c)
        conv3b = self.conv3b(conv3a)
        conv3c = self.conv3c(conv3b)
        conv3d = self.conv3d(conv3c)
        conv4a = self.conv4a(conv3d)
        conv4b = self.conv4b(conv4a)
        conv4c = self.conv4c(conv4b)
        conv4d = self.conv4d(conv4c)
        conv4e = self.conv4e(conv4d)
        conv4f = self.conv4f(conv4e)
        conv5a = self.conv5a(conv4f)
        conv5b = self.conv5b(conv5a)
        conv5c = self.conv5c(conv5b)

        conv1_down = self.conv1_down(conv1)
        conv2a_down = self.conv2a_down(conv2a)
        conv2b_down = self.conv2b_down(conv2b)
        conv2c_down = self.conv2c_down(conv2c)
        conv3a_down = self.conv3a_down(conv3a)
        conv3b_down = self.conv3b_down(conv3b)
        conv3c_down = self.conv3c_down(conv3c)
        conv3d_down = self.conv3d_down(conv3d)
        conv4a_down = self.conv4a_down(conv4a)
        conv4b_down = self.conv4b_down(conv4b)
        conv4c_down = self.conv4c_down(conv4c)
        conv4d_down = self.conv4d_down(conv4d)
        conv4e_down = self.conv4e_down(conv4e)
        conv4f_down = self.conv4f_down(conv4f)
        conv5a_down = self.conv5a_down(conv5a)
        conv5b_down = self.conv5b_down(conv5b)
        conv5c_down = self.conv5c_down(conv5c)

        so1_out = self.score_dsn1(conv1_down)
        so2_out = self.score_dsn2(conv2a_down + conv2b_down + conv2c_down)
        so3_out = self.score_dsn3(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_out = self.score_dsn4(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down)
        so5_out = self.score_dsn5(conv5a_down + conv5b_down + conv5c_down)

        so1_outs = self.score_dsn1s(conv1_down)
        so2_outs = self.score_dsn2s(conv2a_down + conv2b_down + conv2c_down)
        so3_outs = self.score_dsn3s(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_outs = self.score_dsn4s(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down)
        so5_outs = self.score_dsn5s(conv5a_down + conv5b_down + conv5c_down)

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2a_down + conv2b_down + conv2c_down, f_weight_deconv2, stride=2 )
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3a_down + conv3b_down + conv3c_down + conv3d_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down, f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5a_down + conv5b_down + conv5c_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s =torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]

        return results,resultss,f_so_fuse

'RESNET101'
class EdgeGeneratorRES101(nn.Module):
    def __init__(self, num_block = [3, 4, 23, 3]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = Bottleneck(inplanes = 64, planes = 64, stride = 1, needing_down = True)
        self.conv2b = Bottleneck(inplanes = 256, planes = 64, stride = 1)
        self.conv2c = Bottleneck(inplanes = 256, planes = 64, stride = 1)

        self.conv3a = Bottleneck(inplanes = 256, planes = 128, stride = 2, needing_down = True)
        self.conv3b = Bottleneck(inplanes = 512, planes = 128, stride = 1)
        self.conv3c = Bottleneck(inplanes = 512, planes = 128, stride = 1)
        self.conv3d = Bottleneck(inplanes = 512, planes = 128, stride = 1)

        self.conv4a = Bottleneck(inplanes = 512, planes = 256, stride = 2, needing_down = True)
        self.conv4b = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4c = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4d = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4e = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4f = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4g = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4h = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4i = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4j = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4k = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4l = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4m = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4n = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4o = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4p = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4q = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4r = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4s = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4t = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4u = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4v = Bottleneck(inplanes = 1024, planes = 256, stride = 1)
        self.conv4w = Bottleneck(inplanes = 1024, planes = 256, stride = 1)

        self.conv5a = Bottleneck(inplanes = 1024, planes = 512, stride = 1, dilation=2, needing_down = True)
        self.conv5b = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2)
        self.conv5c = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2)

        self.conv1_down = DownBlock(in_channels = 64, out_channels = 32)
        self.conv2a_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2b_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2c_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv3a_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3b_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3c_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3d_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv4a_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4b_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4c_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4d_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4e_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4f_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4g_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4h_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4i_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4j_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4k_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4l_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4m_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4n_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4o_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4p_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4q_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4r_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4s_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4t_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4u_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4v_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4w_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv5a_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5b_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5c_down = DownBlock(in_channels = 2048, out_channels = 128)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)


    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv1_maxpool = self.maxpool(conv1)
        conv2a = self.conv2a(conv1_maxpool)
        conv2b = self.conv2b(conv2a)
        conv2c = self.conv2c(conv2b)
        conv3a = self.conv3a(conv2c)
        conv3b = self.conv3b(conv3a)
        conv3c = self.conv3c(conv3b)
        conv3d = self.conv3d(conv3c)
        conv4a = self.conv4a(conv3d)
        conv4b = self.conv4b(conv4a)
        conv4c = self.conv4c(conv4b)
        conv4d = self.conv4d(conv4c)
        conv4e = self.conv4e(conv4d)
        conv4f = self.conv4f(conv4e)
        conv4g = self.conv4g(conv4f)
        conv4h = self.conv4h(conv4g)
        conv4i = self.conv4i(conv4h)
        conv4j = self.conv4j(conv4i)
        conv4k = self.conv4k(conv4j)
        conv4l = self.conv4l(conv4k)
        conv4m = self.conv4m(conv4l)
        conv4n = self.conv4n(conv4m)
        conv4o = self.conv4o(conv4n)
        conv4p = self.conv4p(conv4o)
        conv4q = self.conv4q(conv4p)
        conv4r = self.conv4r(conv4q)
        conv4s = self.conv4s(conv4r)
        conv4t = self.conv4t(conv4s)
        conv4u = self.conv4u(conv4t)
        conv4v = self.conv4v(conv4u)
        conv4w = self.conv4w(conv4v)
        conv5a = self.conv5a(conv4w)
        conv5b = self.conv5b(conv5a)
        conv5c = self.conv5c(conv5b)

        conv1_down = self.conv1_down(conv1)
        conv2a_down = self.conv2a_down(conv2a)
        conv2b_down = self.conv2b_down(conv2b)
        conv2c_down = self.conv2c_down(conv2c)
        conv3a_down = self.conv3a_down(conv3a)
        conv3b_down = self.conv3b_down(conv3b)
        conv3c_down = self.conv3c_down(conv3c)
        conv3d_down = self.conv3d_down(conv3d)
        conv4a_down = self.conv4a_down(conv4a)
        conv4b_down = self.conv4b_down(conv4b)
        conv4c_down = self.conv4c_down(conv4c)
        conv4d_down = self.conv4d_down(conv4d)
        conv4e_down = self.conv4e_down(conv4e)
        conv4f_down = self.conv4f_down(conv4f)
        conv4g_down = self.conv4g_down(conv4g)
        conv4h_down = self.conv4h_down(conv4h)
        conv4i_down = self.conv4i_down(conv4i)
        conv4j_down = self.conv4j_down(conv4j)
        conv4k_down = self.conv4k_down(conv4k)
        conv4l_down = self.conv4l_down(conv4l)
        conv4m_down = self.conv4m_down(conv4m)
        conv4n_down = self.conv4n_down(conv4n)
        conv4o_down = self.conv4o_down(conv4o)
        conv4p_down = self.conv4p_down(conv4p)
        conv4q_down = self.conv4q_down(conv4q)
        conv4r_down = self.conv4r_down(conv4r)
        conv4s_down = self.conv4s_down(conv4s)
        conv4t_down = self.conv4t_down(conv4t)
        conv4u_down = self.conv4u_down(conv4u)
        conv4v_down = self.conv4v_down(conv4v)
        conv4w_down = self.conv4w_down(conv4w)
        conv5a_down = self.conv5a_down(conv5a)
        conv5b_down = self.conv5b_down(conv5b)
        conv5c_down = self.conv5c_down(conv5c)

        so1_out = self.score_dsn1(conv1_down)
        so2_out = self.score_dsn2(conv2a_down + conv2b_down + conv2c_down)
        so3_out = self.score_dsn3(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_out = self.score_dsn4(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                  conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                  conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                  conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_out = self.score_dsn5(conv5a_down + conv5b_down + conv5c_down)

        so1_outs = self.score_dsn1s(conv1_down)
        so2_outs = self.score_dsn2s(conv2a_down + conv2b_down + conv2c_down)
        so3_outs = self.score_dsn3s(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_outs = self.score_dsn4s(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                    conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                    conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                    conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_outs = self.score_dsn5s(conv5a_down + conv5b_down + conv5c_down)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2a_down + conv2b_down + conv2c_down, f_weight_deconv2, stride=2)
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3a_down + conv3b_down + conv3c_down + conv3d_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                                           conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                                           conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                                           conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down
                                                           , f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5a_down + conv5b_down + conv5c_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s = torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]

        return results,resultss,f_so_fuse

'RESNEXT101'
class EdgeGeneratorRESNEXT101(nn.Module):
    def __init__(self, num_block = [3, 4, 23, 3]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = Bottleneck(inplanes = 64, planes = 64, stride = 1, needing_down = True, groups = 32, base_width = 8)
        self.conv2b = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 8)
        self.conv2c = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 8)

        self.conv3a = Bottleneck(inplanes = 256, planes = 128, stride = 2, needing_down = True, groups = 32, base_width = 8)
        self.conv3b = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)
        self.conv3c = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)
        self.conv3d = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)

        self.conv4a = Bottleneck(inplanes = 512, planes = 256, stride = 2, needing_down = True, groups = 32, base_width = 8)
        self.conv4b = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4c = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4d = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4e = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4f = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4g = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4h = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4i = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4j = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4k = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4l = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4m = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4n = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4o = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4p = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4q = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4r = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4s = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4t = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4u = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4v = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4w = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)

        self.conv5a = Bottleneck(inplanes = 1024, planes = 512, stride = 1, dilation=2, needing_down = True, groups = 32, base_width = 8)
        self.conv5b = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2, groups = 32, base_width = 8)
        self.conv5c = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2, groups = 32, base_width = 8)

        self.conv1_down = DownBlock(in_channels = 64, out_channels = 32)
        self.conv2a_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2b_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2c_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv3a_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3b_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3c_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3d_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv4a_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4b_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4c_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4d_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4e_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4f_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4g_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4h_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4i_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4j_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4k_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4l_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4m_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4n_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4o_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4p_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4q_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4r_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4s_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4t_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4u_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4v_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4w_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv5a_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5b_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5c_down = DownBlock(in_channels = 2048, out_channels = 128)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)


    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv1_maxpool = self.maxpool(conv1)
        conv2a = self.conv2a(conv1_maxpool)
        conv2b = self.conv2b(conv2a)
        conv2c = self.conv2c(conv2b)
        conv3a = self.conv3a(conv2c)
        conv3b = self.conv3b(conv3a)
        conv3c = self.conv3c(conv3b)
        conv3d = self.conv3d(conv3c)
        conv4a = self.conv4a(conv3d)
        conv4b = self.conv4b(conv4a)
        conv4c = self.conv4c(conv4b)
        conv4d = self.conv4d(conv4c)
        conv4e = self.conv4e(conv4d)
        conv4f = self.conv4f(conv4e)
        conv4g = self.conv4g(conv4f)
        conv4h = self.conv4h(conv4g)
        conv4i = self.conv4i(conv4h)
        conv4j = self.conv4j(conv4i)
        conv4k = self.conv4k(conv4j)
        conv4l = self.conv4l(conv4k)
        conv4m = self.conv4m(conv4l)
        conv4n = self.conv4n(conv4m)
        conv4o = self.conv4o(conv4n)
        conv4p = self.conv4p(conv4o)
        conv4q = self.conv4q(conv4p)
        conv4r = self.conv4r(conv4q)
        conv4s = self.conv4s(conv4r)
        conv4t = self.conv4t(conv4s)
        conv4u = self.conv4u(conv4t)
        conv4v = self.conv4v(conv4u)
        conv4w = self.conv4w(conv4v)
        conv5a = self.conv5a(conv4w)
        conv5b = self.conv5b(conv5a)
        conv5c = self.conv5c(conv5b)

        conv1_down = self.conv1_down(conv1)
        conv2a_down = self.conv2a_down(conv2a)
        conv2b_down = self.conv2b_down(conv2b)
        conv2c_down = self.conv2c_down(conv2c)
        conv3a_down = self.conv3a_down(conv3a)
        conv3b_down = self.conv3b_down(conv3b)
        conv3c_down = self.conv3c_down(conv3c)
        conv3d_down = self.conv3d_down(conv3d)
        conv4a_down = self.conv4a_down(conv4a)
        conv4b_down = self.conv4b_down(conv4b)
        conv4c_down = self.conv4c_down(conv4c)
        conv4d_down = self.conv4d_down(conv4d)
        conv4e_down = self.conv4e_down(conv4e)
        conv4f_down = self.conv4f_down(conv4f)
        conv4g_down = self.conv4g_down(conv4g)
        conv4h_down = self.conv4h_down(conv4h)
        conv4i_down = self.conv4i_down(conv4i)
        conv4j_down = self.conv4j_down(conv4j)
        conv4k_down = self.conv4k_down(conv4k)
        conv4l_down = self.conv4l_down(conv4l)
        conv4m_down = self.conv4m_down(conv4m)
        conv4n_down = self.conv4n_down(conv4n)
        conv4o_down = self.conv4o_down(conv4o)
        conv4p_down = self.conv4p_down(conv4p)
        conv4q_down = self.conv4q_down(conv4q)
        conv4r_down = self.conv4r_down(conv4r)
        conv4s_down = self.conv4s_down(conv4s)
        conv4t_down = self.conv4t_down(conv4t)
        conv4u_down = self.conv4u_down(conv4u)
        conv4v_down = self.conv4v_down(conv4v)
        conv4w_down = self.conv4w_down(conv4w)
        conv5a_down = self.conv5a_down(conv5a)
        conv5b_down = self.conv5b_down(conv5b)
        conv5c_down = self.conv5c_down(conv5c)

        so1_out = self.score_dsn1(conv1_down)
        so2_out = self.score_dsn2(conv2a_down + conv2b_down + conv2c_down)
        so3_out = self.score_dsn3(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_out = self.score_dsn4(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                  conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                  conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                  conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_out = self.score_dsn5(conv5a_down + conv5b_down + conv5c_down)

        so1_outs = self.score_dsn1s(conv1_down)
        so2_outs = self.score_dsn2s(conv2a_down + conv2b_down + conv2c_down)
        so3_outs = self.score_dsn3s(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_outs = self.score_dsn4s(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                    conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                    conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                    conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_outs = self.score_dsn5s(conv5a_down + conv5b_down + conv5c_down)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2a_down + conv2b_down + conv2c_down, f_weight_deconv2, stride=2)
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3a_down + conv3b_down + conv3c_down + conv3d_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                                           conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                                           conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                                           conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down
                                                           , f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5a_down + conv5b_down + conv5c_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s = torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]

        return results,resultss,f_so_fuse

'RESNET101'
class EdgeGenerator(nn.Module):
    def __init__(self, num_block = [3, 4, 23, 3]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = Bottleneck(inplanes = 64, planes = 64, stride = 1, needing_down = True, groups = 32, base_width = 8)
        self.conv2b = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 8)
        self.conv2c = Bottleneck(inplanes = 256, planes = 64, stride = 1, groups = 32, base_width = 8)

        self.conv3a = Bottleneck(inplanes = 256, planes = 128, stride = 2, needing_down = True, groups = 32, base_width = 8)
        self.conv3b = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)
        self.conv3c = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)
        self.conv3d = Bottleneck(inplanes = 512, planes = 128, stride = 1, groups = 32, base_width = 8)

        self.conv4a = Bottleneck(inplanes = 512, planes = 256, stride = 2, needing_down = True, groups = 32, base_width = 8)
        self.conv4b = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4c = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4d = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4e = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4f = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4g = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4h = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4i = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4j = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4k = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4l = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4m = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4n = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4o = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4p = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4q = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4r = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4s = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4t = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4u = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4v = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)
        self.conv4w = Bottleneck(inplanes = 1024, planes = 256, stride = 1, groups = 32, base_width = 8)

        self.conv5a = Bottleneck(inplanes = 1024, planes = 512, stride = 1, dilation=2, needing_down = True, groups = 32, base_width = 8)
        self.conv5b = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2, groups = 32, base_width = 8)
        self.conv5c = Bottleneck(inplanes = 2048, planes = 512, stride = 1, dilation=2, groups = 32, base_width = 8)

        self.conv1_down = DownBlock(in_channels = 64, out_channels = 32)
        self.conv2a_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2b_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv2c_down = DownBlock(in_channels = 256, out_channels = 64)
        self.conv3a_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3b_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3c_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv3d_down = DownBlock(in_channels = 512, out_channels = 64)
        self.conv4a_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4b_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4c_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4d_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4e_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4f_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4g_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4h_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4i_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4j_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4k_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4l_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4m_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4n_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4o_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4p_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4q_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4r_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4s_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4t_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4u_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4v_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv4w_down = DownBlock(in_channels = 1024, out_channels = 128)
        self.conv5a_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5b_down = DownBlock(in_channels = 2048, out_channels = 128)
        self.conv5c_down = DownBlock(in_channels = 2048, out_channels = 128)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)


    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv1_maxpool = self.maxpool(conv1)
        conv2a = self.conv2a(conv1_maxpool)
        conv2b = self.conv2b(conv2a)
        conv2c = self.conv2c(conv2b)
        conv3a = self.conv3a(conv2c)
        conv3b = self.conv3b(conv3a)
        conv3c = self.conv3c(conv3b)
        conv3d = self.conv3d(conv3c)
        conv4a = self.conv4a(conv3d)
        conv4b = self.conv4b(conv4a)
        conv4c = self.conv4c(conv4b)
        conv4d = self.conv4d(conv4c)
        conv4e = self.conv4e(conv4d)
        conv4f = self.conv4f(conv4e)
        conv4g = self.conv4g(conv4f)
        conv4h = self.conv4h(conv4g)
        conv4i = self.conv4i(conv4h)
        conv4j = self.conv4j(conv4i)
        conv4k = self.conv4k(conv4j)
        conv4l = self.conv4l(conv4k)
        conv4m = self.conv4m(conv4l)
        conv4n = self.conv4n(conv4m)
        conv4o = self.conv4o(conv4n)
        conv4p = self.conv4p(conv4o)
        conv4q = self.conv4q(conv4p)
        conv4r = self.conv4r(conv4q)
        conv4s = self.conv4s(conv4r)
        conv4t = self.conv4t(conv4s)
        conv4u = self.conv4u(conv4t)
        conv4v = self.conv4v(conv4u)
        conv4w = self.conv4w(conv4v)
        conv5a = self.conv5a(conv4w)
        conv5b = self.conv5b(conv5a)
        conv5c = self.conv5c(conv5b)

        conv1_down = self.conv1_down(conv1)
        conv2a_down = self.conv2a_down(conv2a)
        conv2b_down = self.conv2b_down(conv2b)
        conv2c_down = self.conv2c_down(conv2c)
        conv3a_down = self.conv3a_down(conv3a)
        conv3b_down = self.conv3b_down(conv3b)
        conv3c_down = self.conv3c_down(conv3c)
        conv3d_down = self.conv3d_down(conv3d)
        conv4a_down = self.conv4a_down(conv4a)
        conv4b_down = self.conv4b_down(conv4b)
        conv4c_down = self.conv4c_down(conv4c)
        conv4d_down = self.conv4d_down(conv4d)
        conv4e_down = self.conv4e_down(conv4e)
        conv4f_down = self.conv4f_down(conv4f)
        conv4g_down = self.conv4g_down(conv4g)
        conv4h_down = self.conv4h_down(conv4h)
        conv4i_down = self.conv4i_down(conv4i)
        conv4j_down = self.conv4j_down(conv4j)
        conv4k_down = self.conv4k_down(conv4k)
        conv4l_down = self.conv4l_down(conv4l)
        conv4m_down = self.conv4m_down(conv4m)
        conv4n_down = self.conv4n_down(conv4n)
        conv4o_down = self.conv4o_down(conv4o)
        conv4p_down = self.conv4p_down(conv4p)
        conv4q_down = self.conv4q_down(conv4q)
        conv4r_down = self.conv4r_down(conv4r)
        conv4s_down = self.conv4s_down(conv4s)
        conv4t_down = self.conv4t_down(conv4t)
        conv4u_down = self.conv4u_down(conv4u)
        conv4v_down = self.conv4v_down(conv4v)
        conv4w_down = self.conv4w_down(conv4w)
        conv5a_down = self.conv5a_down(conv5a)
        conv5b_down = self.conv5b_down(conv5b)
        conv5c_down = self.conv5c_down(conv5c)

        so1_out = self.score_dsn1(conv1_down)
        so2_out = self.score_dsn2(conv2a_down + conv2b_down + conv2c_down)
        so3_out = self.score_dsn3(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_out = self.score_dsn4(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                  conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                  conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                  conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_out = self.score_dsn5(conv5a_down + conv5b_down + conv5c_down)

        so1_outs = self.score_dsn1s(conv1_down)
        so2_outs = self.score_dsn2s(conv2a_down + conv2b_down + conv2c_down)
        so3_outs = self.score_dsn3s(conv3a_down + conv3b_down + conv3c_down + conv3d_down)
        so4_outs = self.score_dsn4s(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                    conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                    conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                    conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down)
        so5_outs = self.score_dsn5s(conv5a_down + conv5b_down + conv5c_down)

        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2a_down + conv2b_down + conv2c_down, f_weight_deconv2, stride=2)
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3a_down + conv3b_down + conv3c_down + conv3d_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4a_down + conv4b_down + conv4c_down + conv4d_down + conv4e_down + conv4f_down +
                                                           conv4g_down + conv4h_down + conv4i_down + conv4j_down + conv4k_down + conv4l_down +
                                                           conv4m_down + conv4n_down + conv4o_down + conv4p_down + conv4q_down + conv4r_down +
                                                           conv4s_down + conv4t_down + conv4u_down + conv4v_down + conv4w_down
                                                           , f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5a_down + conv5b_down + conv5c_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s = torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]

        return results,resultss,f_so_fuse
    def __init__(self):
        super(EdgeGenerator_VGG16, self).__init__()
        # lr 1 2 decay 1 0
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 =nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 =nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_dsn1s = nn.Conv2d(21, 1, 1)
        self.score_dsn2s = nn.Conv2d(21, 1, 1)
        self.score_dsn3s = nn.Conv2d(21, 1, 1)
        self.score_dsn4s = nn.Conv2d(21, 1, 1)
        self.score_dsn5s = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)
        self.score_finals = nn.Conv2d(5, 1, 1)



    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]


        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        so1_outs = self.score_dsn1s(conv1_1_down + conv1_2_down)
        so2_outs = self.score_dsn2s(conv2_1_down + conv2_2_down)
        so3_outs = self.score_dsn3s(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_outs = self.score_dsn4s(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_outs = self.score_dsn5s(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        ## transpose and crop way
        f_weight_deconv2 = make_bilinear_weights(4, 21).cuda()
        f_weight_deconv3 = make_bilinear_weights(8, 21).cuda()
        f_weight_deconv4 = make_bilinear_weights(16, 21).cuda()
        f_weight_deconv5 = make_bilinear_weights(32, 21).cuda()

        f_upsample1 = conv1_1_down + conv1_2_down
        f_upsample2 = torch.nn.functional.conv_transpose2d(conv2_1_down + conv2_2_down, f_weight_deconv2, stride=2)
        f_upsample3 = torch.nn.functional.conv_transpose2d(conv3_1_down + conv3_2_down + conv3_3_down, f_weight_deconv3, stride=4)
        f_upsample4 = torch.nn.functional.conv_transpose2d(conv4_1_down + conv4_2_down + conv4_3_down, f_weight_deconv4, stride=8)
        f_upsample5 = torch.nn.functional.conv_transpose2d(conv5_1_down + conv5_2_down + conv5_3_down, f_weight_deconv5, stride=8)

        f_so1 = crop(f_upsample1, img_H, img_W)
        f_so2 = crop(f_upsample2, img_H, img_W)
        f_so3 = crop(f_upsample3, img_H, img_W)
        f_so4 = crop(f_upsample4, img_H, img_W)
        f_so5 = crop(f_upsample5, img_H, img_W)

        f_so_fuse = torch.cat((f_so1, f_so2, f_so3, f_so4, f_so5), dim=1)


        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        upsample2s = torch.nn.functional.conv_transpose2d(so2_outs, weight_deconv2, stride=2)
        upsample3s = torch.nn.functional.conv_transpose2d(so3_outs, weight_deconv3, stride=4)
        upsample4s = torch.nn.functional.conv_transpose2d(so4_outs, weight_deconv4, stride=8)
        upsample5s = torch.nn.functional.conv_transpose2d(so5_outs, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        so1s = crop(so1_outs, img_H, img_W)
        so2s = crop(upsample2s, img_H, img_W)
        so3s = crop(upsample3s, img_H, img_W)
        so4s = crop(upsample4s, img_H, img_W)
        so5s = crop(upsample5s, img_H, img_W)


        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fusecats = torch.cat((so1s, so2s, so3s, so4s, so5s), dim=1)
        fuse = self.score_final(fusecat)
        fuses = self.score_finals(fusecats)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]

        resultss = [so1s, so2s, so3s, so4s, so5s, fuses]
        resultss = [torch.sigmoid(r) for r in resultss]
        return results,resultss,f_so_fuse


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]

'''
class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid


        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),  # F.to_tensor(img).float() 0-1之间
            #spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),  # F.to_tensor(img).float() 0-1之间
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            #spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        #print(outputs.shape)
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs
'''

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.fc = nn.Sequential(
            nn.Linear(105, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

        #if init_weights:
        #    self.init_weights()

    def forward(self, x):
        outputs = self.fc(x)

        return outputs


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module