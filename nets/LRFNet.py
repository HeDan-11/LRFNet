import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

savepath = './models/'


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = int(out_channels / 2)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1))
        self.residual = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, k, stride=1, padding=int(k/2), groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1),)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        primary_conv = self.primary_conv(x)
        residual = self.residual(primary_conv)
        x1 = torch.cat([primary_conv, residual], dim=1)
        x1 = x1 + self.shortcut(x)
        x1 = channel_shuffle(x1, 2)
        return x1

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv2 = nn.Conv2d(1, 16, 3, 1, 1)

        en_c = [16, 32, 64]
        de_c = [64, 32, 1]

        self.bn2 = nn.BatchNorm2d(en_c[0]*3)
        self.bn3 = nn.BatchNorm2d(en_c[1]*3)

        self.conv3 = Unit(en_c[0], en_c[0], k=7)
        self.conv4 = Unit(en_c[0], en_c[1], k=5)
        self.conv5 = Unit(en_c[1], en_c[2], k=3)
        # self.dual_conv1 = Unit(2, en_c[0], k=3)  # nn.Conv2d(2, 16, 3, 1, 1)
        self.dual_conv1 = nn.Conv2d(2, en_c[0], 3, 1, 1)
        self.dual_conv21 = Unit(en_c[0], en_c[0], k=3)
        self.dual_conv2 = Unit(en_c[0]*3, en_c[1], k=3)
        self.dual_conv3 = Unit(en_c[1]*3, en_c[2], k=3)
        self.conv6 = Unit(en_c[2]*3+en_c[1]*2+en_c[0]*2, de_c[0], k=3)
        self.conv7 = Unit(de_c[0], de_c[1], k=3)

        self.conv8 = nn.Conv2d(de_c[1], de_c[2], 1, 1, 0)

        self.norm1 = nn.BatchNorm2d(en_c[2]*3+en_c[1]*2+en_c[0]*2)
        self.norm2 = nn.BatchNorm2d(de_c[0])
        self.norm3 = nn.BatchNorm2d(de_c[1])

    def forward(self, x, y):


        temz1 = self.dual_conv1(torch.cat((x, y), 1))
        temx2 = self.conv2(x)
        temy2 = self.conv2(y)

        temx3 = self.conv3(temx2)
        temy3 = self.conv3(temy2)
        temz1 = self.dual_conv21(temz1)
        temz2 = self.dual_conv2(self.bn2(torch.cat((temx3, temy3, temz1), 1)))

        temx4 = self.conv4(temx3)
        temy4 = self.conv4(temy3)
        temz3 = self.dual_conv3(self.bn3(torch.cat((temx4, temy4, temz2), 1)))

        temx5 = self.conv5(temx4)
        temy5 = self.conv5(temy4)

        tem = torch.cat((temx3, temx4, temx5, temy3, temy4, temy5, temz3), 1)
        # tem = torch.cat((temx5,temy5, temz3), 1)
        tem = self.norm1(tem)
        res1 = self.conv6(tem)
        res1 = self.norm2(res1)
        res2 = self.conv7(res1)
        res2 = self.norm3(res2)
        res3 = self.conv8(res2)

        # return res3, res1, res2, res3
        return res3