import torch
from torch import nn
import math



class SCSE(nn.Module):
    def __init__(self, ch,tho, re=16):
        super(SCSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.Linear(ch, ch // re, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // re, ch, bias=False),
            nn.Sigmoid()
        )
        self.tho = tho
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(
            nn.Conv2d(ch, 1, 1, bias=False),  # Spatial attention
            nn.Sigmoid()  # Spatial-wise attention
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Apply channel-wise attention
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.cSE(y).view(b, c, 1, 1)
        cse_out = x * y.expand_as(x)
        # Apply spatial-wise attention
        sse_out =self.sSE(x)
        sse_out = x * sse_out
        # Combine the two attention maps
        return cse_out + sse_out * self.tho


class se_block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            #                m.bias.data.zero_()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]          意为[batch_size, channels, height, width]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs