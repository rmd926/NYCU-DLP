import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):#ResNet最一開始的block
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        ##接下來做shortcut的部分
        if stride != 1 or in_channels != out_channels: 
            ##shortcut當stride不為1或是in_channels不等於out_channels時，要使用1x1的convolution來調整
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()#不做任何調整

    def forward(self, x):#順序是conv1->bn1->relu->conv2->bn2->shortcut->relu
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))#對x做convolution->batch normalization->relu
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):#U-Net的convolution block
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)

class ResNet34_Unet(nn.Module):
    def __init__(self):
        super(ResNet34_Unet, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder1 = self._make_layer(64, 64, 3, stride=1) 
        self.encoder2 = self._make_layer(64, 128, 4, stride=2)
        self.encoder3 = self._make_layer(128, 256, 6, stride=2)
        self.encoder4 = self._make_layer(256, 512, 3, stride=2)
        #bottleneck部分從512->256
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upconv_blocks = nn.ModuleList([
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),#512+256
            nn.ConvTranspose2d(288, 288, kernel_size=2, stride=2),#256+32#原架構圖有誤 不應該是512+32
            nn.ConvTranspose2d(160, 160, kernel_size=2, stride=2),#128+32
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)#64+32
        ])
        self.conv_blocks = nn.ModuleList([
            ConvBlock(768, 32),#藍加灰->藍
            ConvBlock(288, 32),
            ConvBlock(160, 32),
            ConvBlock(96, 32),
        ])
        #最後一個上採樣 32->32
        self.final_upconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    #input channel, output channel, block數量, stride
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):#3,4,6,3
            layers.append(BasicBlock(out_channels, out_channels))#把block加進去
        return nn.Sequential(*layers)#將list中的元素展開成參數
    #相當於layers = [BasicBlock(in_channels, out_channels, stride)]+[BasicBlock(out_channels, out_channels)]*blocks

    def forward(self, x):
        x = self.initial_layer(x)
        #encoder部分
        encoder1_out = self.encoder1(x)
        encoder2_out = self.encoder2(encoder1_out)
        encoder3_out = self.encoder3(encoder2_out)
        encoder4_out = self.encoder4(encoder3_out)
        #bottleneck部分
        x = self.bottleneck(encoder4_out)
        #decoder部分包含上採樣、連接
        x = self.upconv_blocks[0](torch.cat([x, encoder4_out], dim=1))
        x = self.conv_blocks[0](x)
        x = self.upconv_blocks[1](torch.cat([x, encoder3_out], dim=1))
        x = self.conv_blocks[1](x)
        x = self.upconv_blocks[2](torch.cat([x, encoder2_out], dim=1))
        x = self.conv_blocks[2](x)
        x = self.upconv_blocks[3](torch.cat([x, encoder1_out], dim=1))
        x = self.conv_blocks[3](x)
        x = self.final_upconv(x)
        x = self.final_conv(x)

        return F.sigmoid(x)

