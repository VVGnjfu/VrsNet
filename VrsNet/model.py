import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn import init


class CA_Block(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(CA_Block,self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel,out_channels=channel//reduction,kernel_size=1,stride = 1,bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels= channel // reduction,out_channels=channel,kernel_size=1,stride=1,bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Pytorch: init
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def forward(self,x):
        _,_,h,w = x.size()
        #-- (b,c,h,w)->(b,c,h,1)->(b,c,1,h)
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2)
        #(b,c,h,w)->(b,c,1,w)
        x_w = torch.mean(x,dim=2,keepdim=True)

        #-- x_w+x_h --> (b,c/r,1,w+h)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h,x_w),3))))

        #-- (b,c/r,1,w+h) --> (b,c/r,1,h) + (b,c/r,1,w)
        x_cat_conv_split_h,x_cat_conv_split_w = x_cat_conv_relu.split([h,w],3)

        #-- (b,c/r,1,h) --> (b,c,h,1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0,1,3,2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class Resnet101FPNWithAttention(nn.Module):
    def __init__(self):
        super(Resnet101FPNWithAttention, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

        # MLCA
        self.ca_map3 = CA_Block(512)
        self.ca_map4 = CA_Block(1024)

        # 将ResNet参数冻结
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)

        feat_map3 = self.ca_map3(feat_map3)
        feat_map4 = self.ca_map4(feat_map4)

        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class CountRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0), keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            for i in range(0, num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0), keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0, keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)