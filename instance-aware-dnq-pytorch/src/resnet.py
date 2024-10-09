import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.quant import DymQuanConv as MyConv
from src.quant import QuanConv
from src.gumbelsoftmax import GumbleSoftmax
import torch.nn.functional as F
def _weight_variable(shape, factor=0.01):
    return torch.Tensor(torch.randn(*shape) * factor)

class BasicCell(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(BasicCell, self).__init__()

        self.conv1 = MyConv(in_planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MyConv(planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, one_hot):
        residual = x

        out = self.conv1(x, one_hot)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, one_hot)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x, one_hot)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)
        return out

class QuanBasicCell(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(QuanBasicCell, self).__init__()

        self.conv1 = QuanConv(in_planes, planes, 3, name_w, name_a, nbit_w, nbit_a,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuanConv(planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, name_w, name_a, nbit_w, nbit_a, num_bits=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(QuanBasicCell, 64, num_blocks[0], name_w, name_a, nbit_w, nbit_a, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], name_w, name_a, nbit_w, nbit_a, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool_policy = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc1 = nn.Linear(64*8*8, 64)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, num_bits)
        self.gumbelsoftmax = GumbleSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, name_w, name_a, nbit_w, nbit_a, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                MyConv(self.in_planes, planes * block.expansion, 1, name_w,
                       name_a, nbit_w, nbit_a, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, name_w, name_a, nbit_w, nbit_a))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for m in self.layer1:
            x = m(x)

        middle = x

        feat = self.avgpool_policy(x)
        feat = self.fc1(feat.view(x.shape[0], -1))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        one_hot = self.gumbelsoftmax(feat)

        for m in self.layer2:
            x = m(x, one_hot)
        for m in self.layer3:
            x = m(x, one_hot)
        for m in self.layer4:
            x = m(x, one_hot)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = self.fc(x)
        return x, one_hot

def resnet18(name_w='dorefa', name_a='dorefa', nbit_w=4, nbit_a=4, num_bits=3, num_classes=1000):
    return ResNet(BasicCell, [2, 2, 2, 2], name_w, name_a, nbit_w, nbit_a, num_bits, num_classes)
