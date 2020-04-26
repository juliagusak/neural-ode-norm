import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import numpy as np
from copy import deepcopy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,\
                 norm_layer=None, act_layer=None, param_norm=lambda x: x
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = norm_layer(planes)
    
        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = norm_layer(planes)
    
        self.act = act_layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                param_norm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim, norm_layer=None, act_layer=None,
                param_norm=lambda x: x):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0
        self.nbe = 0
        self.forward_t = list()
        self.backward_t = list()
        self.dt = list()
        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        # Replace BN to GN because BN doesn't work with our method normaly
        self.bn1 = norm_layer(planes)

        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = norm_layer(planes)
    
        self.act = act_layer

        self.shortcut = nn.Sequential()

    def forward(self, t, x):
        self.nfe += 1
        if isinstance(x, tuple):
            x = x[0]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ODEBlock_=None,\
                 norm_layers_=(None, None, None),
                 param_norm_layers_=(lambda x: x, lambda x: x, lambda x: x),
                 act_layers_=(None, None, None),
                 in_planes_=64):
        '''
        norm_layers_: tuple of normalization layers for (BasicBlock, BasicBlock2, bn1)
        param_norm_layers_: tuple of normalizations for weights in (BasicBlock, BasicBlock2, conv1)
        act_layers_: tuple of activation layers for (BasicBlock, BasicBlock2, activation after bn1)
        
        '''
        
        super(ResNet, self).__init__()
        self.in_planes = in_planes_
        self.ODEBlock = ODEBlock_
        
        self.ODEBlocks = []

        self.n_layers = len(num_blocks)
        self.n_features_linear = in_planes_
        
        self.conv1 = param_norm_layers_[2](nn.Conv2d(3, in_planes_, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layers_[2](in_planes_)
        self.act = act_layers_[2]

        
        self.layer1_1, self.layer1_2 = self._make_layer(in_planes_, num_blocks[0], stride=1,
                                                        norm_layers_=norm_layers_[:2],
                                                        param_norm_layers_=param_norm_layers_[:2],
                                                        act_layers_=act_layers_[:2])
        
        if self.n_layers >= 2:
            self.n_features_linear *= 2
            self.layer2_1, self.layer2_2 = self._make_layer(in_planes_*2, num_blocks[1], stride=2,
                                                            norm_layers_ = norm_layers_[:2],
                                                            param_norm_layers_=param_norm_layers_[:2],
                                                            act_layers_ = act_layers_[:2])
            
        if self.n_layers >= 3:
            self.n_features_linear *= 2
            self.layer3_1, self.layer3_2 = self._make_layer(in_planes_*4, num_blocks[2], stride=2,
                                                            norm_layers_=norm_layers_[:2],
                                                            param_norm_layers_=param_norm_layers_[:2],
                                                            act_layers_=act_layers_[:2])

        if self.n_layers >= 4:
            self.n_features_linear *= 2
            self.layer4_1, self.layer4_2 = self._make_layer(in_planes_*8, num_blocks[3], stride=2,
                                                            norm_layers_=norm_layers_[:2],
                                                            param_norm_layers_=param_norm_layers_[:2],
                                                            act_layers_=act_layers_[:2])
            
        self.linear = nn.Linear(self.n_features_linear * block.expansion, num_classes)

        
    def _make_layer(self, planes, num_blocks, stride, norm_layers_, param_norm_layers_, act_layers_):
        '''
        num_blocks: tuple (num_ResBlocks, num_ODEBlocks)
        stride: stride of first conv layer in ResNet layer
        '''
        num_resblocks, num_odeblocks = num_blocks
        
        strides = [stride] + [1] * (num_resblocks + num_odeblocks - 1)
        layers_res = []
        layers_ode = []
        
        for stride in strides[:num_resblocks]:
            layers_res.append(BasicBlock(self.in_planes, planes, stride,
                                         norm_layer = norm_layers_[0],
                                         param_norm=param_norm_layers_[0],
                                         act_layer = act_layers_[0]))
            self.in_planes = planes * BasicBlock.expansion
            
        for stride in strides[num_resblocks:]:
            layers_ode.append(self.ODEBlock(BasicBlock2(self.in_planes,
                                                        norm_layer=norm_layers_[1],
                                                        param_norm=param_norm_layers_[1],
                                                        act_layer=act_layers_[1])))
            
        self.ODEBlocks += layers_ode
            
        return nn.Sequential(*layers_res),  nn.Sequential(*layers_ode)

    # self.forward_t = list()
    # self.backward_t = list()
    # self.dt = list()
    @property
    def nfe(self):
        return {idx: layer.nfe for idx, layer in enumerate(self.ODEBlocks)}

    @nfe.setter
    def nfe(self, value):
        for layer in self.ODEBlocks:
            layer.nfe = value

    @property
    def nbe(self):
        return {idx: layer.nbe for idx, layer in enumerate(self.ODEBlocks)}

    @nbe.setter
    def nbe(self, value):
        for layer in self.ODEBlocks:
            layer.nbe = value

    @property
    def forward_t(self):
        return {idx: np.mean(layer.forward_t) for idx, layer in enumerate(self.ODEBlocks)}

    @forward_t.setter
    def forward_t(self, value):
        for layer in self.ODEBlocks:
            layer.forward_t = deepcopy(value)

    @property
    def backward_t(self):
        return {idx: np.mean(layer.backward_t) for idx, layer in enumerate(self.ODEBlocks)}

    @backward_t.setter
    def backward_t(self, value):
        for layer in self.ODEBlocks:
            layer.backward_t = deepcopy(value)

    @property
    def dt(self):
        return {idx: layer.dt for idx, layer in enumerate(self.ODEBlocks)}

    @dt.setter
    def dt(self, value):
        for layer in self.ODEBlocks:
            layer.dt = deepcopy(value)

    @property
    def f_t(self):
        return {idx: layer.f_t for idx, layer in enumerate(self.ODEBlocks)}

    @property
    def z_t(self):
        return {idx: layer.z_t for idx, layer in enumerate(self.ODEBlocks)}

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1_1(out)
        out = self.layer1_2(out)
        if self.n_layers >= 2:
            out = self.layer2_1(out)
            out = self.layer2_2(out)
        if self.n_layers >= 3:
            out = self.layer3_1(out)
            out = self.layer3_2(out)
        if self.n_layers >= 3:
            out = self.layer4_1(out)
            out = self.layer4_2(out)
            
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet4(ODEBlock, norm_layers, param_norm_layers, act_layers, in_planes):
    if ODEBlock:
        num_blocks = [(0, 1)]
    else:
        num_blocks = [(1, 0)]
    return ResNet(BasicBlock, num_blocks, ODEBlock_=ODEBlock,
                  norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                  in_planes_=in_planes)

    
def ResNet6(ODEBlock, norm_layers, param_norm_layers, act_layers, in_planes):
    if ODEBlock:
        num_blocks = [(1, 1)]
    else:
        num_blocks = [(2, 0)]
    return ResNet(BasicBlock, num_blocks, ODEBlock_=ODEBlock,
                  norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                  in_planes_=in_planes)

    
def ResNet10(ODEBlock, norm_layers, param_norm_layers, act_layers, in_planes):
    if ODEBlock:
        num_blocks = [(1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0)]
    return ResNet(BasicBlock, num_blocks, ODEBlock_=ODEBlock,
                  norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                  in_planes_=in_planes)

    
def ResNet18(ODEBlock, norm_layers, param_norm_layers, act_layers, in_planes):
    if ODEBlock:
        num_blocks =  [(1, 1), (1, 1), (1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0), (2, 0), (2, 0)]
    return ResNet(BasicBlock, num_blocks, ODEBlock_=ODEBlock,
                  norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                  in_planes_=in_planes)

    
def ResNet34(ODEBlock, norm_layers, param_norm_layers, act_layers, in_planes):
    if ODEBlock:
        num_blocks = [(1, 2), (1, 3), (1, 5), (1, 2)]
    else:
        num_blocks = [(3, 0), (4, 0), (6, 0), (3, 0)]
    return ResNet(BasicBlock, num_blocks, ODEBlock_=ODEBlock,
                  norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                  in_planes_=in_planes)
     

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

