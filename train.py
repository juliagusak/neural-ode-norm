# *
# @file ANODE training driver based on arxiv:1902.10298
# This file is part of ANODE library.
#
# ANODE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANODE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ANODE.  If not, see <http://www.gnu.org/licenses/>.
# *
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
# from tensorboardX import SummaryWriter
import math
import sys
import os

from torch.nn.utils import spectral_norm, weight_norm

from copy import deepcopy

import anode.utils as utils

from functools import partial
from interpolated_torchdiffeq import odeint_chebyshev_func

parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, choices=['resnet34', 'resnet18', 'resnet10', 'resnet6', 'resnet4',\
                                                    'preresnet34', 'preresnet18', 'preresnet10', 'preresnet6', 'preresnet4',\
                                                    'sqnxt'], default='sqnxt')
parser.add_argument('--method', type=str, choices=['Euler',
                                                   'RK2',
                                                   'RK4',
                                                   'dopri5',
                                                   'dopri5_old',
                                                   'dopri5_old_cheb',
                                                   'euler_cheb',
                                                   'rk2_cheb',
                                                   'rk4_cheb',
                                                   'ODEFree'],
                    default='Euler')
parser.add_argument('--backward_solver', type=str, choices=['dopri5',
                                                            'euler_cheb',
                                                            'rk2_cheb',
                                                            'rk4_cheb'],
                    default=None)
parser.add_argument('--atol_scheduler', type=str, default='{150: 1.0, 250: 1.0}') ## default: 1e-1, 1e-1
parser.add_argument('--rtol_scheduler', type=str, default='{150: 1.0, 250: 1.0}')
parser.add_argument('--num_epochs', type=int, default=350)
parser.add_argument('--n_nodes', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--log_every', type=int, default=2)  ## log_every_iter
parser.add_argument('--save_every', type=int, default=10)  ## save_every_epoch
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument('--data_root', type=str, default='./data')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--atol', type=float, default=1e-5)
# parser.add_argument('--Nt', type=int, default=None) # set it equal to n_nodes in the code 
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--normalization_resblock', type=str, default='BN', \
                    choices=['BN', 'LN', 'IN', 'NormFree'])
parser.add_argument('--normalization_odeblock', type=str, default='LN', \
                    choices=['BN', 'LN', 'IN',  'NormFree'])
parser.add_argument('--normalization_bn1', type=str, default='BN', \
                    choices=['BN', 'LN', 'IN',  'NormFree'])
parser.add_argument('--param_normalization_resblock', type=str, default='ParamNormFree', \
                    choices=['WN', 'SpecN', 'ParamNormFree'])
parser.add_argument('--param_normalization_odeblock', type=str, default='ParamNormFree', \
                    choices=['WN', 'SpecN', 'ParamNormFree'])
parser.add_argument('--param_normalization_bn1', type=str, default='ParamNormFree', \
                    choices=['WN', 'SpecN', 'ParamNormFree'])
parser.add_argument('--activation_resblock', type=str, default='ReLU', \
                    choices=['ReLU', 'ActFree'])
parser.add_argument('--activation_odeblock', type=str, default='ReLU', \
                    choices=['ReLU', 'ActFree'])
parser.add_argument('--activation_bn1', type=str, default='ReLU', \
                    choices=['ReLU', 'ActFree'])
parser.add_argument('--use_backward_cheb_points', type=eval, default=False, choices=[True, False])

parser.add_argument('--inplanes', type=int, default=64)

parser.add_argument('--return_inter_points', type=eval, default=False)

parser.add_argument('--torch_seed', type=int, default=502)


args = parser.parse_args()
args.Nt = args.n_nodes

# Set random seed for reproducibility
torch.manual_seed(args.torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.network == 'sqnxt':
    from models.sqnxt import SqNxt_23_1x, lr_schedule

#     events_dir = "{}/events/".format(args.save)
#     writer = SummaryWriter('{}/{}/'.format(events_dir, args.network) + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')
elif args.network.startswith('resnet'):
    from models.resnet import ResNet18, ResNet10, ResNet4, ResNet6, ResNet34, lr_schedule
    
elif args.network.startswith('preresnet'):
    from models.preresnet import PreResNet18, PreResNet10, PreResNet4, PreResNet6, PreResNet34, lr_schedule

#     writer = SummaryWriter('{}/{}/'.format(events_dir, args.network) + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')

args.rtol_scheduler = eval(args.rtol_scheduler)
args.atol_scheduler = eval(args.atol_scheduler)


def tol_scheduler(tol, epoch, politics):
    optim_factor = 0.0
    for epoch_number, factor in politics.items():
        if epoch_number < epoch:
            optim_factor = factor
    return tol/math.pow(10, (optim_factor))

if args.method in ['dopri5', 'euler_cheb', 'rk2_cheb', 'rk4_cheb']:
    method_alias = {
        'dopri5': 'dopri5',
        'euler_cheb': 'euler',
        'rk2_cheb': 'midpoint',
        'rk4_cheb': 'rk4'
    }
    if args.backward_solver is not None:
        args.method = (method_alias[args.method], method_alias[args.backward_solver])
    else:
        args.method = method_alias[args.method]

    odesolver = partial(odeint_chebyshev_func, t=torch.tensor([0.0, 1.0]),
                        n_nodes=args.n_nodes, method=args.method)
elif args.method == 'dopri5_old_cheb':
    odesolver = partial(odeint_chebyshev_func, t=torch.tensor([0.0, 1.0]),
                        n_nodes=args.n_nodes, method='dopri5_old',
                        rtol=args.rtol, atol=args.atol)
else:
    from anode import odesolver_adjoint as odesolver

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = 1
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
best_acc = 0.
best_epoch = start_epoch


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        if args.use_backward_cheb_points:
            self.options.update({'method': args.method})
            self.options.update({'use_backward_cheb_points': args.use_backward_cheb_points})
        if args.method != 'dopri5':
            self.options.update({'Nt': int(args.Nt)})
            self.options.update({'method': args.method})
        if args.method == 'dopri5_old':
            self.options.update({'atol': args.atol})
            self.options.update({'rtol': args.rtol})

    def forward(self, x):
        if args.method in ['dopri5', 'euler', 'midpoint', 'rk4', 'dopri5_old_cheb'] or \
                isinstance(args.method, tuple):
            out = odesolver(self.odefunc, x,
                            options=self.options,
                            rtol=tol_scheduler(args.rtol, _epoch, args.rtol_scheduler),
                            atol=tol_scheduler(args.atol, _epoch, args.atol_scheduler),
                            return_intermediate_points=args.return_inter_points,)
        else:
            out = odesolver(self.odefunc,
                            x,
                            options=self.options,)
        if args.method in ['dopri5', 'euler', 'midpoint', 'rk4', 'dopri5_old_cheb'] or \
                isinstance(args.method, tuple):
            out = out[1, ...]
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

    @property
    def nbe(self):
        return self.odefunc.nbe

    @nbe.setter
    def nbe(self, value):
        self.odefunc.nbe = value

    @property
    def forward_t(self):
        return self.odefunc.forward_t

    @forward_t.setter
    def forward_t(self, value):
        self.odefunc.forward_t = deepcopy(value)

    @property
    def backward_t(self):
        return self.odefunc.backward_t

    @backward_t.setter
    def backward_t(self, value):
        self.odefunc.backward_t = deepcopy(value)

    @property
    def dt(self):
        return self.odefunc.dt

    @dt.setter
    def dt(self, value):
        self.odefunc.dt = deepcopy(value)

    @property
    def f_t(self):
        if args.return_inter_points:
            return self.odefunc.f_t.clone()

    @property
    def z_t(self):
        if args.return_inter_points:
            return self.odefunc.z_t.clone()


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root=args.data_root, transform=transform_train, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=args.data_root, transform=transform_test, train=False, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_normalization(key):
    if key == 'BN':
        return nn.BatchNorm2d
    elif key == 'LN':
        return partial(nn.GroupNorm, 1)
    elif key == 'IN':
        return nn.InstanceNorm2d
    elif key == 'NormFree':
        return Identity
    else:
        raise NameError('Unknown layer normalization type')

def get_param_normalization(key):
    if key == 'SpecN':
        return spectral_norm
    elif key == 'WN':
        return weight_norm
    elif key == 'ParamNormFree':
        return lambda x: x 
    else:
        raise NameError('Unknown param normalization type')

def get_activation(key):
    if key == 'ReLU':
        return F.relu
    elif key == 'ActFree':
        return partial(F.leaky_relu, negative_slope=1)
    else:
        raise NameError('Unknown activation type')


### Initializa normalization layers
norm_layers = (get_normalization(args.normalization_resblock), \
               get_normalization(args.normalization_odeblock), \
               get_normalization(args.normalization_bn1))

param_norm_layers = (get_param_normalization(args.param_normalization_resblock),\
               get_param_normalization(args.param_normalization_odeblock),\
               get_param_normalization(args.param_normalization_bn1))

act_layers = (get_activation(args.activation_resblock),\
              get_activation(args.activation_odeblock),\
              get_activation(args.activation_bn1))

if args.method == 'ODEFree':
    ODEBlock = None

if args.network == 'sqnxt':
    net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet18':
    net = ResNet18(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'resnet10':
    net = ResNet10(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'resnet4':
    net = ResNet4(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'resnet6':
    net = ResNet6(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'resnet34':
    net = ResNet34(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'preresnet18':
    net = PreResNet18(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'preresnet10':
    net = PreResNet10(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'preresnet4':
    net = PreResNet4(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'preresnet6':
    net = PreResNet6(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)
elif args.network == 'preresnet34':
    net = PreResNet34(ODEBlock, norm_layers, param_norm_layers, act_layers, args.inplanes)

net.apply(conv_init)
logger.info(args)
logger.info(net)
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
os.makedirs(args.save, exist_ok=True)


def train(epoch):
    # import pydevd
    #
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=5e-4)

    log_message = 'TrainingEpoch: {:d} | LR: {:.4f}'.format(epoch, lr_schedule(lr, epoch))
    logger.info(log_message)
    net.module.dt = list()
    net.module.dt = list()
    net.module.forward_t = list()
    net.module.backward_t = list()
    net.module.nbe = 0
    net.module.nfe = 0
    batches = []
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            batches.append((inputs.to(device), labels.to(device)))
        else:
            batches.append((inputs, labels))
    for idx, (inputs, labels) in enumerate(batches):
        # if is_use_cuda:
        #     inputs, labels = inputs.to(device), labels.to(device)
        epoch_time = time.time()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        f_t = net.module.f_t
        z_t = net.module.z_t
        forward_dt = net.module.dt
        net.module.dt = list()
        loss.backward()
        optimizer.step()

        backward_dt = net.module.dt
        net.module.dt = list()

        #         writer.add_scalar('Train/Loss', loss.item(), epoch * 50000 + batch_size * (idx + 1))
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        log_message = 'TrainingEpoch [{:d}/{:d}] | Iter[{:d}/{:d}] | Loss: {:.8f} | Acc@1: {:.4f} | FdT {:} | ' \
                      'BdT {:} | TotalTime {:.4f} | tF {:} | tB {:} | NFE {:} | NBE {:} | PeakMemory: {:d} | ' \
                      'z_t {:} | f_t {:}'.format(
            epoch, num_epochs, idx, len(train_dataset) // batch_size,
                                    train_loss / (batch_size * (idx + 1)), correct / total,
            # str(forward_dt), str(backward_dt),
            '[]', '[]',
                                    time.time() - epoch_time,
            str(net.module.forward_t), str(net.module.backward_t),
            str(net.module.nbe),
            str(net.module.nfe),
            # TODO: first, nfe and nbe have to be swaped here. Second, we have to take nfe before backward call.
            torch.cuda.max_memory_allocated(device),
            str(z_t),
            str(f_t),
        )

        if idx % args.log_every == 0:
            logger.info(log_message)
        #             torch.save({
        #                 "args": args,
        #                 "state_dict": net.state_dict() if torch.cuda.is_available() else net.state_dict(),
        #                 "optim_state_dict": optimizer.state_dict(),
        #             }, os.path.join(args.save, "checkpt_{:d}_{:d}.pth".format(idx, epoch)))

        net.module.forward_t = list()
        net.module.backward_t = list()
        net.module.nbe = 0
        net.module.nfe = 0

    log_message = 'Train/Accuracy | Acc@1: {:.3f} | Epoch {:d}'.format(correct / total, epoch)
    logger.info(log_message)

    if (epoch - 1) % args.save_every == 0:
        torch.save({
            "args": args,
            "state_dict": net.state_dict() if torch.cuda.is_available() else net.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }, os.path.join(args.save, "checkpt_{:d}.pth".format(epoch)))

    torch.save({
        "args": args,
        "state_dict": net.state_dict() if torch.cuda.is_available() else net.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }, os.path.join(args.save, "checkpt.pth"))


#     writer.add_scalar('Train/Accuracy', correct / total, epoch)


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    net.module.dt = list()
    net.module.dt = list()
    net.module.forward_t = list()
    net.module.backward_t = list()
    net.module.nbe = 0
    net.module.nfe = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        epoch_time = time.time()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        #         writer.add_scalar('Test/Loss', loss.item(), epoch * 50000 + test_loader.batch_size * (idx + 1))

        log_message = 'TestingEpoch [{:d}/{:d}] | Iter[{:d}/{:d}] | Loss: {:.8f} | Acc@1: {:.4f} | ' \
                      'TotalTime {:.4f} | tF {:} | PeakMemory: {:d}'.format(epoch, num_epochs, idx,
                                                                            len(test_dataset) // test_loader.batch_size,
                                                                            test_loss / (100 * (idx + 1)),
                                                                            correct / total,
                                                                            time.time() - epoch_time,
                                                                            net.module.forward_t,
                                                                            torch.cuda.max_memory_allocated(device))
        logger.info(log_message)

        net.module.dt = list()
        net.module.dt = list()
        net.module.forward_t = list()
        net.module.backward_t = list()
        net.module.nbe = 0
        net.module.nfe = 0
        # sys.stdout.flush()

    epoch_acc = correct / total
    log_message = 'Test/Accuracy | Acc@1: {:.4f} | Epoch {:d}'.format(epoch_acc, epoch)
    logger.info(log_message)
    
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch



#     writer.add_scalar('Test/Accuracy', correct / total, epoch)

for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    test(_epoch)
    end_time = time.time()

    log_message = 'Epoch {:d} | Cost(sec) {}'.format(_epoch, end_time - start_time)
    logger.info(log_message)

log_message = 'BestAcc@1: {:.4f}, BestEpoch: {}'.format(best_acc * 100, best_epoch)
logger.info(log_message)
# writer.close()

