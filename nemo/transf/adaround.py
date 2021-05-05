#
# adaround.py
# Francesco Conti <f.conti@unibo.it>
#
# Copyright (C) 2018-2021 ETH Zurich and University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# code partially inspired from BRECQ release: https://github.com/yhhhli/BRECQ

import torch
from nemo.precision import Precision
from nemo.quant.pact import *
from nemo.graph import DeployGraph
from torch.nn.modules.utils import _single,_pair
from collections import OrderedDict
import types
import logging
import numpy as np
import copy
import math
import torchvision.models
import re
from nemo.transf.common import *
#from nemo.quant.pact import pact_quantize_asymm_inference
from tqdm import tqdm
import nemo

__global_ave_grads = {}
__global_max_grads = {}

def reset_grad_flow(net, __global_ave_grads, __global_max_grads):
    for n, p in net.named_parameters():
        __global_ave_grads[n] = []
        __global_max_grads[n] = []

def save_grad_flow(net):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    named_parameters = net.named_parameters()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if hasattr(p, 'grad'):
                if not p.grad is None:
                    ave_grads.append(p.grad.abs().mean().item())
                    max_grads.append(p.grad.abs().max().item())
        try:
            __global_ave_grads[n].extend(ave_grads)
            __global_max_grads[n].extend(max_grads)
        except KeyError:
            __global_ave_grads[n] = ave_grads
            __global_max_grads[n] = max_grads



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def l2_norm(prediction, target, reduction=None):
    if reduction is None:
        return torch.square(prediction-target).sum(1).mean()
    else:
        return torch.square(prediction-target).mean()

def l1_norm(prediction, target, reduction=None):
    if reduction is None:
        return (prediction-target).abs().sum(1).mean()
    else:
        return (prediction-target).abs().mean()

def adaround_h_hard(x, gamma=-0.1, zeta=1.1, threshold=0.5):
    hx = adaround_h(x, gamma=gamma, zeta=zeta)
    ret = torch.zeros_like(hx)
    ret[hx <  threshold] = 0
    ret[hx >= threshold] = 1
    return ret

def adaround_h(x, gamma=-0.1, zeta=1.1):
    return torch.clamp(torch.sigmoid(x) * (zeta-gamma) + gamma, 0, 1)

def adaround_h_inv(x, gamma=-0.1, zeta=1.1):
    return -torch.log((zeta - gamma) / (x - gamma) - 1)

def adaround_freg(x, beta):
    return torch.sum(1 - torch.pow(torch.abs(2 * adaround_h(x) - 1), beta))

def _hook_latent(module, buffer_, hooks):
    def hk(module, input_, output):
        buffer_['in'] = input_
        buffer_['out'] = output
    hooks['hook'] = module.register_forward_hook(hk)

def _unhook_latent(hooks):
    hooks['hook'].remove()

# Set weight clipping parameters according to LAPQ
def _adaround_train(self, calibration_loader, validation_loader, validate=True, nb_epochs=1, nb_batches=20000, warmup=5000, temp_start=20.0, temp_end=2.0, lr=0.01, batch_size=64, lambda_=0.01, reconstruction_loss='l2_loss', layer_group=[], layer_bits={}, precision_dict={}, use_default=False, adaquant=False, brecq=False, init_only=False, delta=1e-6):

    module_dict = {}
    if not layer_bits:
        layer_bits = {}
        use_default = True
    for n,m in self.named_modules():
        module_dict[n] = m
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            if use_default:
                layer_bits[n] = m.W_precision.bits
            m.adaround = False

    Woffset = OrderedDict([])

    hooks_in = {}
    hooks_out = {}
    buffer_in = {}
    buffer_out = {}

    for n in layer_bits.keys():

        print("[AdaRound] Layer %s" % n)
        m = module_dict[n]

        # PACT_Act, etc. are also in the module_dict, but they have no adaround param
        if not hasattr(m, 'adaround'):
            continue

        # deactivate quantization (we model this with 20bits) for all layers before this layer/block
        for nn in module_dict.keys():
            mm = module_dict[nn]
            if nn != n and hasattr(mm, 'adaround') and not mm.adaround:
                mm.W_precision = nemo.precision.Precision(bits=20)
        m.W_precision = nemo.precision.Precision(bits=layer_bits[n])

        # single-layer, will want to change it for BRECQ
        with torch.no_grad():
            # extremely small weights are very confusing for the quantization procedure
            m.weight.data[m.weight.data.abs() < delta] = 0
            Wbase = m.weight.data.clone().detach()
            Walpha = m.W_alpha.clone().detach()
            Wbeta = m.W_beta.clone().detach()
            Weps = (Walpha + Wbeta) / (2 ** layer_bits[n] - 1)
            Wquant = nemo.quant.pact.pact_quantize_asymm_inference(Wbase - 0.5*Weps, Weps, m.W_alpha, m.W_beta)
            Wrest = ((Wbase - Wquant) / Weps).clamp(0,1)
            adaround_param = adaround_h_inv(Wrest)
            m.adaround_param = torch.nn.Parameter(adaround_param)
            optimizer = torch.optim.Adam([m.adaround_param, ], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs, eta_min=0.)

        if init_only:
            continue

        _hook_latent(module_dict[layer_group[0]], buffer_in, hooks_in)
        _hook_latent(module_dict[layer_group[-1]], buffer_out, hooks_out)

        # training loop 
        if nb_batches is not None:
            length = nb_batches
        else:
            length = len(calibration_loader)

        rec_loss_fn = l2_norm if reconstruction_loss == 'l2_loss' else l1_norm

        self.freeze_bn()

        def __adaround_validation():
        
            meter1 = AverageMeter()
            meter2 = AverageMeter()
            meter3 = AverageMeter()

            with tqdm(total=len(validation_loader)) as t:
    
                adareg_loss = torch.zeros(1)
    
                for i,(inputs,target) in enumerate(validation_loader):
    
                    # collect latent calibration data on a batch
                    with torch.no_grad():
                        m.adaround = False
                        m.quantize_W = False
                        m.W_precision.bits = 20
                        m.reset_alpha_weights()
                        self.eval()
    
                        if torch.cuda.is_available():
                            target = target.to('cuda', non_blocking=True)
                            inputs = inputs.to('cuda', non_blocking=True)
        
                        # compute output
                        output = self(inputs)
                        x = buffer_in['in'][0].clone().detach()

                        # compute non-quantized output
                        xx = x
                        for l in layer_group:
                            mm = module_dict[l]
                            xx = mm(xx)
                        y_nq = xx
        
                        m.adaround = False
                        m.quantize_W = True
                        m.W_precision.bits = layer_bits[n]
                        m.W_alpha.data[0] = Walpha
                        m.W_beta.data[0] = Wbeta

                        # compute quantized output without adaround
                        xx = x
                        for l in layer_group:
                            mm = module_dict[l]
                            xx = mm(xx)
                        y_na = xx
        
                        m.adaround = True

                        # compute quantized output without adaround
                        xx = x
                        for l in layer_group:
                            mm = module_dict[l]
                            xx = mm(xx)
                        y_a = xx

                        # compute loss and backprop
                        rec_loss_ada_vs_noqnt = rec_loss_fn(y_a, y_nq)
                        rec_loss_ada_vs_noada = rec_loss_fn(y_a, y_na)
                        rec_loss_noada_vs_noqnt = rec_loss_fn(y_na, y_nq)
                   
                        meter1.update(rec_loss_ada_vs_noqnt)
                        meter2.update(rec_loss_ada_vs_noada)
                        meter3.update(rec_loss_noada_vs_noqnt)
                        t.set_postfix({'ada_vs_noqnt': meter1.avg.item(), 'ada_vs_noada': meter2.avg.item(), 'noada_vs_noqnt': meter3.avg.item() })
                        t.update(1)
      
        if validate:
            print("[AdaRound] Initial validation")
            __adaround_validation()

        reset_grad_flow(self, __global_ave_grads, __global_max_grads)
        for e in range(nb_epochs):

            loss_meter = AverageMeter()
            l2_meter = AverageMeter()
            reg_meter = AverageMeter()
        
            meter1 = AverageMeter()
            meter2 = AverageMeter()
            meter3 = AverageMeter()

            ### training loop
            print("[AdaRound] Training epoch %d" % e)
            with tqdm(total=length) as t:
    
                adareg_loss = torch.zeros(1)
    
                for i,(inputs,target) in enumerate(calibration_loader):
    
                    # collect latent calibration data on a batch
                    with torch.no_grad():
                        m.adaround = False
                        m.quantize_W = False
                        m.W_precision.bits = 20
                        m.reset_alpha_weights()
                        self.eval()
    
                        # measure data loading time
                        if i==length:
                            break
    
                        if torch.cuda.is_available():
                            target = target.to('cuda', non_blocking=True)
                            inputs = inputs.to('cuda', non_blocking=True)
        
                        # compute output
                        output = self(inputs)
                        x = buffer_in['in'][0].clone().detach()
#                        y_nq = buffer_out['out'].clone().detach()

                        # compute quantized output
                        xx = x
                        for l in layer_group:
                            mm = module_dict[l]
                            xx = mm(xx)
                        y_nq = xx
        
                    self.train()
                    m.adaround = True
                    m.quantize_W = True
                    m.W_precision.bits = layer_bits[n]
                    m.W_alpha.data[0] = Walpha
                    m.W_beta.data[0] = Wbeta
    
                    # replace weights with Wbase + Woffset
                    optimizer.zero_grad()
    
                    # compute quantized output
                    xx = x
                    for l in layer_group:
                        mm = module_dict[l]
                        xx = mm(xx)
                    y_q = xx

                    # compute loss and backprop
                    rec_loss = rec_loss_fn(y_q, y_nq)
        
#                    if (e==0 and i==0) or (e==nb_epochs-1 and i==0):
#                        import IPython; IPython.embed()

                    if torch.isnan(rec_loss).any():
                        import IPython; IPython.embed()
    
                    ii = e*nb_batches + i
                    beta = temp_end + (nb_epochs*nb_batches - ii - 1) * (temp_start - temp_end) / (nb_epochs*nb_batches - warmup)
                    if ii < warmup:
                        loss = rec_loss
                    else:
                        adareg_loss = lambda_ * adaround_freg(m.adaround_param, beta)
                        loss = rec_loss + adareg_loss
    
                    loss.backward()
                    save_grad_flow(self)
                   
                    loss_meter.update(loss)
                    l2_meter.update(rec_loss)
                    reg_meter.update(adareg_loss)
                    t.set_postfix({'loss': loss_meter.avg.item(), 'rec': l2_meter.avg.item(), 'round': reg_meter.avg.item(), 'beta': beta})
                    t.update(1)
    
                    # step optimizer per batch
                    optimizer.step()

                # step scheduler per epoch
                scheduler.step()

            ### validation loop
            if validate:
                print("[AdaRound] Validation epoch %d" % e)
                __adaround_validation()

        _unhook_latent(hooks_in)
        _unhook_latent(hooks_out)

        m.adaround = True
        m.quantize_W = True
        m.W_precision.bits = layer_bits[n]
        m.W_alpha.data[0] = Walpha
        m.W_beta.data[0] = Wbeta

#        import IPython; IPython.embed(); import sys; sys.exit(0)

# Set weight clipping parameters according to LAPQ
def _adaround_harden(self, unharden=False, threshold=0.5):

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            module_dict[n] = m

    for n in module_dict.keys():
        m = module_dict[n]

        if m.adaround:

            Weps = (m.W_alpha + m.W_beta) / (2 ** m.W_precision.bits - 1)
    
            # apply hardened adaround parameters
            with torch.no_grad():
                adaround_param_tmp = adaround_h_inv(adaround_h_hard(m.adaround_param.data, threshold=threshold))
#                adaround_param_tmp = m.adaround_param.data
                if not unharden:
                    m.adaround_hardened = True
                    m.weight.data = m.weight.data + (adaround_h(adaround_param_tmp)) * Weps
                else:
                    m.adaround_hardened = False
                    m.weight.data = m.weight.data - (adaround_h(adaround_param_tmp)) * Weps
    
