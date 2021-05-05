#
# lapq.py
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
from nemo.quant.pact import pact_quantize_asymm_inference
from tqdm import tqdm
import numpy as np

# code from BRECQ
def lp_norm(prediction, target, p=2.0, reduction=None):
    if reduction is None:
        return (prediction-target).abs().pow(p).sum(1).mean()
    else:
        return (prediction-target).abs().pow(p).mean()

def _weight_clip_lapq_sweep(self, validation_fn, asymmetric=True, layer_bits={}, reduction_factor=0.01, reduction_max=0.8, verbose=False, early_exit=False, early_exit_tol=2.0, lp_norm_p_min=2.0, lp_norm_p_max=4.0, lp_norm_p_step=0.05):
    best_cost = 1000
    best_loss = 1000
    best_eps = None
    best_p = 0
    for lp_norm_p in tqdm(np.arange(lp_norm_p_min, lp_norm_p_max, lp_norm_p_step)):
        cost, eps = self.weight_clip_lapq(asymmetric=asymmetric, layer_bits=layer_bits, reduction_factor=reduction_factor, reduction_max=reduction_max, verbose=verbose, early_exit=early_exit, early_exit_tol=early_exit_tol, lp_norm_p=lp_norm_p, reset_alpha=True)
        loss = validation_fn()
        if loss < best_loss:
            best_cost = cost
            best_loss = loss
            best_eps = eps
            best_p = lp_norm_p
    self.reset_alpha_weights()
    self.weight_clip_lapq(asymmetric=asymmetric, layer_bits=layer_bits, reduction_factor=reduction_factor, reduction_max=reduction_max, verbose=verbose, early_exit=early_exit, early_exit_tol=early_exit_tol, lp_norm_p=best_p, reset_alpha=True)
    return best_loss, best_cost, best_eps, best_p

# Set weight clipping parameters according to LAPQ
def _weight_clip_lapq(self, asymmetric=True, layer_bits={}, reduction_factor=0.01, reduction_max=0.8, verbose=False, early_exit=True, early_exit_tol=1.1, lp_norm_p=2.4, reset_alpha=False):

    module_dict = {}
    use_default = False
    if not layer_bits:
        layer_bits = {}
        use_default = True
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            if use_default:
                module_dict[n] = m
                layer_bits[n] = m.W_precision.get_bits()
            elif n in layer_bits.keys():
                module_dict[n] = m

    for n in module_dict.keys():
        m = module_dict[n]

        if reset_alpha:
            m.reset_alpha_weights()

        Wbeta  = m.weight.abs().max()
        Weps   = (2 * Wbeta) / (2 ** layer_bits[n] - 2)
        Walpha = Wbeta + Weps

        Wbeta_final  = Wbeta 
        Weps_final   = Weps
        Walpha_final = Walpha

        best_cost = lp_norm(m.weight, pact_quantize_asymm_inference(m.weight, Weps, Walpha, Wbeta), p=lp_norm_p)

        # with tqdm(total=int(reduction_max / reduction_factor)) as t:
        for i in range(int(reduction_max / reduction_factor)):
            Wbeta_new = Wbeta * (1.0 - (i * reduction_factor)) 
            Weps_new   = (2 * Wbeta_new) / (2 ** layer_bits[n] - 2)
            Walpha_new = Wbeta_new + Weps_new
    
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            cost = lp_norm(m.weight, pact_quantize_asymm_inference(m.weight, Weps_new, Walpha_new, Wbeta_new), p=lp_norm_p)
#            cost = lp_norm(m.weight, pact_quantize_asymm_inference(m.weight, Weps_new, Walpha_new, Wbeta_new), p=2.4, reduction='all')
            if cost > early_exit_tol*best_cost and early_exit:
                print(cost.item(), best_cost.item(), Walpha_final.item())
                break
            elif cost < best_cost:
                best_cost = cost
                Weps_final = Weps_new
                Walpha_final = Walpha_new 
                Wbeta_final = Walpha_new - Weps_final
            if verbose:
                print(cost.item(), best_cost.item(), Walpha_final.item())
            # t.set_postfix({'cost': cost.item(), 'best': best_cost.item(), 'Walpha': Walpha_final.item() })
            # t.update(1)

        m.W_alpha.data[:] = Walpha_final
        m.W_beta.data[:]  = Wbeta_final

        if verbose:
            print("[weight clip LAPQ] %s: alpha=%.3e beta=%.3e" % (n, m.W_alpha.data.item(), m.W_beta.data.item()))

        return best_cost.item(), Weps_final.item()

