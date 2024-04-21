# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from matplotlib.cbook import flatten

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F

import models.configs as configs

from .modeling_resnet import ResNetV2


# My:
import random
#


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
logger = logging.getLogger(__name__)

ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.softmax2 = Softmax(dim=-2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
    
        # print("trans", (key_layer.transpose(-1, -2)).shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print('sum or:',attention_scores.sum(-1,keepdim=True)[0,0,:])
        # print('sum shape or:',attention_scores.sum(-1,keepdim=True).shape)



        mask_guide = None
        if mask_guide is not None:

            #mask = None
            # if mask is not None:

            #     print_info = False

            #mask_out = torch.mul(mask, (-5000)) # for 4, 5, 6 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #mask = mask * (-1e4) # -1e9 too small for gradiend for some reason
            #mask = torch.mul(mask, (-1e4)) # not for 8 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #print("mask before", mask)


            ### 1st option:
            # old, with a mistake of -20000:
            # attention_scores[:, :, :, 1:] = torch.add( attention_scores[:, :, :, 1:] , mask[:, None, None, :] ) # [28, 12, 626, 1:626] x [28, None, None, 625] # patches_w + cls_w
            # attention_scores[:, :, 1:, :] = torch.add( attention_scores[:, :, 1:, :] , mask[:, None, :, None] ) # [28, 12, 1:626, 626] x [28, None, 625, None] # patches_h + cls_h


            #- new+:  
            # mask_diag = torch.zeros(mask.size(0), (mask.size(1) + 1), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            # mask_diag = mask_diag.to(device='cuda')

            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
            
            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -15000.0, mask_diag_temp, torch.tensor(-10000.).cuda()) # < -10000 but -15000 just in case (expects -20000)

            # attention_scores[:, :, :, :] = torch.add( attention_scores[:, :, :, :] , mask_diag[:, None, :, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]

            # print( attention_scores[0, 0, 0, 2] )
            # print( attention_scores[0, 0, 0, 16] )
            # print("attn after", attention_scores.shape)



            ### 2nd option:
            # attention_scores[:, :, :, 1:] = torch.add( attention_scores[:, :, :, 1:] , mask[:, None, None, :] ) # [28, 12, 626, 1:626] x [28, None, None, 625] # patches_w + cls_w



            ### 3d option:
            # attention_scores[:, :, 1:, :] = torch.add( attention_scores[:, :, 1:, :] , mask[:, None, :, None] ) # [28, 12, 1:626, 626] x [28, None, 625, None] # patches_h + cls_h


            #- 3.1 new with cls_w:
            # attention_scores[:, :, 1:, :] = torch.add( attention_scores[:, :, 1:, :] , mask[:, None, :, None] ) # [28, 12, 1:626, 626] x [28, None, 625, None] # patches_h + cls_h
            # #print("attn mid", attention_scores)

            # attention_scores[:, :, 0, 1:] = torch.add( attention_scores[:, :, 0, 1:] , mask[:, None, :] ) # [28, 12, 0, 1:626] x [28, None, None, 625] # cls_w
            # #print("attn after", attention_scores)



            # ### for 4, 5, 6, 7 options:
            # #mask_diag = torch.zeros(28, 626, 626)
            # mask_diag = torch.zeros(mask.size(0), (mask.size(1) + 1), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            # mask_diag = mask_diag.to(device='cuda')
            


            ## 4th option:
            #- old, with a mistake of outliers:
            # #mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # #mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h
            

            #- new (mistake):
            # mask_out = torch.mul(mask, (-5000))

            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h

            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask_out[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h

            # print("mask mid", mask_diag)

            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # # 0 if > -7000 just in case (expects -5000)

            # print("mask after", mask_diag)


            #- new++:
            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask_out[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h

            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # # 0 if > -7000 just in case (expects -5000)

            # mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h
            



            ## 5th option:
            #- old, with a mistake of outliers:
            # #mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
            # #mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w


            #- new_good (a mistake technically):
            # mask_out = torch.mul(mask, (-5000)) # mistake, need to multiply before changing mask !!!

            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
            # mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask_out[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h

            # #print("mask mid", mask_diag)

            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if > -7000 just in case (expects -5000)
                        
            # #print("mask after", mask_diag)


            #- new++:
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h (-10k)
            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask_out[:, None, :] ) # [28, 626, 1:626] x [28, None, 625]  # patches_w + cls_w (-5k)

            #print("mask mid", mask_diag)

            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if > -7000 just in case (expects -5000)

            # mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w  (-10k)
            
            #print("mask after", mask_diag)



            ## 6th option:
            #- old, with a mistake of outliers:

            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
            
            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     #mask_diag[i, :, :] = torch.where( (mask_diag[i, :, :] < (-15000)), mask_diag[i, :, :], torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -15000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)


            # new+:
            # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
            # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
            
            # print("mask mid", mask_diag)

            # for i in range(mask_diag.size(0)):
            #     mask_diag_temp = mask_diag[i, :, :]

            #     #mask_diag[i, :, :] = torch.where( (mask_diag[i, :, :] < (-15000)), mask_diag[i, :, :], torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)
            #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -15000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)
            
            # print("mask after", mask_diag)



            ## 7th option:
            # mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h
            # mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w


            #- 7.1 with CLS_W only
            #mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            #print("before", mask_diag)


            #- 7.2 with CLS_H only
            #mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h
            #print("before", mask_diag)

            

            ## for 4, 5, 6, 7 options:
            # #mask_diag = mask_diag.fill_diagonal_(0)
            # for i in range(mask_diag.size(0)):
            #     mask_diag[i, :, :] = mask_diag[i, :, :].fill_diagonal_(0) # allow diag attention
            #     #mask_diag[i, :, :] = mask_diag[i, :, :].fill_diagonal_(-10000) # don't allow diag attention



            ### End for 1-7 options:

            #attention_scores[:, :, :, :] = torch.add( attention_scores[:, :, :, :] , mask_diag[:, None, :, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
            #print("attn after", attention_scores)





            ## 8th option (less strict mask):
            '''
            #- 8.1 mask - 0.95 (x0.54), obj - 1.05 (x1.79)

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            mask_mx = 0.7244 # (x0.02) , 0.685 # (x0.01) , 0.83 (x0.1) , 0.95 (x0.54) # final mx = mask_mx ^ 12
            obj_mx = 1.386 # (x50.2) , 1.469 # (x100.9) , 1.215 (x10.3) , 1.05 (x1.79) # final mx = obj_mx ^ 12 

            diff_mx = mask_mx - obj_mx # negative value (-0.1): mask_mx < obj_mx

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (diff_mx)) # make mask values = diff_mx (1* diff_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = mask_mx (diff_mx + obj_mx), make obk values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , (obj_mx * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)

            # end for 8th option
            attention_scores[:, :, 0, :] = torch.mul( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
            #print("attn scores", mask_diag)
            '''
            

            '''
            #- 8.3 obj + x

            #print("mask before", mask)

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            obj_mx = 0.3 # 0.3 (+3.6), 10.0 (+120), 5.0 (+60) # final mx = obj_mx * 12

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (-obj_mx)) # make mask values = - obj_mx (1* -obj_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = 0, make obj values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , obj_mx ) # * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)


            # print("attn scores max_min before:")
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))

            #print("attn scores", attention_scores)


            # end for 8.3 option
            attention_scores[:, :, 0, :] = torch.add( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
           
            # print("attn scores max_min after:")
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))
           
            #print("attn scores", attention_scores)
            '''



            '''
            #- 8.4 mask - 0.95 (x0.54), obj - 1.05 (x1.79), obj_2 - ~0.5*obj

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            mask_mx = 0.7244 # (x0.02) , 0.685 # (x0.01) , 0.83 (x0.1) , 0.95 (x0.54) # final mx = mask_mx ^ 12
            obj_mx = 1.386 # (x50.2) , 1.469 # (x100.9) , 1.215 (x10.3) , 1.05 (x1.79) # final mx = obj_mx ^ 12 

            diff_mx = mask_mx - obj_mx # negative value (-0.1): mask_mx < obj_mx

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (diff_mx)) # make mask values = diff_mx (1* diff_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = mask_mx (diff_mx + obj_mx), make obk values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , (obj_mx * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)

            # end for 8th option
            attention_scores[:, :, 0, :] = torch.mul( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
            #print("attn scores", attention_scores)
            '''


            '''   
            #- 8.6 obj + x, obj_mask + 0.5*x, mask + 0

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            obj_mx = 0.3 # 0.6 (+7.2), 0.1 (+1.2), 0.2 (+2.4), 1.0 (+12), 5.0 (+60) # final mx = obj_mx * 12

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (-obj_mx)) # make mask values = - obj_mx (1* -obj_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = 0, make obj values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , obj_mx ) # * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)


            # print("attn scores max_min before:")
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))


            # end for 8.3 option
            attention_scores[:, :, 0, :] = torch.add( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
           
            # print("attn scores max_min after":)
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))
           
            # print("attn scores", attention_scores)
            '''


            
            #- 8.7 obj + x, mask - y

            '''
            # top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
            # #print(top_k_max.size())
            # print(top_k_max[0].size())
            # top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

            # print("attn scores max_min before:")
            # print(torch.max(attention_scores[:, :, 0, :]))
            # print(torch.min(attention_scores[:, :, 0, :]))
            
            # print(top_k_max)
            # print(top_k_min)



            # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
            # #print(top_k_max.size())
            # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

            # print("attn scores max_min before 2:")
            # print(torch.max(attention_scores[:, :, :, 0]))
            # print(torch.min(attention_scores[:, :, :, 0]))
            
            # print(top_k_max)
            # print(top_k_min)



            # print("attn scores before", attention_scores[:, :, 0, :])



            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            mask_mx = -0.5 # (x0.02) , 0.685 # (x0.01) , 0.83 (x0.1) , 0.95 (x0.54) # final mx = mask_mx ^ 12
            obj_mx = 0.3 # (x50.2) , 1.469 # (x100.9) , 1.215 (x10.3) , 1.05 (x1.79) # final mx = obj_mx ^ 12 

            diff_mx = mask_mx - obj_mx # negative value (-0.1): mask_mx < obj_mx

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (diff_mx)) # make mask values = diff_mx (1* diff_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = mask_mx (diff_mx + obj_mx), make obk values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , obj_mx ) # * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)

            # end for 8.7 option
            attention_scores[:, :, 0, :] = torch.add( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
            


            # top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
            # top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

            # print("attn scores max_min after:")
            # print(torch.max(attention_scores[:, :, 0, :]))
            # print(torch.min(attention_scores[:, :, 0, :]))
           
            # print(top_k_max)
            # print(top_k_min)
            


            # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
            # #print(top_k_max.size())
            # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

            # print("attn scores max_min before 2:")
            # print(torch.max(attention_scores[:, :, :, 0]))
            # print(torch.min(attention_scores[:, :, :, 0]))
            
            # print(top_k_max)
            # print(top_k_min)




            # print("attn scores after", attention_scores[:, :, 0, :])
            '''


            '''
            #- 8.8 both hight and weight: obj + x 

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            obj_mx = 0.3 # 0.3 (+3.6), 10.0 (+120), 5.0 (+60) # final mx = obj_mx * 12

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (-obj_mx)) # make mask values = - obj_mx (1* -obj_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = 0, make obj values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w

            mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask_8[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h


            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , obj_mx ) # * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)


            # print("attn scores max_min before:")
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))


            # end for 8.8 option
            attention_scores[:, :, 0, :] = torch.add( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
            attention_scores[:, :, 1:, 0] = torch.add( attention_scores[:, :, 1:, 0] , mask_diag[:, None, 1:, 0] ) # [28, 12, 626, 626] x [28, None, 626, 626]

            # print("attn scores max_min after":)
            # print(torch.max(attention_scores))
            # print(torch.min(attention_scores))
           
            # print("attn scores", attention_scores)
            '''
            


            '''
            #- 8.10 obj + x, obj_mask + 0.5*x, mask - y

            # !!!!!!!! don't forget to return mask at the beginning of the if !!!!!!!!
            mask_mx = -0.5 # (x0.02) , 0.685 # (x0.01) , 0.83 (x0.1) , 0.95 (x0.54) # final mx = mask_mx ^ 12
            obj_mx = 0.3 # 0.6 (+7.2), 0.1 (+1.2), 0.2 (+2.4), 1.0 (+12), 5.0 (+60) # final mx = obj_mx * 12

            diff_mx = mask_mx - obj_mx # negative value (-0.1): mask_mx < obj_mx

            # WORKERS !!!!!!!!!

            mask_8 = torch.mul(mask, (diff_mx)) # make mask values = - obj_mx (1* -obj_mx)
            #print("my mask mul", mask_8)

            mask_8 = torch.add(mask_8, (obj_mx)) # make mask values = 0, make obj values = obj_mx (0 + obj_mx)
            #print("my mask add", mask_8)


            mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask_8[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
            mask_diag[:, 0, 0] = torch.add( mask_diag[:, 0, 0] , obj_mx ) # * 0.9999) ) # [28, 0, 0] x [28, 0] # cls_w[0] 
                                                                # p.s. * 0.9999 so that all the obj_mx values for softmax were the same (cuz after add it is obj_mx * 0.9999)
            #print("mask after", mask_diag)


            print("attn scores max_min before:")
            print(torch.max(attention_scores))
            print(torch.min(attention_scores))

            print("attn scores before", attention_scores)


            # end for 8.3 option
            attention_scores[:, :, 0, :] = torch.add( attention_scores[:, :, 0, :] , mask_diag[:, None, 0, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
           
            print("attn scores max_min after:")
            print(torch.max(attention_scores))
            print(torch.min(attention_scores))
           
            print("attn scores after", attention_scores)
            '''



            '''
            #- 8.11 obj + x, mask - y (new)

            x = random.random()
            if (x > 0.05) and (x < 0.07):
                print_info = True
            else:
                print_info = False

            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])



            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)



            # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
            # #print(top_k_max.size())
            # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

            # print("attn scores max_min before 2:")
            # print(torch.max(attention_scores[:, :, :, 0]))
            # print(torch.min(attention_scores[:, :, :, 0]))
            
            # print(top_k_max)
            # print(top_k_min)



            #print("attn scores before", attention_scores[:, :, 0, :])


            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)

            # for batch in range( attention_scores.size(0) ):
            #     for head in range( attention_scores.size(1) ):
            #         #for val in range( attention_scores.size(2) ):
            #             #attention_scores[batch, head, 0, val] = \

            #         attention_scores[batch, head, 0, :] = \
            #             torch.where( mask_626[batch, :] < 0.5, \
            #                  torch.add( attention_scores[batch, head, 0, :], (max_as[batch, head] * pow(1.025, (min_as[batch, head] - attention_scores[batch, head, 0, :]))) ), \
            #                       torch.add( attention_scores[batch, head, 0, :], (min_as[batch, head] * pow(1.025, (attention_scores[batch, head, 0, :] - max_as[batch, head]))) )
            #                         )

            #         # if ((i+1)%2 == 0): #or i ==2:
            #         #     vals[i] = vals[i] + (max_val * pow(1.025, (min_val - vals[i])))
            #         #     #vals[i] = vals[i] + (max_val * pow(1.1, ( (min_val/(max_val - min_val)) - vals[i]))) 

            #         # if ((i+3)%2 == 1):
            #         #     vals[i] = vals[i] + (min_val * pow(1.025, (vals[i] - max_val))) 


            # attention_scores[:, :, 0, :] = \
            #     torch.where( mask_626[:, :] < 0.5, \
            #             torch.add( attention_scores[:, :, 0, :], ( torch.mul( max_as[:, :, None], torch.pow(1.025, ( torch.add(min_as[:, :, None], attention_scores[:, :, 0, :]))))) ), \
            #                 torch.add( attention_scores[:, :, 0, :], \
            #                     torch.mul( min_as[:, :, None], \
            #                         torch.pow(1.025, \
            #                             torch.add(attention_scores[:, :, 0, :], max_as[:, :, None]))) )
            #                 )


            # print(min_as.size(), max_as.size())
            # print(attention_scores[:, :, 0, :].size())


            # attention_scores[:, :, 0, :] = \
            #     torch.where( mask_626[:, : , None] < 0.5, \
            #             torch.add( attention_scores[:, :, 0, :], torch.mul( max_as, ( torch.pow( 1.025, torch.sub( min_as, attention_scores[:, :, 0, :])))) ), \
            #                 torch.add( attention_scores[:, :, 0, :], \
            #                     torch.mul( min_as, \
            #                         ( torch.pow( 1.025, \
            #                             torch.sub( attention_scores[:, :, 0, :], max_as)))) )
            #                )



            #attention_scores_tmp = attention_scores[:, :, 0, :]
            #print(attention_scores_tmp.size())

            pow_my = 1.005
            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( attention_scores[:, :, 0, :], torch.mul( max_as[:, :, None], torch.pow( torch.tensor(pow_my).cuda(), torch.sub(min_as[:, :, None], attention_scores[:, :, 0, :]))) ), \
                            torch.add( attention_scores[:, :, 0, :], \
                                torch.mul( min_as[:, :, None], \
                                    torch.pow( torch.tensor(pow_my).cuda(), \
                                        torch.sub( attention_scores[:, :, 0, :], max_as[:, :, None]))) )
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            


            # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
            # #print(top_k_max.size())
            # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

            # print("attn scores max_min before 2:")
            # print(torch.max(attention_scores[:, :, :, 0]))
            # print(torch.min(attention_scores[:, :, :, 0]))
            
            # print(top_k_max)
            # print(top_k_min)


            
            # # 8.11 old:
            # # top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
            # # #print(top_k_max.size())
            # # #print(top_k_max[0].size())
            # # top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

            # # print("attn scores max_min before:")
            # # print(attention_scores[:, :, 0, :].size())

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = max_as.to(device='cuda')
            # min_as = min_as.to(device='cuda')

            # # print(min_as, max_as)
            # # print(min_as.size(), max_as.size())

            # # print(top_k_max)
            # # print(top_k_min)



            # # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
            # # #print(top_k_max.size())
            # # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

            # # print("attn scores max_min before 2:")
            # # print(torch.max(attention_scores[:, :, :, 0]))
            # # print(torch.min(attention_scores[:, :, :, 0]))
            
            # # print(top_k_max)
            # # print(top_k_min)



            # #print("attn scores before", attention_scores[:, :, 0, :])


            # mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            # mask_626 = mask_626.to(device='cuda')
            # mask_626[:, 1:] = mask[:, :]
            # mask_626[:, 0] = 0
            # mask_626 = mask_626.to(device='cuda')

            # #print(mask_626)


            # # for batch in range( attention_scores.size(0) ):
            # #     for head in range( attention_scores.size(1) ):
            # #         #for val in range( attention_scores.size(2) ):
            # #             #attention_scores[batch, head, 0, val] = \

            # #         attention_scores[batch, head, 0, :] = \
            # #             torch.where( mask_626[batch, :] < 0.5, \
            # #                  torch.add( attention_scores[batch, head, 0, :], torch.mul( max_as[batch, head], torch.pow( 1.025, torch.sub( min_as[batch, head], attention_scores[batch, head, 0, :]))) ), \
            # #                       torch.add( attention_scores[batch, head, 0, :], torch.mul( min_as[batch, head], torch.pow( 1.025, torch.sub( attention_scores[batch, head, 0, :], max_as[batch, head]))) )
            # #                         )

            # #         # if ((i+1)%2 == 0): #or i ==2:
            # #         #     vals[i] = vals[i] + (max_val * pow(1.025, (min_val - vals[i])))
            # #         #     #vals[i] = vals[i] + (max_val * pow(1.1, ( (min_val/(max_val - min_val)) - vals[i]))) 

            # #         # if ((i+3)%2 == 1):
            # #         #     vals[i] = vals[i] + (min_val * pow(1.025, (vals[i] - max_val))) 


            # for batch in range( attention_scores.size(0) ):
            #     attention_scores[batch, :, 0, :] = \
            #         torch.where( mask_626[batch, :] < 0.5, \
            #                 torch.add( attention_scores[batch, :, 0, :], torch.mul( max_as[batch, :, None], torch.pow( 1.025, torch.sub( min_as[batch, :, None], attention_scores[batch, :, 0, :]))) ), \
            #                     torch.add( attention_scores[batch, :, 0, :], torch.mul( min_as[batch, :, None], torch.pow( 1.025, torch.sub( attention_scores[batch, :, 0, :], max_as[batch, :, None]))) )
            #                     )   
            '''



            '''
            #- 8.12 obj + x, mask - y (stable)

            x = random.random()
            if (x > 0.00005) and (x < 0.00007):
                print_info = True
            else:
                print_info = False


            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])



            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)


                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)



            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)


            pow_my = 1.001 # 1.001, 1.0005 , 1.0001
            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( \
                            torch.div( attention_scores[:, :, 0, :] , torch.tensor(2.0).cuda()), \
                                torch.mul( \
                                    torch.div( max_as[:, :, None], torch.tensor(2.0).cuda()), \
                                        torch.pow( torch.tensor(pow_my).cuda(), \
                                            torch.sub(max_as[:, :, None], attention_scores[:, :, 0, :]))) ), \
                        torch.add( \
                            torch.div( attention_scores[:, :, 0, :], torch.tensor(2.0).cuda()), \
                                torch.mul( \
                                    torch.div( min_as[:, :, None], torch.tensor(2.0).cuda()), \
                                        torch.pow( torch.tensor(pow_my).cuda(), \
                                            torch.sub( attention_scores[:, :, 0, :], min_as[:, :, None]))) )
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            

                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)
            '''



            '''
            #- 8.12_2 (positive only) obj + x, mask - y (stable)

            x = random.random()
            if (x > 0.00005) and (x < 0.00007):
                print_info = True
            else:
                print_info = False


            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])



            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)


                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)



            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)


            pow_my = 1.01 # 1.001, 1.0005 , 1.0001
            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( \
                            torch.div( attention_scores[:, :, 0, :] , torch.tensor(2.0).cuda()), \
                                torch.mul( \
                                    torch.div( max_as[:, :, None], torch.tensor(2.0).cuda()), \
                                        torch.pow( torch.tensor(pow_my).cuda(), \
                                            torch.sub(max_as[:, :, None], attention_scores[:, :, 0, :]))) ), \
                        attention_scores[:, :, 0, :].float()
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            

                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)
            '''




            '''
            #- 8.15 (based on 8.12_2, positive only, smooth), obj + x, (stable)

            x = random.random()
            if (x > 0.00005) and (x < 0.00007):
                print_info = True
            else:
                print_info = False


            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])


            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)


                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)



            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)


            pow_my = 1.005 # 1.001, 1.0005 , 1.0001
            coeff_init = 0.6
            coeff_new = 0.4
            #coeff_new = 1.0 - coeff_init

            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( \
                            torch.mul( attention_scores[:, :, 0, :] , torch.tensor(coeff_init).cuda()), \
                                torch.mul( \
                                    torch.mul( max_as[:, :, None], torch.tensor(coeff_new).cuda()), \
                                        torch.pow( torch.tensor(pow_my).cuda(), \
                                            torch.sub(max_as[:, :, None], attention_scores[:, :, 0, :]))) ), \
                        attention_scores[:, :, 0, :].float()
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            

                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)
            '''


            '''
            #- 8.16 (based on 8.12_2, positive only, smooth), obj + x*0.7 (NO POW)

            x = random.random()
            if (x > 0.00005) and (x < 0.00007):
                print_info = True
            else:
                print_info = False


            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])



            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)


                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)



            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)


            #pow_my = 1.005 # 1.001, 1.0005 , 1.0001
            coeff_init = 0.5
            coeff_new = 0.5
            #coeff_new = 1.0 - coeff_init
            
            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( \
                            torch.mul( attention_scores[:, :, 0, :] , torch.tensor(coeff_init).cuda()), \
                                # torch.mul( \
                                    torch.mul( max_as[:, :, None], torch.tensor(coeff_new).cuda()) ), \
                                        # torch.pow( torch.tensor(pow_my).cuda(), \
                                        #     torch.sub(max_as[:, :, None], attention_scores[:, :, 0, :]))) ), \
                        attention_scores[:, :, 0, :] #.float()
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            

                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)
            '''




            #- 8.24 (positive only), obj + max*coeff

            x = random.random()
            if (x > 0.00005) and (x < 0.00007):
                print_info = True
            else:
                print_info = False


            if print_info:
                print("mask before", mask)

                print("attn scores before:")
                #print(attention_scores[:, :, 0, :].size())
                print(attention_scores[:, :, 0, :])



            max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]
            min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=False)[0]

            # max_as = torch.max(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]
            # min_as = torch.min(attention_scores[:, :, 0, :], dim=2, keepdim=True)[0]

            max_as = max_as.to(device='cuda')
            min_as = min_as.to(device='cuda')


            if print_info:

                print("attn scores max_min before:")
                print(max_as, min_as)
                #print(min_as.size(), max_as.size())

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                #print(top_k_max.size())
                #print(top_k_max[0].size())
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print(top_k_max)
                print(top_k_min)


                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)



            mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
            mask_626 = mask_626.to(device='cuda')
            mask_626[:, 1:] = mask[:, :]
            mask_626[:, 0] = 0

            if print_info:
                print("mask626:", mask_626)


            coeff_max = 0.25 #0.25
            #coeff_new = 1.0 - coeff_init
            
            attention_scores[:, :, 0, :] = \
                torch.where( mask_626[:, None, :] < 0.5, \
                        torch.add( attention_scores[:, :, 0, :], \
                            torch.mul( max_as[:, :, None] , torch.tensor(coeff_max).cuda()) ), \
                        attention_scores[:, :, 0, :] #.float()
                            )


            if print_info:

                print("attn scores after", attention_scores[:, :, 0, :])

                top_k_max = torch.topk(attention_scores[:, :, 0, :], 5, largest=True)
                top_k_min = torch.topk(attention_scores[:, :, 0, :], 5, largest=False)

                print("attn scores max_min after:")
                print(torch.max(attention_scores[:, :, 0, :]))
                print(torch.min(attention_scores[:, :, 0, :]))
            
                print(top_k_max)
                print(top_k_min)
            

                # top_k_max = torch.topk(attention_scores[:, :, :, 0], 5, largest=True)
                # #print(top_k_max.size())
                # top_k_min = torch.topk(attention_scores[:, :, :, 0], 5, largest=False)

                # print("attn scores max_min before 2:")
                # print(torch.max(attention_scores[:, :, :, 0]))
                # print(torch.min(attention_scores[:, :, :, 0]))
                
                # print(top_k_max)
                # print(top_k_min)




            '''
            # old:
            if True:
                #mask = mask * (-1e4) # -1e9 too small for gradiend for some reason
                
                mask_out = torch.mul(mask, (-5000)) # for 4, 5, 6

                mask = torch.mul(mask, (-1e4))
                #print("mask before", mask)


                ### 1st option:
                # old, with a mistake of -20000:
                #attention_scores[:, :, :, 1:] = torch.add( attention_scores[:, :, :, 1:] , mask[:, None, None, :] ) # [28, 12, 626, 1:626] x [28, None, None, 625] # patches_w + cls_w
                #attention_scores[:, :, 1:, :] = torch.add( attention_scores[:, :, 1:, :] , mask[:, None, :, None] ) # [28, 12, 1:626, 626] x [28, None, 625, None] # patches_h + cls_h

                # new:  
                # mask_diag = torch.zeros(mask.size(0), (mask.size(1) + 1), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
                # mask_diag = mask_diag.to(device='cuda')

                # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
                # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
                
                # for i in range(mask_diag.size(0)):
                #     mask_diag_temp = mask_diag[i, :, :]
                #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -15000.0, mask_diag_temp, torch.tensor(-10000.).cuda()) # < -10000 but -15000 just in case (expects -20000)

                # attention_scores[:, :, :, :] = torch.add( attention_scores[:, :, :, :] , mask_diag[:, None, :, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]

                # print( attention_scores[0, 0, 0, 2] )
                # print( attention_scores[0, 0, 0, 16] )
                # print("attn after", attention_scores.shape)


                ### 2nd option:
                # attention_scores[:, :, :, 1:] = torch.add( attention_scores[:, :, :, 1:] , mask[:, None, None, :] ) # [28, 12, 626, 1:626] x [28, None, None, 625] # patches_w + cls_w


                ### 3d option:
                # attention_scores[:, :, 1:, :] = torch.add( attention_scores[:, :, 1:, :] , mask[:, None, :, None] ) # [28, 12, 1:626, 626] x [28, None, 625, None] # patches_h + cls_h



                ### for 4, 5, 6 options !!!:
                #mask_diag = torch.zeros(28, 626, 626)
                mask_diag = torch.zeros(mask.size(0), (mask.size(1) + 1), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
                mask_diag = mask_diag.to(device='cuda')


                ## 4th option:
                # old, with a mistake of outliers:
                # #mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
                # #mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h
                
                # new:
                # mask_out = torch.mul(mask, (-5000))

                # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
                # mask_diag[:, 1:, 0] = torch.add( mask_diag[:, 1:, 0] , mask[:, :] ) # [28, 1:626, 0] x [28, 625] # cls_h

                # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask_out[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h

                # for i in range(mask_diag.size(0)):
                #     mask_diag_temp = mask_diag[i, :, :]

                #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
                #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # # 0 if > -7000 just in case (expects -5000)



                ## 5th option:
                # old, with a mistake of outliers:
                # #mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
                # #mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w

                # new_good (a mistake technically):
                #mask_out = torch.mul(mask, (-5000)) # need to multiply before changing mask !!!

                # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
                # mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w
                # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask_out[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h


                # for i in range(mask_diag.size(0)):
                #     mask_diag_temp = mask_diag[i, :, :]

                #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
                #     mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if > -7000 just in case (expects -5000)
                

                # new++:
                mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h (-10k)
                mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask_out[:, None, :] ) # [28, 626, 1:626] x [28, None, 625]  # patches_w + cls_w (-5k)

                #print("mask mid", mask_diag)

                for i in range(mask_diag.size(0)):
                    mask_diag_temp = mask_diag[i, :, :]

                    mask_diag[i, :, :] = torch.where( mask_diag_temp > -13000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -13000 just in case (expects -15000)
                    mask_diag[i, :, :] = torch.where( mask_diag_temp < -7000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if > -7000 just in case (expects -5000)

                mask_diag[:, 0, 1:] = torch.add( mask_diag[:, 0, 1:] , mask[:, :] ) # [28, 0, 1:626] x [28, 625] # cls_w  (-10k)
                
                #print("mask after", mask_diag)


                ## 6th option:
                # mask_diag[:, :, 1:] = torch.add( mask_diag[:, :, 1:] , mask[:, None, :] ) # [28, 626, 1:626] x [28, None, 625] # patches_w + cls_w
                # mask_diag[:, 1:, :] = torch.add( mask_diag[:, 1:, :] , mask[:, :, None] ) # [28, 1:626, 626] x [28, 625, None] # patches_h + cls_h
                
                # for i in range(mask_diag.size(0)):
                #     mask_diag_temp = mask_diag[i, :, :]

                #     #mask_diag[i, :, :] = torch.where( (mask_diag[i, :, :] < (-15000)), mask_diag[i, :, :], torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)
                #     mask_diag[i, :, :] = torch.where( mask_diag_temp > -15000.0, mask_diag_temp, torch.tensor(0.).cuda()) # 0 if < -15000 just in case (expects -20000)


                #-# for 4, 5, 6 options !!!:
                for i in range(mask_diag.size(0)):
                    mask_diag[i, :, :] = mask_diag[i, :, :].fill_diagonal_(0)
                    #mask_diag = mask_diag.fill_diagonal_(0)

                attention_scores[:, :, :, :] = torch.add( attention_scores[:, :, :, :] , mask_diag[:, None, :, :] ) # [28, 12, 626, 626] x [28, None, 626, 626]
                
                #print("attn after", attention_scores)
            '''

            ###



            # print('sum mask:',attention_scores.sum(-1,keepdim=True)[0,0,:])
            # print('sum shape mask:',attention_scores.sum(-1,keepdim=True).shape)

        attention_probs = self.softmax(attention_scores)
        

        # if print_info: # False at the beginning
        #     print("attn softmax", attention_probs)


        # print( "Softmax", attention_probs[0, 0, 0, 2] )
        # print( "Softmax", attention_probs[0, 0, 0, 16] )


        # print('sum:',attention_probs.sum(-2,keepdim=True)[0,0,:])
        # print('sum shape:',attention_probs.sum(-2,keepdim=True).shape)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights, self.softmax2(attention_scores)[:,:,:,0]
    


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)



        # My overlap:
        # overlap = True
        #slide = 12
        #


        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])


            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

            # My overlap:
            # #n_patches = ((img_size[0] - (patch_size[0] - slide)) // slide) * ((img_size[1] - (patch_size[1] - slide)) // slide) # my (same)
            #_patches = ((img_size[0] - patch_size[0]) // slide + 1) * ((img_size[1] - patch_size[1]) // slide + 1) # transFG
            #
            

            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       
                                       stride=patch_size,

                                       # My overlap:    
                                       #stride=(slide, slide) # transFG
                                       #                                       

                                       )


        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)


        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, mask=None):
        h = x
        x = self.attention_norm(x)
        x, weights, contribution = self.attn(x, mask)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights, contribution

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

            
class MAWS(nn.Module):
    # mutual attention weight selection
    def __init__(self):
        super(MAWS, self).__init__()

    def forward(self, x, contributions):
        length = x.size()[1]

        contributions = contributions.mean(1)
        weights = x[:,:,0,:].mean(1)

        scores = contributions*weights

        max_inx = torch.argsort(scores, dim=1,descending=True)


        return None, max_inx            

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.feature_fusion=config.feature_fusion
        self.num_token = config.num_token
        num_layers = config.transformer["num_layers"]
        if config.feature_fusion:
            self.ff_token_select = MAWS()
            self.ff_last_layer = Block(config,vis)
            num_layers -= 1
            self.ff_encoder_norm=LayerNorm(config.hidden_size, eps=1e-6)
        else:
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))


    #def forward(self, hidden_states):
    def forward(self, hidden_states, mask=None):

        attn_weights = []
        contributions = []
        tokens = [[] for i in range(hidden_states.shape[0])]
        for layer_block in self.layer:

            #hidden_states, weights, contribution = layer_block(hidden_states)
            hidden_states, weights, contribution = layer_block(hidden_states, mask)

            if self.feature_fusion:
                # perform feature fusion
                selected_num, selected_inx = self.ff_token_select(weights,contribution)
                B = selected_inx.shape[0]
                for i in range(B):
                    tokens[i].extend(hidden_states[i, selected_inx[i,:self.num_token]])
            #'''
            if self.vis:
                attn_weights.append(weights)
                contributions.append(contribution)

        if self.feature_fusion:
            # perform feature fusion
            tokens=[torch.stack(token) for token in tokens]
            tokens = torch.stack(tokens).squeeze(1)
            concat = torch.cat((hidden_states[:,0].unsqueeze(1), tokens), dim=1)
            #print('concat shape', concat.shape)
            ff_states, ff_weights, ff_contri = self.ff_last_layer(concat)
            encoded = self.ff_encoder_norm(ff_states) 
        else:
            encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
        


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    #def forward(self, input_ids):
    def forward(self, input_ids, mask=None):

        embedding_output = self.embeddings(input_ids)

        #encoded, attn_weights = self.encoder(embedding_output)
        encoded, attn_weights = self.encoder(embedding_output, mask)

        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False, vis=False, dataset='cotton'):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.smoothing_value = smoothing_value
        self.classifier = config.classifier
        self.dataset=dataset

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        self.feature_fusion = config.feature_fusion

    #def forward(self, x, labels=None):
    def forward(self, x, x_crop=None, labels=None, mask=None, mask_crop=None):

        x, attn_weights = self.transformer(x, mask)
        feats = x[:, 0]
        logits = self.head(x[:, 0])


        # My refine:
        if x_crop is not None:
            x_crop, attn_weights_crop = self.transformer(x_crop, mask_crop)
            logits_crop = self.head(x_crop[:, 0])
        #


        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
                #loss_fct = FocalLoss()

            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
                # refine_loss_criterion = LabelSmoothing(self.smoothing_value)
                # loss_fct = CrossEntropyLoss()
                # loss_fct = FocalLoss()



            #refine_loss_criterion = CrossEntropyLoss()
            refine_loss_criterion = FocalLoss()


            
            # My refine:
            if x_crop is not None:
                #print("[INFO]: Refine los")
                #ce_loss = loss_fct(logits_crop.view(-1, self.num_classes), labels.view(-1))
                ce_loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

                ##refine_loss = F.kl_div(logits_crop.softmax(dim=-1).log(), logits.softmax(dim=-1), reduction='batchmean') #reduction='sum')
                #refine_loss = F.kl_div(logits_crop.log_softmax(dim=-1), logits.softmax(dim=-1), reduction='batchmean') #reduction='sum')
                
                refine_loss = refine_loss_criterion(logits_crop.view(-1, self.num_classes), logits.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())

                if torch.isinf(refine_loss):
                    print("[INFO]: Skip Refine Loss")
                    loss = ce_loss
                else:
                    loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.1) # 0.1 #0.01
                    #loss = ce_loss + (refine_loss * 0.1) #0.01
                    #loss = (0.5 * ce_loss) + (0.5 * refine_loss) #0.01

                print("[INFO]: ce loss:", ce_loss.item(), "Refine loss:", refine_loss.item(), "Final loss:", loss.item())

            else:

                ce_loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

                mask_loss_check = False # True # False
                delayed = False #True
                if mask_loss_check:               
                    if delayed:
                        if ce_loss > 0.1: #0.01: #0.005:
                            loss = ce_loss # FFVT
                            print("ce loss:", ce_loss)
                        else:
                            mask_loss_val = mask_loss(attn_weights, mask)
                            loss = ce_loss + mask_loss_val # or mb 0.01*mask_loss_val
                            print("[INFO]: Loss delay")
                            print("ce loss:", ce_loss.item(), "mask loss:", mask_loss_val.item(), "final loss:", loss.item())
                    else:
                        mask_loss_val = mask_loss(attn_weights, mask)
                        loss = ce_loss + mask_loss_val # or mb 0.01*mask_loss_val
                        print("ce loss:", ce_loss.item(), "mask loss:", mask_loss_val.item(), "final loss:", loss.item())

                else:
                    # contrast_loss = con_loss(x[:, 0], labels.view(-1)) # transFG
                    # loss = ce_loss + contrast_loss # transFG
                    loss = ce_loss # FFVT


            if x_crop is not None:
                #print("[INFO]: Refine los indeed")

                return loss, logits #logits_crop

            else:
                return loss, logits

        else:
            #return logits, attn_weights
            return logits, feats

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            if self.feature_fusion:
                pass
            else:
                self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('ff') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss                        



def mask_loss(attn_weights, mask): # technically cross-entropy loss with only negative part
    

    cls_attn_weights = attn_weights[-1][:, :, 0, :] # 12x(bs, 12, 626, 626) -> (bs, 12, 626); mb 1:626 ?
    cls_attn_weights = cls_attn_weights.to(device='cuda')


    # max_as = torch.max(cls_attn_weights[:, :, :], dim=2, keepdim=False)[0]
    # min_as = torch.min(cls_attn_weights[:, :, :], dim=2, keepdim=False)[0]

    # # max_as = torch.max(attention_scores[:, :, :], dim=2, keepdim=True)[0]
    # # min_as = torch.min(attention_scores[:, :, :], dim=2, keepdim=True)[0]

    # # max_as = max_as.to(device='cuda')
    # # min_as = min_as.to(device='cuda')

    #if print_info:
        # print("cls attn scores max_min before:")
        # print(max_as, min_as)
        # #print(min_as.size(), max_as.size())

        # top_k_max = torch.topk(cls_attn_weights[:, :, :], 50, largest=True)
        # #print(top_k_max.size())
        # #print(top_k_max[0].size())
        # top_k_min = torch.topk(cls_attn_weights[:, :, :], 50, largest=False)

        # print(top_k_max)
        # print(top_k_min)


    mask_626 = torch.zeros(mask.size(0), (mask.size(1) + 1)) #, dtype=torch.float64) # dtype=torch.double)
    mask_626 = mask_626.to(device='cuda')
    mask_626[:, 1:] = mask[:, :]
    mask_626[:, 0] = 0 # 0 - positive (object), 1 - negative (background)
    
    ##mask626_nonzero = torch.count_nonzero(mask_626, dim=1)
    #mask626_nonzero = (mask_626 == 1.).sum(dim=1, keepdim=False)
    #mask626_nonzero = mask_626.size(1) - mask626_nonzero

            
    mask_loss_val_temp = torch.zeros(cls_attn_weights.size(0), cls_attn_weights.size(1), cls_attn_weights.size(2)) #, dtype=torch.float64) # dtype=torch.double)
    mask_loss_val_temp = mask_loss_val_temp.to(device='cuda')


    mask_loss_val_temp = mask_loss_val_temp.half()
    mask_loss_val_temp[:, :, :] = \
        torch.where( mask_626[:, None, :] > 0.5, \
                cls_attn_weights[:, :, :].half() , \
                mask_loss_val_temp[:, :, :].half() #.float() .cuda()
                )

    mask_loss_val_temp_nonzero = (mask_loss_val_temp == 0.).sum(dim=2, keepdim=False)
    mask_loss_val_temp_nonzero = mask_loss_val_temp.size(2) - mask_loss_val_temp_nonzero
    mask_loss_val_temp_nonzero = mask_loss_val_temp_nonzero.half().cuda()

    mask_loss_val_temp_nonzero[:, :] = \
            torch.where( mask_loss_val_temp_nonzero[:, :] == 0, \
                    torch.tensor(1.0).half().cuda() , \
                    mask_loss_val_temp_nonzero[:, :].half()
                    )

    #print("temp loss", mask_loss_val_temp)

    # #negCEloss = torch.sum(torch.log(1 - mask_loss_val_temp), dim=2, keepdim=True) / mask_loss_val_temp.size(2)
    # negCEloss = - ( torch.sum(torch.log(1 - mask_loss_val_temp)) / ( mask_loss_val_temp.size(0) * mask_loss_val_temp.size(2) ) )

    # for batch in range(mask_loss_val_temp.size(0))
    #negCEloss = ( torch.sum(torch.log(1 - mask_loss_val_temp), dim=2, keepdim=False) / (mask_loss_val_temp.size(0) * (mask626_nonzero.unsqueeze(1))) )
    negCEloss = ( torch.sum(torch.log(1 - mask_loss_val_temp), dim=2, keepdim=False) / ((mask_loss_val_temp.size(0) * mask_loss_val_temp_nonzero)) ) # + 1e-16) )
    negCEloss = - ( torch.sum(negCEloss))


    loss_multiplier = 0.01 # In crop + add: 0.001 (91.3) # 0.01 (91.25) # 0.1 (91.13) # 1 (90.45) # 10 (90.75) # 100 (90.28)
    #negCEloss = (negCEloss * loss_multiplier)

    # if negCEloss < 0.1: negCEloss = (negCEloss * loss_multiplier) # too much if crop is disabled
    # if negCEloss < 0.1: negCEloss = (negCEloss * loss_multiplier)
    #if negCEloss < 0.1: negCEloss = (negCEloss * loss_multiplier)
    
    #print("mask loss:", negCEloss)

    return (negCEloss * loss_multiplier)



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
