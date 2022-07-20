# coding: utf-8
# https://linuxtut.com/en/fe2d3308b3ba56a80c7a/

import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import random
from seq_dataset import data_provider
from skimage.measure import compare_ssim
from VAE_model import VanillaVAE
import lpips
from models import *
loss_fn_alex = lpips.LPIPS(net='alex')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class CONV_NN_v2(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(CONV_NN_v2, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.args = args
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 3, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list, m_rest):
        pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list\
            = self.forward(x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list, m_rest)
        return pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden, h_t, c_t, delta_c_list, delta_m_list, m_rest):
        if self.args.num_views == 4:
            msg_feat = torch.cat([m_t, m_t_others] + m_rest[:2], -1)
            x = torch.cat([x_t, msg_feat], -1)
        elif self.args.num_views == 8:
            msg_feat = torch.cat([m_t, m_t_others] + m_rest[:6], -1)
            x = torch.cat([x_t, msg_feat], -1)
        else:
            msg_feat = torch.cat([m_t, m_t_others], -1)
            x = torch.cat([x_t, msg_feat], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list, m_rest):
        pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list\
            = self.forward(x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list, m_rest)
        return pred_x_tp1.data, message.data, memory, h_t, c_t, delta_c_list, delta_m_list


def run_steps(x_batch, model_0, model_1, vae, inference=True, args=None, model_2=None, model_3=None,
              model_4=None, model_5=None, model_6=None, model_7=None):
    '''
    x_batch: [bs, T, 2*2],
    pred_batch: [bs, T, 1*2],
    x_batch: [bs, T, H, W, C], C=3
    pred_batch: [bs, T, H, W, C],
    '''
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    batch = x_batch.shape[0]
    height = x_batch.shape[2]
    width = x_batch.shape[3]

    h_t_0 = []
    c_t_0 = []
    h_t_1 = []
    c_t_1 = []
    delta_c_list_0 = []
    delta_m_list_0 = []
    delta_c_list_1 = []
    delta_m_list_1 = []
    if args.num_views in [4, 8]:
        h_t_2 = []
        c_t_2 = []
        h_t_3 = []
        c_t_3 = []
        delta_c_list_2 = []
        delta_m_list_2 = []
        delta_c_list_3 = []
        delta_m_list_3 = []
        if args.num_views in [8, ]:
            h_t_4 = []
            c_t_4 = []
            h_t_5 = []
            c_t_5 = []
            h_t_6 = []
            c_t_6 = []
            h_t_7 = []
            c_t_7 = []
            delta_c_list_4 = []
            delta_m_list_4 = []
            delta_c_list_5 = []
            delta_m_list_5 = []
            delta_c_list_6 = []
            delta_m_list_6 = []
            delta_c_list_7 = []
            delta_m_list_7 = []
    decouple_loss = []

    for i in range(len(num_hidden)):
        zeros = None  # torch.zeros([batch, num_hidden[i], height, width]).to(args.device)
        h_t_0.append(zeros)
        c_t_0.append(zeros)
        h_t_1.append(zeros)
        c_t_1.append(zeros)
        delta_c_list_0.append(zeros)
        delta_m_list_0.append(zeros)
        delta_c_list_1.append(zeros)
        delta_m_list_1.append(zeros)
        if args.num_views in [4, 8]:
            h_t_2.append(zeros)
            c_t_2.append(zeros)
            h_t_3.append(zeros)
            c_t_3.append(zeros)
            delta_c_list_2.append(zeros)
            delta_m_list_2.append(zeros)
            delta_c_list_3.append(zeros)
            delta_m_list_3.append(zeros)
            if args.num_views in [8, ]:
                h_t_4.append(zeros)
                c_t_4.append(zeros)
                h_t_5.append(zeros)
                c_t_5.append(zeros)
                h_t_6.append(zeros)
                c_t_6.append(zeros)
                h_t_7.append(zeros)
                c_t_7.append(zeros)
                delta_c_list_4.append(zeros)
                delta_m_list_4.append(zeros)
                delta_c_list_5.append(zeros)
                delta_m_list_5.append(zeros)
                delta_c_list_6.append(zeros)
                delta_m_list_6.append(zeros)
                delta_c_list_7.append(zeros)
                delta_m_list_7.append(zeros)

    memory_0 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    memory_1 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    pred_batch_0_list = []
    pred_batch_1_list = []
    message_0_list = []
    message_1_list = []
    if args.num_views in [4, ]:
        memory_2 = None
        memory_3 = None
        x_0_t, x_1_t, x_2_t, x_3_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
        pred_batch_2_list = []
        pred_batch_3_list = []
        message_2_list = []
        message_3_list = []
    if args.num_views in [8, ]:
        memory_2 = None
        memory_3 = None
        memory_4 = None
        memory_5 = None
        memory_6 = None
        memory_7 = None
        x_0_t, x_1_t, x_2_t, x_3_t, x_4_t, x_5_t, x_6_t, x_7_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
        pred_batch_2_list = []
        pred_batch_3_list = []
        message_2_list = []
        message_3_list = []
        pred_batch_4_list = []
        pred_batch_5_list = []
        message_4_list = []
        message_5_list = []
        pred_batch_6_list = []
        pred_batch_7_list = []
        message_6_list = []
        message_7_list = []
    else:
        x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // 2, dim=-1)
    if args.message_type == 'raw_data':
        message_0 = x_0_t[:, 0:0 + 1]
        message_1 = x_1_t[:, 0:0 + 1]
        if args.num_views in [4, 8]:
            message_2 = x_2_t[:, 0:0 + 1]
            message_3 = x_3_t[:, 0:0 + 1]
            if args.num_views in [8, ]:
                message_4 = x_4_t[:, 0:0 + 1]
                message_5 = x_5_t[:, 0:0 + 1]
                message_6 = x_6_t[:, 0:0 + 1]
                message_7 = x_7_t[:, 0:0 + 1]
        else:
            message_2 = None
            message_3 = None
            message_4 = None
            message_5 = None
            message_6 = None
            message_7 = None
    elif args.message_type == 'vae':
        message_0 = vae.get_message(x_0_t[:, 0:0 + 1])
        message_1 = vae.get_message(x_1_t[:, 0:0 + 1])
        if args.num_views in [4, 8]:
            message_2 = vae.get_message(x_2_t[:, 0:0 + 1])
            message_3 = vae.get_message(x_3_t[:, 0:0 + 1])
            if args.num_views in [8,]:
                message_4 = vae.get_message(x_4_t[:, 0:0 + 1])
                message_5 = vae.get_message(x_5_t[:, 0:0 + 1])
                message_6 = vae.get_message(x_6_t[:, 0:0 + 1])
                message_7 = vae.get_message(x_7_t[:, 0:0 + 1])
        else:
            message_2 = None
            message_3 = None
            message_4 = None
            message_5 = None
            message_6 = None
            message_7 = None
    else:
        message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
        message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
        if args.num_views in [4, 8]:
            message_2 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
            message_3 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
            if args.num_views in [8, ]:
                message_4 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
                message_5 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
                message_6 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
                message_7 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
        else:
            message_2 = None
            message_3 = None
            message_4 = None
            message_5 = None
            message_6 = None
            message_7 = None
    if args.eval_mode == 'multi_step_eval'and inference == True:  # and inference == True
        x_0_t_pred_prev = x_0_t[:, 0:1]
        x_1_t_pred_prev = x_1_t[:, 0:1]
        use_gt_flag = False
        if args.num_views in [4, 8]:
            x_2_t_pred_prev = x_2_t[:, 0:1]
            x_3_t_pred_prev = x_3_t[:, 0:1]
            if args.num_views in [8,]:
                x_4_t_pred_prev = x_4_t[:, 0:1]
                x_5_t_pred_prev = x_5_t[:, 0:1]
                x_6_t_pred_prev = x_6_t[:, 0:1]
                x_7_t_pred_prev = x_7_t[:, 0:1]
        else:
            x_2_t_pred_prev = None
            x_3_t_pred_prev = None
            x_4_t_pred_prev = None
            x_5_t_pred_prev = None
            x_6_t_pred_prev = None
            x_7_t_pred_prev = None
        for t in range(args.eval_num_step + args.num_past):
            x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
                model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
                        delta_m_list_0, [message_2, message_3, message_4, message_5, message_6, message_7])
            if args.message_type in ['vae']:
                # print('x_0_t_pred.shape: ', x_0_t_pred.shape)
                if t < args.num_past or use_gt_flag: # or t % args.mask_per_step == 0:
                    message_0 = vae.get_message(x_0_t[:, t:t + 1])
                else:
                    message_0 = vae.get_message(x_0_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                if t < args.num_past or use_gt_flag: # or t % args.mask_per_step == 0:
                    message_0 = x_0_t[:, t:t + 1]
                else:
                    message_0 = x_0_t_pred
            elif args.message_type == 'zeros':
                message_0 = torch.zeros_like(message_0)
            elif args.message_type == 'randn':
                message_0 = torch.randn_like(message_0)
            x_1_t_pred, message_1, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1 = \
                model_1(x_1_t_pred_prev, message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1,
                        delta_m_list_1, [message_2, message_3, message_4, message_5, message_6, message_7])
            if args.message_type in ['vae']:
                if t < args.num_past or use_gt_flag: # or t % args.mask_per_step == 0:
                    message_1 = vae.get_message(x_1_t[:, t:t + 1])
                else:
                    message_1 = vae.get_message(x_1_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                if t < args.num_past or use_gt_flag: # or t % args.mask_per_step == 0:
                    message_1 = x_1_t[:, t:t + 1]
                else:
                    message_1 = x_1_t_pred
            elif args.message_type == 'zeros':
                message_1 = torch.zeros_like(message_1)
            elif args.message_type == 'randn':
                message_1 = torch.randn_like(message_1)
            if args.num_views in [4, 8]:
                x_2_t_pred, message_2, memory_2, h_t_2, c_t_2, delta_c_list_2, delta_m_list_2 = \
                    model_2(x_2_t_pred_prev, message_1, message_0, memory_2, h_t_2, c_t_2, delta_c_list_2,
                            delta_m_list_2, [message_2, message_3, message_4, message_5, message_6, message_7])
                if args.message_type in ['vae']:
                    if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                        message_2 = vae.get_message(x_2_t[:, t:t + 1])
                    else:
                        message_2 = vae.get_message(x_2_t_pred_prev.detach())
                elif args.message_type in ['raw_data']:
                    if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                        message_2 = x_2_t[:, t:t + 1]
                    else:
                        message_2 = x_2_t_pred
                elif args.message_type == 'zeros':
                    message_2 = torch.zeros_like(message_2)
                elif args.message_type == 'randn':
                    message_2 = torch.randn_like(message_2)

                x_3_t_pred, message_3, memory_3, h_t_3, c_t_3, delta_c_list_3, delta_m_list_3 = \
                    model_3(x_3_t_pred_prev, message_1, message_0, memory_3, h_t_3, c_t_3, delta_c_list_3,
                            delta_m_list_3, [message_2, message_3, message_4, message_5, message_6, message_7])
                if args.message_type in ['vae']:
                    if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                        message_3 = vae.get_message(x_3_t[:, t:t + 1])
                    else:
                        message_3 = vae.get_message(x_3_t_pred_prev.detach())
                elif args.message_type in ['raw_data']:
                    if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                        message_3 = x_3_t[:, t:t + 1]
                    else:
                        message_3 = x_3_t_pred
                elif args.message_type == 'zeros':
                    message_3 = torch.zeros_like(message_3)
                elif args.message_type == 'randn':
                    message_3 = torch.randn_like(message_3)
                if args.num_views in [8, ]:
                    x_4_t_pred, message_4, memory_4, h_t_4, c_t_4, delta_c_list_4, delta_m_list_4 = \
                        model_4(x_4_t_pred_prev, message_1, message_0, memory_4, h_t_4, c_t_4, delta_c_list_4,
                                delta_m_list_4, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_4 = vae.get_message(x_4_t[:, t:t + 1])
                        else:
                            message_4 = vae.get_message(x_4_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_4 = x_4_t[:, t:t + 1]
                        else:
                            message_4 = x_4_t_pred
                    elif args.message_type == 'zeros':
                        message_4 = torch.zeros_like(message_4)
                    elif args.message_type == 'randn':
                        message_4 = torch.randn_like(message_4)

                    x_5_t_pred, message_5, memory_5, h_t_5, c_t_5, delta_c_list_5, delta_m_list_5 = \
                        model_5(x_5_t_pred_prev, message_1, message_0, memory_5, h_t_5, c_t_5, delta_c_list_5,
                                delta_m_list_5, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_5 = vae.get_message(x_5_t[:, t:t + 1])
                        else:
                            message_5 = vae.get_message(x_5_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_5 = x_5_t[:, t:t + 1]
                        else:
                            message_5 = x_5_t_pred
                    elif args.message_type == 'zeros':
                        message_5 = torch.zeros_like(message_5)
                    elif args.message_type == 'randn':
                        message_5 = torch.randn_like(message_5)

                    x_6_t_pred, message_6, memory_6, h_t_6, c_t_6, delta_c_list_6, delta_m_list_6 = \
                        model_6(x_6_t_pred_prev, message_1, message_0, memory_6, h_t_6, c_t_6, delta_c_list_6,
                                delta_m_list_6, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_6 = vae.get_message(x_6_t[:, t:t + 1])
                        else:
                            message_6 = vae.get_message(x_6_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_6 = x_6_t[:, t:t + 1]
                        else:
                            message_6 = x_6_t_pred
                    elif args.message_type == 'zeros':
                        message_6 = torch.zeros_like(message_6)
                    elif args.message_type == 'randn':
                        message_6 = torch.randn_like(message_6)

                    x_7_t_pred, message_7, memory_7, h_t_7, c_t_7, delta_c_list_7, delta_m_list_7 = \
                        model_7(x_7_t_pred_prev, message_1, message_0, memory_7, h_t_7, c_t_7, delta_c_list_7,
                                delta_m_list_7, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_7 = vae.get_message(x_7_t[:, t:t + 1])
                        else:
                            message_7 = vae.get_message(x_7_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        if t < args.num_past or use_gt_flag:  # or t % args.mask_per_step == 0:
                            message_7 = x_7_t[:, t:t + 1]
                        else:
                            message_7 = x_7_t_pred
                    elif args.message_type == 'zeros':
                        message_7 = torch.zeros_like(message_7)
                    elif args.message_type == 'randn':
                        message_7 = torch.randn_like(message_7)

            else:
                message_2 = None
                message_3 = None
                message_4 = None
                message_5 = None
                message_6 = None
                message_7 = None
            if t < args.num_past: # or t % args.mask_per_step == 0:
                x_0_t_pred_prev = x_0_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t + 1:t + 2]
                if args.num_views in [4, 8]:
                    x_2_t_pred_prev = x_2_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                    x_3_t_pred_prev = x_3_t[:, t + 1:t + 2]
                    if args.num_views in [8,]:
                        x_4_t_pred_prev = x_4_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_5_t_pred_prev = x_5_t[:, t + 1:t + 2]
                        x_6_t_pred_prev = x_6_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_7_t_pred_prev = x_7_t[:, t + 1:t + 2]
            elif use_gt_flag:
                x_0_t_pred_prev = x_0_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t + 1:t + 2]
                pred_batch_0_list.append(x_0_t_pred.cpu())
                pred_batch_1_list.append(x_1_t_pred.cpu())
                message_0_list.append(message_0.cpu())
                message_1_list.append(message_1.cpu())
                if args.num_views in [4, 8]:
                    x_2_t_pred_prev = x_2_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                    x_3_t_pred_prev = x_3_t[:, t + 1:t + 2]
                    pred_batch_2_list.append(x_2_t_pred.cpu())
                    pred_batch_3_list.append(x_3_t_pred.cpu())
                    message_2_list.append(message_2.cpu())
                    message_3_list.append(message_3.cpu())
                    if args.num_views in [8, ]:
                        x_4_t_pred_prev = x_4_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_5_t_pred_prev = x_5_t[:, t + 1:t + 2]
                        pred_batch_4_list.append(x_4_t_pred.cpu())
                        pred_batch_5_list.append(x_5_t_pred.cpu())
                        message_4_list.append(message_4.cpu())
                        message_5_list.append(message_5.cpu())
                        x_6_t_pred_prev = x_6_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_7_t_pred_prev = x_7_t[:, t + 1:t + 2]
                        pred_batch_6_list.append(x_6_t_pred.cpu())
                        pred_batch_7_list.append(x_7_t_pred.cpu())
                        message_6_list.append(message_6.cpu())
                        message_7_list.append(message_7.cpu())
            else:
                x_0_t_pred_prev = x_0_t_pred.detach()
                x_1_t_pred_prev = x_1_t_pred.detach()
                pred_batch_0_list.append(x_0_t_pred.cpu())
                pred_batch_1_list.append(x_1_t_pred.cpu())
                message_0_list.append(message_0.cpu())
                message_1_list.append(message_1.cpu())
                if args.num_views in [4, 8]:
                    x_2_t_pred_prev = x_2_t_pred.detach()
                    x_3_t_pred_prev = x_3_t_pred.detach()
                    pred_batch_2_list.append(x_2_t_pred.cpu())
                    pred_batch_3_list.append(x_3_t_pred.cpu())
                    message_2_list.append(message_2.cpu())
                    message_3_list.append(message_3.cpu())
                    if args.num_views in [8, ]:
                        x_4_t_pred_prev = x_4_t_pred.detach()
                        x_5_t_pred_prev = x_5_t_pred.detach()
                        pred_batch_4_list.append(x_4_t_pred.cpu())
                        pred_batch_5_list.append(x_5_t_pred.cpu())
                        message_4_list.append(message_4.cpu())
                        message_5_list.append(message_5.cpu())
                        x_6_t_pred_prev = x_6_t_pred.detach()
                        x_7_t_pred_prev = x_7_t_pred.detach()
                        pred_batch_6_list.append(x_6_t_pred.cpu())
                        pred_batch_7_list.append(x_7_t_pred.cpu())
                        message_6_list.append(message_6.cpu())
                        message_7_list.append(message_7.cpu())
            if t % args.eval_per_step == 0 and args.mode != 'train':
                memory_0 = None
                memory_1 = None
                if args.num_views in [4, 8]:
                    memory_2 = None
                    memory_3 = None
                    if args.num_views in [8, ]:
                        memory_4 = None
                        memory_5 = None
                        memory_6 = None
                        memory_7 = None
                use_gt_flag = not use_gt_flag
                print('t: {}, use_gt_flag: {}'.format(t, use_gt_flag))
    else:
        x_0_t_pred_prev = x_0_t[:, 0:1]
        x_1_t_pred_prev = x_1_t[:, 0:1]
        if args.num_views in [4, 8]:
            x_2_t_pred_prev = x_2_t[:, 0:1]
            x_3_t_pred_prev = x_3_t[:, 0:1]
            if args.num_views in [8, ]:
                x_4_t_pred_prev = x_4_t[:, 0:1]
                x_5_t_pred_prev = x_5_t[:, 0:1]
                x_6_t_pred_prev = x_6_t[:, 0:1]
                x_7_t_pred_prev = x_7_t[:, 0:1]
        else:
            x_2_t_pred_prev = None
            x_3_t_pred_prev = None
            x_4_t_pred_prev = None
            x_5_t_pred_prev = None
            x_6_t_pred_prev = None
            x_7_t_pred_prev = None
        for t in range(args.num_step + args.num_past):
            x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
                model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
                        delta_m_list_0, [message_2, message_3, message_4, message_5, message_6, message_7])
            if args.message_type in ['vae']:
                # message_0 = vae.get_message(x_0_t_pred.detach())
                if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                    message_0 = vae.get_message(x_0_t[:, t:t + 1])
                else:
                    message_0 = vae.get_message(x_0_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                message_0 = x_0_t[:, t:t + 1]
            elif args.message_type == 'zeros':
                message_0 = torch.zeros_like(message_0)
            elif args.message_type == 'randn':
                message_0 = torch.randn_like(message_0)
            x_1_t_pred, message_1, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1 = \
                model_1(x_1_t_pred_prev, message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1,
                        delta_m_list_1, [message_2, message_3, message_4, message_5, message_6, message_7])
            if args.message_type in ['vae']:
                if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step): # or t % args.mask_per_step == 0:
                    message_1 = vae.get_message(x_1_t[:, t:t + 1])
                else:
                    message_1 = vae.get_message(x_1_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                message_1 = x_1_t[:, t:t + 1]
            elif args.message_type == 'zeros':
                message_1 = torch.zeros_like(message_1)
            elif args.message_type == 'randn':
                message_1 = torch.randn_like(message_1)
            if args.num_views in [4, 8]:
                x_2_t_pred, message_2, memory_2, h_t_2, c_t_2, delta_c_list_2, delta_m_list_2 = \
                    model_2(x_2_t_pred_prev, message_1, message_0, memory_2, h_t_2, c_t_2, delta_c_list_2,
                            delta_m_list_2, [message_2, message_3, message_4, message_5, message_6, message_7])
                if args.message_type in ['vae']:
                    if t < args.num_past or np.random.uniform(0, 1) > (
                            1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                        message_2 = vae.get_message(x_2_t[:, t:t + 1])
                    else:
                        message_2 = vae.get_message(x_2_t_pred_prev.detach())
                elif args.message_type in ['raw_data']:
                    message_2 = x_2_t[:, t:t + 1]
                elif args.message_type == 'zeros':
                    message_2 = torch.zeros_like(message_2)
                elif args.message_type == 'randn':
                    message_2 = torch.randn_like(message_2)
                x_3_t_pred, message_3, memory_3, h_t_3, c_t_3, delta_c_list_3, delta_m_list_3 = \
                    model_3(x_3_t_pred_prev, message_1, message_0, memory_3, h_t_3, c_t_3, delta_c_list_3,
                            delta_m_list_3, [message_2, message_3, message_4, message_5, message_6, message_7])
                if args.message_type in ['vae']:
                    if t < args.num_past or np.random.uniform(0, 1) > (
                            1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                        message_3 = vae.get_message(x_3_t[:, t:t + 1])
                    else:
                        message_3 = vae.get_message(x_3_t_pred_prev.detach())
                elif args.message_type in ['raw_data']:
                    message_3 = x_3_t[:, t:t + 1]
                elif args.message_type == 'zeros':
                    message_3 = torch.zeros_like(message_3)
                elif args.message_type == 'randn':
                    message_3 = torch.randn_like(message_3)
                if args.num_views in [8, ]:
                    x_4_t_pred, message_4, memory_4, h_t_4, c_t_4, delta_c_list_4, delta_m_list_4 = \
                        model_4(x_4_t_pred_prev, message_1, message_0, memory_4, h_t_4, c_t_4, delta_c_list_4,
                                delta_m_list_4, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or np.random.uniform(0, 1) > (
                                1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                            message_4 = vae.get_message(x_4_t[:, t:t + 1])
                        else:
                            message_4 = vae.get_message(x_4_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        message_4 = x_4_t[:, t:t + 1]
                    elif args.message_type == 'zeros':
                        message_4 = torch.zeros_like(message_4)
                    elif args.message_type == 'randn':
                        message_4 = torch.randn_like(message_4)
                    x_5_t_pred, message_5, memory_5, h_t_5, c_t_5, delta_c_list_5, delta_m_list_5 = \
                        model_5(x_5_t_pred_prev, message_1, message_0, memory_5, h_t_5, c_t_5, delta_c_list_5,
                                delta_m_list_5, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or np.random.uniform(0, 1) > (
                                1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                            message_5 = vae.get_message(x_5_t[:, t:t + 1])
                        else:
                            message_5 = vae.get_message(x_5_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        message_5 = x_5_t[:, t:t + 1]
                    elif args.message_type == 'zeros':
                        message_5 = torch.zeros_like(message_5)
                    elif args.message_type == 'randn':
                        message_5 = torch.randn_like(message_5)
                    x_6_t_pred, message_6, memory_6, h_t_6, c_t_6, delta_c_list_6, delta_m_list_6 = \
                        model_6(x_6_t_pred_prev, message_1, message_0, memory_6, h_t_6, c_t_6, delta_c_list_6,
                                delta_m_list_6, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or np.random.uniform(0, 1) > (
                                1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                            message_6 = vae.get_message(x_6_t[:, t:t + 1])
                        else:
                            message_6 = vae.get_message(x_6_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        message_6 = x_6_t[:, t:t + 1]
                    elif args.message_type == 'zeros':
                        message_6 = torch.zeros_like(message_6)
                    elif args.message_type == 'randn':
                        message_6 = torch.randn_like(message_6)
                    x_7_t_pred, message_7, memory_7, h_t_7, c_t_7, delta_c_list_7, delta_m_list_7 = \
                        model_7(x_7_t_pred_prev, message_1, message_0, memory_7, h_t_7, c_t_7, delta_c_list_7,
                                delta_m_list_7, [message_2, message_3, message_4, message_5, message_6, message_7])
                    if args.message_type in ['vae']:
                        if t < args.num_past or np.random.uniform(0, 1) > (
                                1 - 1 / args.mask_per_step):  # or t % args.mask_per_step == 0:
                            message_7 = vae.get_message(x_7_t[:, t:t + 1])
                        else:
                            message_7 = vae.get_message(x_7_t_pred_prev.detach())
                    elif args.message_type in ['raw_data']:
                        message_7 = x_7_t[:, t:t + 1]
                    elif args.message_type == 'zeros':
                        message_7 = torch.zeros_like(message_7)
                    elif args.message_type == 'randn':
                        message_7 = torch.randn_like(message_7)
            if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step): # or t % args.mask_per_step == 0:
                x_0_t_pred_prev = x_0_t[:, t+1:t+2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t+1:t+2]
                if args.num_views in [4, 8]:
                    x_2_t_pred_prev = x_2_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                    x_3_t_pred_prev = x_3_t[:, t + 1:t + 2]
                    if args.num_views in [8]:
                        x_4_t_pred_prev = x_4_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_5_t_pred_prev = x_5_t[:, t + 1:t + 2]
                        x_6_t_pred_prev = x_6_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                        x_7_t_pred_prev = x_7_t[:, t + 1:t + 2]
            else:
                x_0_t_pred_prev = x_0_t_pred.detach()
                x_1_t_pred_prev = x_1_t_pred.detach()
                if args.num_views in [4, 8]:
                    x_2_t_pred_prev = x_2_t_pred.detach()
                    x_3_t_pred_prev = x_3_t_pred.detach()
                    if args.num_views in [8, ]:
                        x_4_t_pred_prev = x_4_t_pred.detach()
                        x_5_t_pred_prev = x_5_t_pred.detach()
                        x_6_t_pred_prev = x_6_t_pred.detach()
                        x_7_t_pred_prev = x_7_t_pred.detach()
            pred_batch_0_list.append(x_0_t_pred)
            pred_batch_1_list.append(x_1_t_pred)
            message_0_list.append(message_0)
            message_1_list.append(message_1)
            if args.num_views in [4, 8]:
                pred_batch_2_list.append(x_2_t_pred)
                pred_batch_3_list.append(x_3_t_pred)
                message_2_list.append(message_2)
                message_3_list.append(message_3)
                if args.num_views in [8, ]:
                    pred_batch_4_list.append(x_4_t_pred)
                    pred_batch_5_list.append(x_5_t_pred)
                    message_4_list.append(message_4)
                    message_5_list.append(message_5)
                    pred_batch_6_list.append(x_6_t_pred)
                    pred_batch_7_list.append(x_7_t_pred)
                    message_6_list.append(message_6)
                    message_7_list.append(message_7)

    pred_batch_0 = torch.cat(pred_batch_0_list, 1)
    pred_batch_1 = torch.cat(pred_batch_1_list, 1)
    if args.num_views in [4, ]:
        pred_batch_2 = torch.cat(pred_batch_2_list, 1)
        pred_batch_3 = torch.cat(pred_batch_3_list, 1)
        pred_batch = torch.cat([pred_batch_0, pred_batch_1, pred_batch_2, pred_batch_3], -1)
    elif args.num_views in [8, ]:
        pred_batch_2 = torch.cat(pred_batch_2_list, 1)
        pred_batch_3 = torch.cat(pred_batch_3_list, 1)
        pred_batch_4 = torch.cat(pred_batch_4_list, 1)
        pred_batch_5 = torch.cat(pred_batch_5_list, 1)
        pred_batch_6 = torch.cat(pred_batch_6_list, 1)
        pred_batch_7 = torch.cat(pred_batch_7_list, 1)
        pred_batch = torch.cat([pred_batch_0, pred_batch_1, pred_batch_2, pred_batch_3,
                                pred_batch_4, pred_batch_5, pred_batch_6, pred_batch_7], -1)
    else:
        pred_batch = torch.cat([pred_batch_0, pred_batch_1], -1)
    message_batch_0 = torch.cat(message_0_list, 1)
    message_batch_1 = torch.cat(message_1_list, 1)
    if args.num_views in [4, ]:
        message_batch_2 = torch.cat(message_2_list, 1)
        message_batch_3 = torch.cat(message_3_list, 1)
        message_batch = torch.cat([message_batch_0, message_batch_1, message_batch_2, message_batch_3], -1)
    elif args.num_views in [8, ]:
        message_batch_2 = torch.cat(message_2_list, 1)
        message_batch_3 = torch.cat(message_3_list, 1)
        message_batch_4 = torch.cat(message_4_list, 1)
        message_batch_5 = torch.cat(message_5_list, 1)
        message_batch_6 = torch.cat(message_6_list, 1)
        message_batch_7 = torch.cat(message_7_list, 1)
        message_batch = torch.cat([message_batch_0, message_batch_1, message_batch_2, message_batch_3,
                                   message_batch_4, message_batch_5, message_batch_6, message_batch_7], -1)
    else:
        message_batch = torch.cat([message_batch_0, message_batch_1], -1)
    return pred_batch, message_batch


def training(N, Nte, bs, n_epoch, act, data_mode, args):
    # x_train, t_train, x_test, t_test, step_test = get_data(N, Nte, num_step, data_mode)
    train_input_handle, test_input_handle = data_provider(
        args.data_name, args.train_data_paths, args.valid_data_paths, args.bs, args.img_width,
        seq_length=args.num_step + args.num_past + 1, is_training=True, num_views=args.num_views, img_channel=args.img_channel,
        baseline=args.baseline, eval_batch_size=args.vis_bs, args=args)
    if args.message_type in ['raw_data']:
        input_dim = 3 + 3 * args.num_views
    elif args.message_type in ['vae']:
        input_dim = 3 + args.vae_latent_dim * args.num_views
    else:
        input_dim = 3+1 * args.num_views
    # h_units = [input_dim, input_dim]
    h_units = [int(x) for x in args.num_hidden.split(',')]
    if args.mode == 'eval' and args.ckpt_dir is not None:
        model_0_path = os.path.join(args.ckpt_dir, "model_0.pt")
        model_1_path = os.path.join(args.ckpt_dir, "model_1.pt")
        model_0 = torch.load(model_0_path)
        model_1 = torch.load(model_1_path)
        if args.num_views in [4, 8]:
            model_2_path = os.path.join(args.ckpt_dir, "model_2.pt")
            model_3_path = os.path.join(args.ckpt_dir, "model_3.pt")
            model_2 = torch.load(model_2_path)
            model_3 = torch.load(model_3_path)
            if args.num_views in [8, ]:
                model_4_path = os.path.join(args.ckpt_dir, "model_4.pt")
                model_5_path = os.path.join(args.ckpt_dir, "model_5.pt")
                model_4 = torch.load(model_4_path)
                model_5 = torch.load(model_5_path)
                model_6_path = os.path.join(args.ckpt_dir, "model_6.pt")
                model_7_path = os.path.join(args.ckpt_dir, "model_7.pt")
                model_6 = torch.load(model_6_path)
                model_7 = torch.load(model_7_path)
        print('Loaded num_view: {}, model_0 from {}, model_1 from {}'.format(args.num_views, model_0_path, model_1_path))
    else:
        model_0 = CONV_NN_v2(input_dim, h_units, act, args)
        model_1 = CONV_NN_v2(input_dim, h_units, act, args)
        if args.num_views in [4,]:
            model_2 = CONV_NN_v2(input_dim, h_units, act, args)
            model_3 = CONV_NN_v2(input_dim, h_units, act, args)
            print('Created model_0, model_1, model_2, model_3')
        elif args.num_views in [8,]:
            model_2 = CONV_NN_v2(input_dim, h_units, act, args)
            model_3 = CONV_NN_v2(input_dim, h_units, act, args)
            model_4 = CONV_NN_v2(input_dim, h_units, act, args)
            model_5 = CONV_NN_v2(input_dim, h_units, act, args)
            model_6 = CONV_NN_v2(input_dim, h_units, act, args)
            model_7 = CONV_NN_v2(input_dim, h_units, act, args)
            print('Created model_0, model_1, model_2, model_3 model_4, model_5, model_6, model_7')
        else:
            print('Created model_0, model_1')
    model_0 = model_0.to(args.device)
    model_1 = model_1.to(args.device)
    # model_list = [model_0, model_1]
    if args.num_views in [4,]:
        model_2 = model_2.to(args.device)
        model_3 = model_3.to(args.device)
    if args.num_views in [8,]:
        model_2 = model_2.to(args.device)
        model_3 = model_3.to(args.device)
        model_4 = model_4.to(args.device)
        model_5 = model_5.to(args.device)
        model_6 = model_6.to(args.device)
        model_7 = model_7.to(args.device)
    else:
        model_2 = None
        model_3 = None
        model_4 = None
        model_5 = None
        model_6 = None
        model_7 = None
        # model_list.extend([model_2, model_3])
    # vae = VanillaVAE(input_dim, h_units, act, args)
    vae_path = os.path.join(args.vae_ckpt_dir, args.data_name, 'vae.pt')
    vae = torch.load(vae_path)
    vae = vae.to(args.device)
    print('Loaded VAE model_0 from {}'.format(vae_path))

    if args.num_views in [4, ]:
        optimizer = optim.Adam(list(model_0.parameters()) + list(model_1.parameters())
                               + list(model_2.parameters()) + list(model_3.parameters()))
    if args.num_views in [8, ]:
        optimizer = optim.Adam(list(model_0.parameters()) + list(model_1.parameters())
                               + list(model_2.parameters()) + list(model_3.parameters())
                               + list(model_4.parameters()) + list(model_5.parameters())
                               + list(model_6.parameters()) + list(model_7.parameters()))
    else:
        optimizer = optim.Adam(list(model_0.parameters()) + list(model_1.parameters()))
    MSE = nn.MSELoss()

    tr_loss = []
    te_loss = []
    root_res_path = os.path.join(args.gen_frm_dir, args.data_name)
    if os.path.exists(os.path.join(root_res_path, "{}/Pred".format(act))) == False:
        os.makedirs(os.path.join(root_res_path, "{}/Pred".format(act)))

    start_time = time.time()
    print("START")
    best_eval_loss = np.inf
    for epoch in range(1, n_epoch + 1):
        if args.mode == 'train':
            model_0.train()
            model_1.train()
            if args.num_views in [4, 8]:
                model_2.train()
                model_3.train()
                if args.num_views in [8,]:
                    model_4.train()
                    model_5.train()
                    model_6.train()
                    model_7.train()
            sum_loss = 0
            print('Training ... {}'.format(epoch))
            train_input_handle.begin(do_shuffle=True)
            while (train_input_handle.no_batch_left() == False):
                ims = train_input_handle.get_batch()
                train_input_handle.next()
                x_batch = ims[:, :]
                gt_batch = ims[:, 1:]
                x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                optimizer.zero_grad()
                pred_batch, message_batch = run_steps(x_batch, model_0, model_1, vae,
                                                      inference=False, args=args,
                                                      model_2=model_2, model_3=model_3, model_4=model_4, model_5=model_5,
                                                      model_6=model_6, model_7=model_7)
                loss = MSE(pred_batch, gt_batch)
                loss.backward()
                optimizer.step()
                sum_loss += loss.data * bs

            ave_loss = sum_loss / (N - args.num_step)
            tr_loss.append(ave_loss.cpu())
            train_stats = {'ave_loss': ave_loss}

        if epoch % 10 == 0:  # 20  args.mode == 'eval' and
            batch_id = 0
            res_path = os.path.join(root_res_path, str(epoch))
            os.makedirs(res_path, exist_ok=True)
            avg_mse = 0
            img_mse, ssim, psnr = [], [], []
            lp = []
            for i in range(args.eval_num_step):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                lp.append(0)
            if args.eval_mode != 'multi_step_eval':
                for i in range(args.num_past):
                    img_mse.append(0)
                    ssim.append(0)
                    psnr.append(0)
                    lp.append(0)
            test_input_handle.begin(do_shuffle=False)
            while (test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                ims = test_input_handle.get_batch()
                test_input_handle.next()
                x_test = ims[:, :]
                if args.eval_mode == 'multi_step_eval':
                    t_test = ims[:, 1 + args.num_past:]
                else:
                    t_test = ims[:, 1:]
                with torch.no_grad():
                    y_test, message = run_steps(torch.from_numpy(x_test.astype(np.float32)).to(args.device),
                                                model_0, model_1, vae,
                                                inference=True, args=args,
                                                model_2=model_2, model_3=model_3, model_4=model_4, model_5=model_5,
                                                model_6=model_6, model_7=model_7)
                y_test = y_test.detach().cpu().numpy()
                message = message.detach().cpu().numpy()
                # MSE per frame
                for i in range(y_test.shape[1]):
                    x = y_test[:, i, :, :, :]
                    gx = t_test[:, i, :, :, :]
                    mse = np.square(x - gx).mean()
                    img_mse[i] += mse
                    avg_mse += mse
                    # cal lpips
                    img_x = np.zeros([y_test.shape[0], 3, y_test.shape[2], y_test.shape[3]])

                    img_x[:, 0, :, :] = x[:, :, :, 0]
                    img_x[:, 1, :, :] = x[:, :, :, 1]
                    img_x[:, 2, :, :] = x[:, :, :, 2]

                    img_x = torch.FloatTensor(img_x)
                    img_gx = np.zeros([y_test.shape[0], 3, y_test.shape[2], y_test.shape[3]])

                    img_gx[:, 0, :, :] = gx[:, :, :, 0]
                    img_gx[:, 1, :, :] = gx[:, :, :, 1]
                    img_gx[:, 2, :, :] = gx[:, :, :, 2]

                    img_gx = torch.FloatTensor(img_gx)
                    lp_loss = loss_fn_alex(img_x, img_gx)
                    lp[i] += torch.mean(lp_loss).item()

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)

                    psnr[i] += batch_psnr(pred_frm, real_frm)
                    for b in range(y_test.shape[0]):
                        score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                        ssim[i] += score

                # save prediction examples
                if batch_id <= args.num_save_samples:
                    path = os.path.join(res_path, str(batch_id))
                    os.mkdir(path)
                    for view_idx in range(args.num_views):
                        for i in range(y_test.shape[1]):
                            name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            img_gt = np.uint8(t_test[0, i, :, :, (view_idx * args.img_channel):(
                                        (view_idx + 1) * args.img_channel)] * 255)
                            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(file_name, img_gt)
                        for i in range(y_test.shape[1]):
                            name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            img_pd = y_test[0, i, :, :,
                                     (view_idx * args.img_channel):((view_idx + 1) * args.img_channel)]
                            # in range (0, 1)
                            img_pd = np.uint8(img_pd * 255)
                            img_pd = cv2.cvtColor(img_pd, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(file_name, img_pd)
                        for i in range(y_test.shape[1]):
                            name = 'msg_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            if args.message_type in ['raw_data']:
                                img_pd = message[0, i, :, :,
                                         (view_idx * 1) * 3:((view_idx + 1) * 3)]
                            else:
                                img_pd = message[0, i, :, :,
                                     (view_idx * 1):((view_idx + 1) * 1)]
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)

                avg_mse = np.mean(img_mse)
                print('mse per seq: ' + str(avg_mse))
                for i in range(args.num_step):
                    print('step {}, mse: {}'.format(i, img_mse[i]))

                ssim = np.asarray(ssim, dtype=np.float32)
                print('ssim per frame: ' + str(np.mean(ssim)))
                for i in range(args.num_step):
                    print('step {}, ssim: {}'.format(i, ssim[i]))

                psnr = np.asarray(psnr, dtype=np.float32)
                print('psnr per frame: ' + str(np.mean(psnr)))
                for i in range(args.num_step):
                    print('step {}, psnr: {}'.format(i, psnr[i]))

                lp = np.asarray(lp, dtype=np.float32)
                print('lpips per frame: ' + str(np.mean(lp)))
                for i in range(args.num_step):
                    print('step {}, lp: {}'.format(i, lp[i]))
                print('Save to {}.'.format(res_path))

        if args.mode == 'train':
            torch.save(model_0, os.path.join(root_res_path, "model_0.pt"))
            torch.save(model_1, os.path.join(root_res_path, "model_1.pt"))
            if args.num_views in [4, 8]:
                torch.save(model_2, os.path.join(root_res_path, "model_2.pt"))
                torch.save(model_3, os.path.join(root_res_path, "model_3.pt"))
                if args.num_views in [8, ]:
                    torch.save(model_4, os.path.join(root_res_path, "model_4.pt"))
                    torch.save(model_5, os.path.join(root_res_path, "model_5.pt"))
                    torch.save(model_6, os.path.join(root_res_path, "model_6.pt"))
                    torch.save(model_7, os.path.join(root_res_path, "model_7.pt"))

    print("END")

    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    plt.figure(figsize=(5, 4))
    plt.plot(tr_loss, label="training")
    plt.plot(te_loss, label="test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.savefig(os.path.join(root_res_path, "{}/loss_history.png".format(act)))
    plt.clf()
    plt.close()

def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data_mode', type=str, default="(y_{t-1}, y_t)->y_{t+1}", help='(y_{t-1}, y_t)->y_{t+1}')
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="eval", help='train / eval')
    parser.add_argument('--eval_mode', type=str, default='multi_step_eval', help='multi_step_eval / single_step_eval')
    parser.add_argument('--eval_num_step', type=int, default=350, help='10, 350')
    parser.add_argument('--eval_per_step', type=int, default=30)
    parser.add_argument('--mask_per_step', type=int, default=1000000000)
    parser.add_argument('--log_per_epoch', type=int, default=10)
    parser.add_argument('--num_step', type=int, default=6, help='10, ')
    parser.add_argument('--num_past', type=int, default=3)
    parser.add_argument('--num_cl_step', type=int, default=20)
    parser.add_argument('--n_epoch', type=int, default=100, help='200')
    parser.add_argument('--bs', type=int, default=5)
    parser.add_argument('--vis_bs', type=int, default=5)
    parser.add_argument('--Nte', type=int, default=20, help='200')
    parser.add_argument('--N', type=int, default=100, help='1000')

    parser.add_argument('--data_name', type=str, default="carla_town02_8_view_20220713_color_split_8",
                        help='SINE; circle_motion; students003, fluid_flow_1')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 cuda:0; cpu:0 cpu:0')
    parser.add_argument('--with_comm', type=str2bool, default=False, help='whether to use communication')
    parser.add_argument('--train_data_paths', type=str, default="../../../../tools/circle_motion_30/train",
                        help='../tools/${DATASET_NAME}/train, ../../../../tools/circle_motion_30/train, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # carla_town02_20211201
    parser.add_argument('--valid_data_paths', type=str, default="../../../../tools/circle_motion_30/eval",
                        help='../tools/${DATASET_NAME}/eval, ../../../../tools/circle_motion_30/eval, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # sumo_sanjose-2021-11-06_20.28.57_30
    # RGB dataset
    parser.add_argument('--img_width', type=int, default=128, help='img width')
    parser.add_argument('--num_views', type=int, default=8, help='num views')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')
    parser.add_argument('--baseline', type=str, default='1_NN_4_img_GCN',
                        help='1_NN_1_img_no_GCN, 1_NN_4_img_no_GCN, 4_NN_4_img_GCN, 1_NN_4_img_GCN, 4_NN_4_img_no_GCN, '
                             '4_NN_4_img_FC, 4_NN_4_img_Identity')
    parser.add_argument('--gen_frm_dir', type=str, default='results/')
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--num_hidden', type=str, default='16', help='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--version', type=str, default='predrnn', help='version')
    parser.add_argument('--message_type', type=str, default='vae', help='normal, zeros, randn, raw_data, vae')
    parser.add_argument('--ckpt_dir', type=str, default='results/20220714-194117/carla_town02_8_view_20220713_color_split_8/', help='checkpoint dir: dir/model_1.pt, results/20220714-074032/carla_town02_8_view_20220713_color_split_8/')
    parser.add_argument('--vae_ckpt_dir', type=str, default='results/vae_carla',
                        help='vae checkpoint dir: kessel: 1c-results/20220109-212140/circle_motion/vae.pt, '
                             '20220110-234817, chpc-gpu005: 1c-20220113-154323, wo ln: 20220117-044713')
    parser.add_argument('--cl_mode', type=str, default='sliding_window', help='full_history, sliding_window')
    parser.add_argument('--vae_latent_dim', type=int, default=1)

    args = parser.parse_args()

    h_units = [10, 10]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.gen_frm_dir = os.path.join(args.gen_frm_dir, timestr)
    args.train_data_paths = "../../../../tools/{}_30/train".format(args.data_name)
    args.valid_data_paths = "../../../../tools/{}_30/eval".format(args.data_name)
    training(args.N, args.Nte, args.bs, args.n_epoch, args.act, args.data_mode, args)
