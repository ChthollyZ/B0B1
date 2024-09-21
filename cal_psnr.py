from utils import utils_image as util
import cv2
import os
import numpy as np
import torch
import math



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 31.00 dB   31.02 31.04

B0_path = '/tmp/dataset/DIV2K_B0_another/'
GT_path = '/tmp/dataset/DIV2K_HR/'
B1_path = '/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/'

B0 = []
for b0_path in sorted(os.listdir(B0_path)):
    B0.append(cv2.imread(B0_path + b0_path))

GT = []
for gt_path in sorted(os.listdir(GT_path)):
    GT.append(cv2.imread(GT_path + gt_path))

B1 = []
B1.append(cv2.imread('/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/babyx4/babyx4_2400.png'))
B1.append(cv2.imread('/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/birdx4/birdx4_2400.png'))
B1.append(cv2.imread('/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/butterflyx4/butterflyx4_2400.png'))
B1.append(cv2.imread('/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/headx4/headx4_2400.png'))
B1.append(cv2.imread('/root/gsr/loss_change/loss_change/B1_L2loss_patch_anotherB0/images/womanx4/womanx4_2400.png'))

psnt_list = []
ave = 0
for i in range(800):
    print(i)
    # b0 = util.bgr2ycbcr(B0[i].astype(np.float32) / 255.) * 255.
    # b1 = util.bgr2ycbcr(B1[i].astype(np.float32) / 255.) * 255.
    b0 = np.float32(B0[i]/255.) 
    # b1 = np.float32(B1[i]/255.)
    # b = ( (b1.astype(np.float32) + b0.astype(np.float32)) / 2 * 255.0).round().astype(np.uint8)
    # b = util.bgr2ycbcr(b.astype(np.float32) / 255.) * 255.

    gt = GT[i]/255.
    # gt = gt.float().clamp_(0, 1)
    gt = np.uint8((gt*255.0).round())
    gt = util.bgr2ycbcr(gt.astype(np.float32) / 255.) * 255.
    # psnt_list.append(util.calculate_psnr(b, gt, border=4))
    psnt_list.append(util.calculate_psnr(util.bgr2ycbcr(b0.astype(np.float32)) * 255., gt, border=4))

psnt_list = np.array(psnt_list)
print(psnt_list)
print(np.mean(psnt_list))
