from __future__ import print_function
import matplotlib.pyplot as plt

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


img_name = 'tulips'
fname = '../data/enhancement/%s.bmp'%img_name

if not os.path.exists('./figs'):
        os.mkdir('./figs')

#read image
img_pil = crop_image(get_image(fname, -1)[0], d=32)
img_np = pil_to_np(img_pil)

#input type
INPUT = 'fourier' # 'meshgrid', 'noise', 'fourier'
var=1
input_depth = 32
net_input = get_noise(input_depth, INPUT, (img_pil.size[1]//32, img_pil.size[0]//32),var=var).type(dtype).detach()

#network parameters
ln_lambda=1 #the lambda in Lipschitz normalization, which is used to control spectral bias
upsample_mode='bilinear'#['deconv', 'nearest', 'bilinear', 'gaussian'], where 'gaussian' denotes our Gaussian upsampling. 
pad = 'reflection'
#decoder is the used network architecture in the paper
net = decoder(num_input_channels=input_depth, num_output_channels=3, ln_lambda=ln_lambda,
                   upsample_mode=upsample_mode, pad=pad, need_sigmoid=True, need_bias=True).type(dtype)

#optimization parameters
OPTIMIZER='adam'
num_iter = 100
LR = 0.001
reg_noise_std = 0#1./30, injecting noise in the input.
show_every = 100

# Loss
mse = torch.nn.MSELoss().type(dtype)
img_torch = np_to_torch(img_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
def closure():

    global i, out, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out, img_torch)
    total_loss.backward()

    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    print ('Iteration: %05d, Loss: %f, PSRN_gt: %f' % (i, total_loss.item(), psrn_gt))

    i += 1

    return total_loss

optimize(OPTIMIZER, net.parameters(), closure, LR, num_iter)

#visualization
out_np = torch_to_np(out)
save2enhanceimg(img_np, out_np, "./figs/%s_enhanced.png" % img_name)#save the enhanced image
save2img(out_np, "./figs/%s_output.png" % img_name)#save the output image
