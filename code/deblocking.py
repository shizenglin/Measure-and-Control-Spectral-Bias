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


#baboon,barbara,boats,lena,peppers
img_name = 'peppers'
fname = '../data/deblocking/classic5/%s.bmp'%img_name

if not os.path.exists('./figs'):
        os.mkdir('./figs')

if not os.path.exists('./logs'):
        os.mkdir('./logs')

log_path = "./logs/%s.txt"%img_name
log_file = open(log_path, "w")

#read image
img_pil = crop_image(get_image(fname, -1)[0], d=32)
img_np = pil_to_np(img_pil)

# compression level
quality_ = 10#10,20,30

compressed_fname = '../data/deblocking/classic5/%s_%d.jpg'%(img_name,quality_)
if not os.path.exists(compressed_fname):
    img_pil.save(compressed_fname, quality=quality_, optimize=True, progressive=True)#get compressed image

img_noisy_pil, img_noisy_np = get_image(compressed_fname, -1)

#input type
INPUT = 'fourier' # 'meshgrid', 'noise', 'fourier'
var=1
input_depth = 32
net_input = get_noise(input_depth, INPUT, (img_pil.size[1]//32, img_pil.size[0]//32),var=var).type(dtype).detach()

#network parameters
ln_lambda=1.4#the lambda in Lipschitz normalization, which is used to control spectral bias
upsample_mode='bilinear'#['deconv', 'nearest', 'bilinear', 'gaussian'], where 'gaussian' denotes our Gaussian upsampling. 
pad = 'reflection'
#decoder is the used network architecture in the paper
net = decoder(num_input_channels=input_depth, num_output_channels=1, ln_lambda=ln_lambda,
                   upsample_mode=upsample_mode, pad=pad, need_sigmoid=True, need_bias=True).type(dtype)

#optimization parameters
OPTIMIZER='adam'
num_iter = 10000
LR = 0.001
reg_noise_std = 0#1./30, injecting noise in the input.
show_every = 100

#automatic stopping
ratio_list = np.zeros((num_iter))
ratio_iter=100#the n in Eq. (8)
ratio_epsilon=0.01#the ratio difference threshold
auto_stop = False

# Loss
mse = torch.nn.MSELoss().type(dtype)
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
last_net = None
psrn_noisy_last = 0

i = 0
def closure():

    global i, out, psrn_noisy_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0])

    pre_img = out.detach().cpu().numpy()[0]
    pre_img = pre_img.transpose(1, 2, 0)
    noisy_img = img_noisy_np.transpose(1, 2, 0)
    
    #frequency-band correspondence metric
    avg_mask_it = get_circular_statastic(pre_img[:,:,0], noisy_img[:,:,0],  size=0.2)

    #automatic stopping
    blur_it = PerceptualBlurMetric (pre_img[:,:,0])#the blurriness of the output image
    sharp_it = MLVSharpnessMeasure(pre_img[:,:,0])#the sharpness of the output image
    ratio_it = blur_it/sharp_it#the ratio

    if auto_stop:
        ratio_list[i] = ratio_it
        if i>ratio_iter*2:
            ratio1 = np.mean(ratio_list[i-ratio_iter*2:i-ratio_iter])
            ratio2 = np.mean(ratio_list[i-ratio_iter+1:i])
            if np.abs(ratio1-ratio2)<ratio_epsilon:
                print("The optimization is automatically stopped!")
                out_np = torch_to_np(out)
                save2img(out_np, "./figs/%s_deblocked.png" % img_name)
                exit()

    print ('Iteration: %05d, Loss: %f, PSRN_gt: %f' % (i, total_loss.item(), psrn_gt))
    log_file.write('Iteration: %05d, Loss: %f, PSRN_gt: %f, mask: %s, ratio: %f\n' % (i, total_loss.item(), psrn_gt, avg_mask_it, ratio_it))
    log_file.flush()

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss

optimize(OPTIMIZER, net.parameters(), closure, LR, num_iter)
log_file.close()

#visualization

out_np = torch_to_np(out)
save2img(out_np, "./figs/%s_deblocked.png" % img_name)#save the denoised image

frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
get_fbc_fig(frequency_lists,num_iter,ylim=1,save_path="./figs/%s_fbc.png"%img_name)#save the fbc figure

data_lists =[]
data_lists.append(psnr_list)
data_lists.append(ratio_list)
get_psnr_ratio_fig(data_lists,num_iter,ylim=35, ylabel='PSNR', save_path="./figs/%s_psnr_ratio.png"%img_name)#save the psnr_ratio figure

