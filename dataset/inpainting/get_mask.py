#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:44:35 2021

@author: zenglin
"""


import imageio
import numpy as np

# im = imageio.imread('people_mask_ori.jpg')
# im[im<125] = 0
# im[im>=125] = 255
# print(np.unique(im))
# imageio.imwrite('people_mask.jpg', 255-im)


im = imageio.imread('bird.jpg')

im_mask = imageio.imread('bird_mask.jpg')
im_mask[im_mask<125] = 0
im_mask[im_mask>=125] = 1
print(np.unique(im_mask))
im = im*im_mask
im[im==0] = 255
imageio.imwrite('bird+mask.jpg', im)