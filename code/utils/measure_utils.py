import os
from .common_utils import *
from scipy.signal import convolve2d as conv2

def get_circular_statastic(img_it, img_gt, size=0.2):

    if len(img_it.shape)==3:
        img_it = rgb2gray(img_it)
    
    if len(img_gt.shape)==3:
        img_gt = rgb2gray(img_gt)

    assert(size>0 and size<1)

    ftimage_it = np.fft.fft2(img_it)
    ftimage_it = abs(np.fft.fftshift(ftimage_it))

    ftimage_gt = np.fft.fft2(img_gt)
    ftimage_gt = abs(np.fft.fftshift(ftimage_gt))

    m_data = ftimage_it/(ftimage_gt+1e-8)
    m_data = np.clip(m_data, 0, 1)

    h,w = m_data.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    avg_mask_list = []
    pre_mask = np.zeros((h,w))
    for sz in np.linspace(size, 1, int(1/size)):

        radius = center[0]*sz#pow(center[0]**2+center[1]**2,0.5)
        mask = dist_from_center <= radius
        mask = mask.astype(np.int32)

        mask_sz = (mask-pre_mask).astype(np.int32)
        pre_mask = mask

        avg_mask_list.append(np.sum(mask_sz*m_data)/np.sum(mask_sz))

    return avg_mask_list

def PerceptualBlurMetric (Image, FiltSize=9):

    if len(Image.shape)==3:
        Image = rgb2gray(Image)

    m, n = Image.shape[0],Image.shape[1]

    Hv = 1.0/FiltSize*np.ones((1,FiltSize))
    Hh = Hv.T
    Bver = conv2(Image, Hv, 'same')
    Bhor = conv2(Image, Hh, 'same')

    s_ind = int(np.ceil(FiltSize/2))
    e_ind = int(np.floor(FiltSize/2))
    Bver = Bver[s_ind:m-e_ind, s_ind:n-e_ind]
    Bhor = Bhor[s_ind:m-e_ind, s_ind:n-e_ind]
    Image = Image[s_ind:m-e_ind, s_ind:n-e_ind]
    m, n = Image.shape[0],Image.shape[1]

    Hv = np.asarray([[1, -1]])
    Hh = Hv.T
    D_Fver = abs(conv2(Image, Hv, 'same'))
    D_Fhor = abs(conv2(Image, Hh, 'same'))
    D_Fver = D_Fver[1:m-1, 1:n-1]
    D_Fhor = D_Fhor[1:m-1, 1:n-1]

    D_Bver = abs(conv2(Bver, Hv, 'same'))
    D_Bhor = abs(conv2(Bhor, Hh, 'same'))
    D_Bver = D_Bver[1:m-1, 1:n-1]
    D_Bhor = D_Bhor[1:m-1, 1:n-1]


    D_Vver = D_Fver-D_Bver
    D_Vver[D_Vver<0] = 0
    D_Vhor = D_Fhor-D_Bhor
    D_Vhor[D_Vhor<0] = 0

    s_Fver = np.sum(D_Fver)
    s_Fhor = np.sum(D_Fhor)
    s_Vver = np.sum(D_Vver)
    s_Vhor = np.sum(D_Vhor)

    b_Fver = (s_Fver - s_Vver)/s_Fver
    b_Fhor = (s_Fhor - s_Vhor)/s_Fhor

    IDM = max(b_Fver, b_Fhor)

    return IDM

def MLVMap(im):

    if len(im.shape)==3:
        im = rgb2gray(im)

    xs, ys = im.shape
    x=im

    x1=np.zeros((xs,ys))
    x2=np.zeros((xs,ys))
    x3=np.zeros((xs,ys))
    x4=np.zeros((xs,ys))
    x5=np.zeros((xs,ys))
    x6=np.zeros((xs,ys))
    x7=np.zeros((xs,ys))
    x8=np.zeros((xs,ys))
    x9=np.zeros((xs,ys))

    x1[1:xs-2,1:ys-2] = x[2:xs-1,2:ys-1]
    x2[1:xs-2,2:ys-1] = x[2:xs-1,2:ys-1]
    x3[1:xs-2,3:ys]   = x[2:xs-1,2:ys-1]
    x4[2:xs-1,1:ys-2] = x[2:xs-1,2:ys-1]
    x5[2:xs-1,2:ys-1] = x[2:xs-1,2:ys-1]
    x6[2:xs-1,3:ys]   = x[2:xs-1,2:ys-1]
    x7[3:xs,1:ys-2]   = x[2:xs-1,2:ys-1]
    x8[3:xs,2:ys-1]   = x[2:xs-1,2:ys-1]
    x9[3:xs,3:ys]     = x[2:xs-1,2:ys-1]

    x1=x1[2:xs-1,2:ys-1]
    x2=x2[2:xs-1,2:ys-1]
    x3=x3[2:xs-1,2:ys-1]
    x4=x4[2:xs-1,2:ys-1]
    x5=x5[2:xs-1,2:ys-1]
    x6=x6[2:xs-1,2:ys-1]
    x7=x7[2:xs-1,2:ys-1]
    x8=x8[2:xs-1,2:ys-1]
    x9=x9[2:xs-1,2:ys-1]

    d1=x1-x5
    d2=x2-x5
    d3=x3-x5
    d4=x4-x5
    d5=x6-x5
    d6=x7-x5
    d7=x8-x5
    d8=x9-x5

    dd=np.maximum(d1,d2)
    dd=np.maximum(dd,d3)
    dd=np.maximum(dd,d4)
    dd=np.maximum(dd,d5)
    dd=np.maximum(dd,d6)
    dd=np.maximum(dd,d7)
    dd=np.maximum(dd,d8)

    return dd

def MLVSharpnessMeasure(im):
    T=1000;
    alpha=-0.01

    im_map = MLVMap(im)
    xs, ys = im_map.shape

    xy_number=xs*ys
    l_number=int(xy_number)
    vec = np.reshape(im_map,(xy_number))
    vec=sorted(vec.tolist(),reverse = True)
    svec=np.array(vec[1:l_number])

    a=range(1,xy_number)
    q=np.exp(np.dot(alpha,a))
    svec=svec*q
    svec=svec[1:T]
    sigma = np.sqrt(np.mean(np.power(svec,2)))

    return sigma
