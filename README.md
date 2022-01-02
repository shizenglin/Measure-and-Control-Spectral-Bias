# Paper: On Measuring and Controlling the Spectral Bias of the Deep Image Prior, IJCV, 2022 [[pdf]](https://arxiv.org/pdf/2107.01125.pdf)

<h1> 1 Motivations </h1>
The deep image prior showed that a randomly initialized network with a suitable architecture can be trained to solve inverse imaging problems by simply optimizing it’s parameters to reconstruct a single degraded image. However, it suffers from two practical limitations: 
<br>1) It remains unclear how to control the prior beyond the choice of the network architecture; 
<br>2) Training requires an oracle stopping criterion as during the optimization the performance degrades after reaching an optimum value. 

<h1> 2 Contributions </h1>

<h2> 2.1 Measuring Spectral Bias </h2>
We introduce a frequency-band correspondence measure to characterize the spectral bias of the deep image prior, where low-frequency image signals are learned  faster and better than high-frequency counterparts.

![image](https://github.com/shizenglin/Measure-and-Control-Spectral-Bias/blob/main/img/fbc_noise.png)
<br>Figure 2.1: Spectral measurement of the deep image prior on image denoising. We observe that 1) the network of the deep image prior (Ulyanov et al., 2020) exhibits a spectral bias during optimization, 2) the peak PSNR performance of the deep image prior occurs when the lowest frequencies are matched nearly perfect, while the highest frequencies are less used, as marked by the green vertical lines, and 3) deep image prior performance degrades when high-frequency noise is learned beyond acertain level, which could affect the high-frequency image details.

<h2> 2.2 Controlling Spectral Bias </h2>
Based on our observations, we propose techniques to prevent the eventual performance degradation and accelerate convergence. We introduce a Lipschitz-controlled convolution layer and a Gaussian-controlled upsampling layer as plug-in replacements for layers used in the deep architectures. 

![image](https://github.com/shizenglin/Measure-and-Control-Spectral-Bias/blob/main/img/lipschitz_control.png)
<br>Figure 2.2.1: Lipschitz-controlled spectral bias for image denoising on image 'peppers’. Setting the right Lipschitz constant (λ=2) avoids performance decay while maintaining a high PSNR.

![image](https://github.com/shizenglin/Measure-and-Control-Spectral-Bias/blob/main/img/gaussian_control.png)
<br>Figure 2.2.2: Gaussian-controlled spectral bias for image denoising on image 'peppers’. Varying the Gaussian kernel by σ controls convergence and performance.

<h2> 2.3 Automatic Stopping Criterion </h2>
With the ability to control the spectral bias, we can fix the number of iterations for network optimization without fear of performance degradation. As different tasks have different levels of convergence, however, using a fixed number of iterations still leads to redundant optimization. To improve efficiency, we compute the blurriness and sharpness for an output image and use their ratio as the metric to automatically perform early stopping.

![image](https://github.com/shizenglin/Measure-and-Control-Spectral-Bias/blob/main/img/automatic_stop.png)
<br> Figure 2.3: Automatic stopping criterion evaluated on image denoising. The vertical green line shows the selected iteration by the proposed stopping criterion. We observe the optimization can be stopped earlier, with a minimal performance loss compared to a fixed stop at 10,000 iterations.


<h1> 3 Applications </h1>
We demonstrate the effectiveness of our method on four inverse imaging applications and one image enhancement application: image denoising, JPEG image deblocking,  image inpainting, image super-resolution and image detail enhancement. 

![image](https://github.com/shizenglin/Measure-and-Control-Spectral-Bias/blob/main/img/applications.png)
<br>Figure 3.1: Image denoising. The experiments show that 1) our method no longer suffers from eventual performance degradation during optimization, relieving us from the need for an oracle criterion to stop early, 2) the automatic stopping criterion avoids superfluous computation, and 3) our method also obtains favorable restoration and enhancement results compared to current approaches, across all tasks.

<h1> 4 Code Usage </h1>

<h2> 4.1 Requirements </h2>
  1) CUDA 8.0 and Cudnn 7.5 or higher
<br>2) GPU memory 4GB or higher
<br>3) Python 2.7 or higher 
<br>4) Pytorch 1.5 or higher.

<h2> 4.2 Running </h2>
 1) You can find the used data in the folder 'dataset'.
<br>2) Set the experimental parameters accordingly (refer to the paper).
<br>3) Run ¨python xxxx.py¨ where xxxx denotes the task name, e.g. denoising

<h2> 4.3 Notes </h2>
     1) The value of lambda in Lipschitz normalization should be tuned for each input image. Generally, we found that lambda=1.4 works well when using bilinear upsampling, and lambda=1.8 works well when using our Gaussian upsamling. 
<br>2) By introduing the Gaussian upsampling, we show that upsampling operation affacts the spectral bias and optimization convergence. However, we found that the Gaussian upsampling is practically slower than bilinear upsampling because we implement the Gaussian upsampling by using nn.ConvTranspose2d. Thus, in the provided code we use the bilinear upsampling by default.
<br>3) We compute spectral norm by using torch.svd because we found that the power method is inaccurate and unstable.


<h1> 5 Citation </h1>
Please cite our paper when you use this code.

     @article{ShiIJCV22,
        title={On Measuring and Controlling the Spectral Bias of the Deep Image Prior},
        author={Zenglin Shi and Pascal Mettes and Subhransu Maji and Cees G M Snoek},
        journal={International Journal of Computer Vision},
        year={2022}
     }

