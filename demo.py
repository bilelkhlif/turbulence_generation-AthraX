'''
Demo code for imaging through turbulence simulation

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

from simulator import Simulator
from turbStats import tilt_mat, corr_mat, get_r0
import cv2
import numpy as np
import torch

# Select device.
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


'''
The corr_mat function is used to generate spatial-temporal correlation matrix 
for point spread functions. It may take over 10 minutes to finish. However, 
for each correlation value, it only needs to be computed once and can be 
used for all D/r0 values. You can also download the pre-generated correlation
matrix from our website. 
https://engineering.purdue.edu/ChanGroup/project_turbulence.html
'''

# Uncomment the following line to generate correlation matrix
# corr_mat(-0.1,'./data/')

# Load image, permute axis if color (headless — uses opencv, no GUI required)
img_bgr = cv2.imread('./images/color.png')
if img_bgr is None:
    raise FileNotFoundError("Could not read ./images/color.png")
# Convert BGR→RGB and normalise to [0, 1] float32
x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

if len(x.shape) == 3: 
    x = x.transpose((2,0,1))
x = torch.tensor(x, device = device, dtype=torch.float32)

D = 0.1
L = 3000
# r0 can also be calculated from Cn2 and L using the get_r0 function: 
# r0 = get_r0(Cn2, L)
r0 = 0.05

im_size = x.shape[1] # The current version works for square image only

# Generate correlation matrix for tilt. Do this once for each different turbulence parameter. 
tilt_mat(im_size, D, r0, L)

# Simulate. 
simulator = Simulator(D/r0, im_size).to(device, dtype=torch.float32)

# Note that the current version does not support batched images. Only one frame at a time. 
out = simulator(x).detach().cpu().numpy()

if len(out.shape) == 3: 
    out = out.transpose((1,2,0))

# save image (headless — no plt.show() or display calls)
out_uint8 = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
cv2.imwrite('./images/out.png', cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR))


