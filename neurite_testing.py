import os
import numpy as np
import torch
import torch.nn.functional as F
from timm_unet import Unet
from skimage import io
from torchvision.transforms import v2 as transforms


debug_plot = False
use_gpu = False
neurite_threshold = -1  # Threshold for neurite activities
soma_threshold = 0  # Threshold for soma activities
timm_model = "tf_efficientnet_b3.ns_jft_in1k"
HW = 288
in_chans = 1
ckpts = "neurite_ckpts"
load_ckpt = "neurites_and_somas.pth"
f = "../simple_unet/B2_attachments/PID2023_20230901-1-msneuron-chr2-cry2tdp43-KS2_T2_0-0_B2_1_RFP1_0.0_0_1_CROP-464.pngred=neurite green=attachment-points blue=soma gray=original.tif"

data = io.imread(f)
image = data[..., 3]
neurite_label = (data[..., 0] > 0).astype(np.float32)
soma_label = (data[..., 2] > 0).astype(np.float32)

if debug_plot:
    data = data[None]
    from matplotlib import pyplot as plt
    plt.subplot(141);plt.imshow(data[0, ..., 0]);plt.subplot(142);plt.imshow(data[0, ..., 1]);plt.subplot(143);plt.imshow(data[0, ..., 2]);plt.subplot(144);plt.imshow(data[0, ..., 3]);plt.show()

image = torch.from_numpy(image).float()
neurite_label = torch.from_numpy(neurite_label).float()
soma_label = torch.from_numpy(soma_label).float()

# Cropping transform. Images are 300 but crop down to 288
tf = transforms.CenterCrop(HW)

if use_gpu:
    image = image.cuda()
    neurite_label = neurite_label.cuda()
    soma_label = soma_label.cuda()

# Preprocess and add a singleton
image = image[None]
neurite_label = neurite_label[None]
soma_label = soma_label[None]
image = tf(image)
image = image[None]
neurite_label = tf(neurite_label)
soma_label = tf(soma_label)

# Build and load model
model = Unet(
    # in_chans=in_chans,
    backbone=timm_model,
    num_classes=1,  # 256,  # DAPI/TDP43
    decoder_channels=(256, 128, 64, 32, 32),
    heads=2,
    normalize_input=True
)
ckpt_weights = torch.load(os.path.join(ckpts, load_ckpt), map_location="cuda:0")
key_check = [x for x in ckpt_weights.keys()][0]
if key_check.split(".")[0] == "module":
    ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items()}
model.load_state_dict(ckpt_weights)

# Process an image
with torch.no_grad():
    neurite, soma = model(image)
    neurite = neurite.squeeze()
    soma = soma.squeeze()

    n_loss = F.binary_cross_entropy_with_logits(neurite, neurite_label.squeeze())
    s_loss = F.binary_cross_entropy_with_logits(soma, soma_label.squeeze())
    loss = n_loss + s_loss

neurite = neurite > neurite_threshold
soma = soma > soma_threshold

from matplotlib import pyplot as plt
f = plt.figure()
plt.title("Total loss {}, Neurite loss {}, Soma loss {}".format(loss, n_loss, s_loss))
plt.subplot(231);plt.imshow(image.squeeze().cpu());plt.subplot(232);plt.imshow(neurite.squeeze().cpu().detach());plt.subplot(233);plt.imshow(soma.squeeze().cpu().detach());plt.subplot(235);plt.imshow(neurite_label.squeeze().cpu());plt.subplot(236);plt.imshow(soma_label.squeeze().cpu())
plt.show()

