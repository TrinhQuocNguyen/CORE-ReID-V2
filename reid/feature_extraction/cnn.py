from __future__ import absolute_import
from collections import OrderedDict


from ..utils import to_torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
from torch.nn import functional as F
import cv2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10

def extract_cnn_feature_map(model, inputs):
    model.eval()
    inputs = to_torch(inputs).cuda()
    x2_ema, x2 = model(inputs, return_featuremaps=True)
    x2_ema = x2_ema.data.cpu()
    
    # use imagenet mean and std
    img_mean = IMAGENET_MEAN
    img_std = IMAGENET_STD
    width = 128
    height = 256
    
    # compute activation maps
    outputs = (x2_ema**2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)       
    for j in range(outputs.size(0)):
        # get image name
        imgs = inputs.cpu()

        # RGB image
        img = imgs[j, ...]
        for t, m, s in zip(img, img_mean, img_std):
            t.mul_(s).add_(m).clamp_(0, 1)
        img_np = np.uint8(np.floor(img.numpy() * 255))
        img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

        # activation map
        am = outputs[j, ...].numpy()
        am = cv2.resize(am, (width, height))
        am = 255 * (am - np.min(am)) / (
            np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

        # overlapped
        overlapped = img_np*0.3 + am*0.7
        overlapped[overlapped > 255] = 255
        overlapped = overlapped.astype(np.uint8)

        # save images in a single figure (add white spacing between images)
        # from left to right: original image, activation map (feature map), overlapped image
        grid_img = 255 * np.ones(
            (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
        )
        grid_img[:, :width, :] = img_np[:, :, ::-1]
        grid_img[:,
                    width + GRID_SPACING:2*width + GRID_SPACING, :] = am
        grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
        
        return grid_img, outputs
    
def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs).cuda()
    if modules is None:
        outputs, _ = model(inputs)
        outputs = outputs.data.cpu()
        return outputs

def extract_cnn_feature_source(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

