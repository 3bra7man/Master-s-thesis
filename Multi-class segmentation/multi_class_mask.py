import torch
import numpy as np
from PIL import Image

def multi_mask(im):
#     im = Image.open(path)

    mask = np.asarray(im)
    mask = mask[:, :, 0].astype(np.uint32) + 256 * mask[:, :, 1].astype(np.uint32) + 65536 * mask[:, :,2].astype(np.uint32)

    mask_background_ind = np.where(mask == 0)
    mask_kwire_ind = np.where(mask == 255)
    mask_tip_ind = np.where(mask == mask.max())

    result = torch.zeros([3, mask.shape[0], mask.shape[1]])

    result[0, mask_background_ind[0], mask_background_ind[1]] = 1
    result[1, mask_kwire_ind[0], mask_kwire_ind[1]] = 1
    result[2, mask_tip_ind[0], mask_tip_ind[1]] = 1
    
    return result