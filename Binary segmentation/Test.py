# import the necessary packages
import Config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
import torchvision
from torchvision.io import read_image
import os
import cv2

def plot_samples(image, mask):
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
#     mask = mask[0, :, :]
    ax[0].imshow(image.squeeze() , cmap = 'gray')
    ax[1].imshow(mask.squeeze() , cmap = 'gray')
    ax[0].set_title("Image")
    ax[1].set_title("Ground Truth")
    ax[0].grid(False)
    ax[1].grid(False)
    figure.tight_layout()
    figure

def save_predictions_as_imgs(imagesPath, loader, model, folder, device):
    filenames = []
    for filename in imagesPath:
        filenames.append(os.path.basename(filename))
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, folder+'//'+filenames[idx])
#         torchvision.utils.save_image(y, f"{folder}{idx}.png")
    model.train()

def result_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.squeeze() , cmap = 'gray')
    ax[1].imshow(origMask.squeeze() , cmap = 'gray')
    ax[2].imshow(predMask.squeeze().cpu().numpy() , cmap = 'gray')
    
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Ground Truth")
    ax[2].set_title("Predicted Mask")
    
#   remove the grid lines
    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    
    # set the layout of the figure and display it
    figure.tight_layout()
    figure

def make_predictions(model, imagePath):
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, cast it to float data type, and scale its pixel values
        image = cv2.imread(imagePath,0)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = image.astype("float32") / 255.0 
        
        # make a copy of the image for visualization
        orig = image.copy()
        
        # find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(Config.Mask_dataset_dir, filename)
        
        # load the ground-truth segmentation mask in grayscale mode and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        
        image =  torch.from_numpy(image).float().to(Config.DEVICE)
        predMask = torch.sigmoid(model(image))
        predMask = (predMask > 0.3).float()
    
        # prepare a plot for visualization
        result_plot(orig, gtMask, predMask)
        return(gtMask, predMask)