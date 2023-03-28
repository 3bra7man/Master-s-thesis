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

def plot_samples(img, mask):
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
    img = np.transpose(img[0,:,:,:],(1,2,0))
    mask = np.array(mask[0,:,:])
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[0].set_title("Image")
    ax[1].set_title("Ground Truth")
    ax[0].grid(False)
    ax[1].grid(False)
    figure.tight_layout()
    figure

def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
    model.train()

def result_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.squeeze() , cmap = 'gray')
    ax[1].imshow(origMask.squeeze() , cmap = 'gray')
    ax[2].imshow(predMask.squeeze() , cmap = 'gray')
    
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
    # set model to evaluation mode
    model.eval()
    
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, cast it to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = image.astype("float32") / 255.0         
        
        # make a copy of the image for visualization
        orig = image.copy()
        
        # find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(Config.Mask_dataset_dir, filename)
        
        # load the ground-truth segmentation mask in grayscale mode and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (Config.Input_Height, Config.Input_Height))
        
        # create a PyTorch tensor, and flash it to the current device
        image = torch.from_numpy(image).to(Config.DEVICE)
        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image)
        predMask = torch.Softmax(predMask)
        predMask = predMask.cpu().numpy()
        
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > Config.Thresh) * 255
        predMask = predMask.astype(np.uint8)
    
        # prepare a plot for visualization
        result_plot(orig, gtMask, predMask)
        return(gtMask, predMask)