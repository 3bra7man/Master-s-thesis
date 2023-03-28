from imutils import paths
import torch
import numpy as np
import cv2
import os

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def get_class_weights(path):
    class_weights = []
    for img in range(len(path)):
        target = cv2.imread(path[img], 0)
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        np.append(samples_weight, class_weights)
        return class_weights
        
Dataset_dir = 'C:\\Users\\z004b1tz\\Desktop\\Master Thesis Project\\Dataset\\Test'

# define the path to the images and masks dataset
Image_dataset_dir = os.path.join(Dataset_dir, "Images")
Mask_dataset_dir = os.path.join(Dataset_dir, "Masks")

# determine the device to be used for training and evaluation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda:0" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
No_channels = 1
No_classes = 2
No_levels = 50

# initialize learning rate, number of epochs to train for, and the
# batch size
Init_LR = 0.0001
Num_epochs = 5
Batch_size = 4

# define the input image dimensions
Input_Width = 800
Input_Height = 800

# define threshold to filter weak predictions
Thresh = 0.5

# define the path to the base output directory
Base_Out = os.path.join(Dataset_dir, "output")

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(Base_Out, "unet_out.pth")
LOSS_PATH = os.path.sep.join([Base_Out, "CELoss.png"])
DICE_PATH = os.path.sep.join([Base_Out, "DiceLoss.png"])
TEST_PATHS = os.path.sep.join([Base_Out, "test_paths.txt"])