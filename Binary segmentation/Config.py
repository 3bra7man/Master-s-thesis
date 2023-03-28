import torch
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

Dataset_dir = 'C:\\Users\\lenovo\\Desktop\\Master Thesis Project\\Dataset'

# define the path to the images and masks dataset
Image_dataset_dir = os.path.join(Dataset_dir, "Images")
Mask_dataset_dir = os.path.join(Dataset_dir, "Mask")

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
Init_LR = 0.001
Num_epochs = 100
Batch_size = 4

# define the input image dimensions
Input_Width = 700
Input_Height = 700

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