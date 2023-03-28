import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import Config
import transforms as T
from torchvision.transforms import functional as F
import pandas as pd
import cv2

class CustomDataset(Dataset):
    def __init__(self, root, json_file, transform=None):
        self.root = root
        with open(json_file) as f:
            self.data = json.load(f)
        self.transform = transform
        self.image_ids = [img["id"] for img in self.data["images"]]
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # Get image ID
        img_id = self.image_ids[idx]
        img = next(image for image in self.data["images"] if image["id"] == img_id)
        
        img_path = os.path.join(self.root, "Images")
        mask_path = os.path.join(self.root, "Masks")
        
        # Load image
        image = Image.open(os.path.join(img_path, img['file_name'])).convert("L")
        mask = np.array(Image.open(os.path.join(mask_path, img['file_name'])).convert("L"))
        
        
        # extract annotations from the json file
        annotations = [ann for ann in self.data["annotations"] if ann["image_id"] == img_id]
        
        # extract labels from annotations
        labels = [ann["label"] for ann in annotations]
        # convert labels to integers
        labels = [label for label in labels]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # extract boxes and convert them to format [x1, y1, x2, y2]
        boxes = [ann["bbox"] for ann in annotations]
        
        num_objects = len(boxes)
        
        # convert bboxes to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # calculate the area of the bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # convert id to tensor
        image_id = torch.tensor([idx])

        # convert the masks to a tensor
        masks = torch.tensor(mask, dtype=torch.float32)
        
        # create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
                    
        # apply the transform if any
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target
    
    def __len__(self):
        return len(self.imgs)