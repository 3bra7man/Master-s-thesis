import torch

def compute_class_weight(loader, num_classes):
    # Initialize an array to store the class frequencies
    class_freq = torch.zeros(num_classes)

    # Iterate over the data loader to compute the class frequencies
    for data, labels in loader:
        ones = torch.ones(labels.shape)
        zeros = torch.zeros(labels.shape)
        for label in range(num_classes):
            class_freq[label] += torch.sum(torch.where(labels==label, ones, zeros))

    # Compute the class weights
    class_weights = 1 / class_freq
    class_weights = class_weights / class_weights.sum()
    
    return class_weights