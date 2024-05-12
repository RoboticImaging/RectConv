import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loaders.base_loader import BaseLoader
import torchvision.transforms as standard_transforms
from torchvision.datasets import Cityscapes

transforms = standard_transforms.Compose([
    standard_transforms.ToTensor(),
])

# Load the data in a dataloader
def load(args):
    trainset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transforms, target_transform=transforms)
    testset = Cityscapes(args.data_path, split='test', mode='fine', target_type='semantic', transform=transforms, target_transform=transforms)
    train_dl = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_dl = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    return train_dl, test_dl