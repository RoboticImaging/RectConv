import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loaders.base_loader import BaseLoader
from torchvision.io import read_image
import torch
import pandas as pd
import numpy as np


class PIROPOLoader(BaseLoader):
    def __init__(self, data_path, transform):
        super(PIROPOLoader, self).__init__(data_path)
        img_list1 = glob.glob(data_path+'omni_1A/omni1A_test1/*.jpg')
        img_list2 = glob.glob(data_path+'omni_1A/omni1A_test2/*.jpg')
        img_list3 = glob.glob(data_path+'omni_1A/omni1A_test3/*.jpg')
        img_list4 = glob.glob(data_path+'omni_1A/omni1A_test4/*.jpg')
        img_list1.sort()
        img_list2.sort()
        img_list3.sort()
        img_list4.sort()
        self.img_list = img_list1 + img_list2 + img_list3 + img_list4
        self.list_lengths = np.cumsum([len(img_list1), len(img_list2), len(img_list3), len(img_list4)])
        self.label_list = glob.glob(data_path+'omni_1A/Ground_Truth_Annotations/groundTruth_omni1A_test*.csv')
        self.label_list.sort()
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        image = read_image(filename).float()/255
        if 'test1' in filename:
            points = torch.Tensor(pd.read_csv(self.label_list[0],header=None).values)[idx, 1:]
        elif 'test2' in filename:
            points = torch.Tensor(pd.read_csv(self.label_list[1],header=None).values)[idx-self.list_lengths[0], 1:]
        elif 'test3' in filename:
            points = torch.Tensor(pd.read_csv(self.label_list[2],header=None).values)[idx-self.list_lengths[1], 1:]
        elif 'test4' in filename:
            points = torch.Tensor(pd.read_csv(self.label_list[3],header=None).values)[idx-self.list_lengths[2], 1:]
        if self.transform:
            image = self.transform(image)

        return image, points

# Load the data in a dataloader
def load(data_path, batch_size, transform=None):
    dataset = PIROPOLoader(data_path, transform)
    trainset, testset = dataset.get_splits(n_test=0.1)
    train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    test_dl = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    return train_dl, test_dl, testset