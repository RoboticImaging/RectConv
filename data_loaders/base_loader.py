import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import read_image
from skimage.io import imread
import torch

class BaseLoader(Dataset):
    def __init__(self, data_path):
        super(BaseLoader, self).__init__()
        self.data_path = data_path
        self.img_list = []
        self.label_list = []
        self.transform = None

    # number of rows in the dataset
    def __len__(self):
        return len(self.img_list)

    # get a row at an index
    def __getitem__(self, idx):
        image = read_image(self.img_list[idx]).float()/255
        seg = torch.Tensor(imread(self.label_list[idx])).long()

        if self.transform:
            image = self.transform(image)

        return image, seg

    @staticmethod
    def calculate_ray_map(cam=None):
        return torch.empty((2,3), dtype=torch.int64)

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * (self.__len__()))
        train_size = self.__len__() - test_size
        # calculate the split
        return random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(42))