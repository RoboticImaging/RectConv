import glob
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loaders.base_loader import BaseLoader

class WoodscapeLoader(BaseLoader):
    def __init__(self, data_path, cam, transform):
        super(WoodscapeLoader, self).__init__(data_path)
        self.img_list = glob.glob(data_path+'/rgb_images/*'+cam+'.png')
        self.label_list = glob.glob(data_path+'/semantic_annotations/gtLabels/*'+cam+'.png')
        self.transform = transform

# Load the data in a dataloader
def load(data_path, batch_size, cam, transform=None):
    dataset = WoodscapeLoader(data_path, cam, transform)
    trainset, testset = dataset.get_splits(n_test=0.1)
    train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    test_dl = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=5)
    return train_dl, test_dl