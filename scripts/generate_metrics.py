# --------------------------------------------------
# Author: Ryan Griffiths (r.griffiths@sydney.edu.au)
# Function to run a test on woodscape dataset
# --------------------------------------------------

from collections import OrderedDict
import torch
from tqdm import tqdm
from statistics import mean
import numpy as np
import torch.nn.functional as F
import time
from PIL import Image
from torchvision import transforms

import scripts.util as util


def generate_metrics(model, dataloader, rectify=None, mask=None, num_classes: int=10):
    """ Generate results woodscape dataset

    Args:
        model (torch module): loaded torch network
        dataloader: woodscape dataloader
        rectify: mapping for rectification, only if rectifying
        mask (str): path to mask for areas to exclude from segmentation
        num_classes (int): number of classes being classified

    Returns:
        metrics (OrderedDict): dict of error metrics
    
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    acc = []
    iou = []
    inference_time = []

    if mask:
        mask = load_mask('woodscape/'+mask+'_mask.png')

    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (image, seg) in tqdm(enumerate(dataloader), total = len(dataloader)):
            image = image.to(device)
            seg = convert_city2woodscape(seg, 'gt').squeeze()

            if rectify:
                image = remap_img(image, rectify[0])

            tic = time.time()
            pre_seg_test = model(image)

            if isinstance(pre_seg_test, OrderedDict):
                pre_seg_test = pre_seg_test['out']

            inference_time.append(time.time()-tic)
            segmap = pre_seg_test.squeeze().cpu().argmax(0) + 1
            segmap = convert_city2woodscape(segmap, 'pred')

            if rectify:
                segmap = remap_img(segmap.unsqueeze(0).unsqueeze(0), rectify[1]).squeeze()

            if mask is not None:
                segmap = mask_output(segmap, mask)

            acc.append(util.metric_accuracy(segmap, seg))
            iou.append(util.metric_iou(segmap, seg, num_classes))

    iou_np = np.array(iou)
    iou_c = []
    for i in range(num_classes):
        if i == 2 or i ==3:
            continue
        iou_c.append(iou_np[~np.isnan(iou_np)[:, i], i].mean())

    metrics = OrderedDict()
    metrics['acc'] = mean(acc)
    metrics['iou'] = iou_c
    metrics['miou'] = mean(iou_c)
    metrics['inference'] = inference_time
    return metrics


# Rectify image using grip map
def remap_img(img, grid_map):
    grid_map = F.interpolate(grid_map.unsqueeze(0).permute(0, 3, 1, 2), size=(img.shape[-2], img.shape[-1])).permute(0, 2, 3, 1)
    remaped = F.grid_sample(img.float(), grid_map.to(img.device).float(), 
                                       mode='nearest', padding_mode='zeros', align_corners=False)
    return remaped


# Load the mask image to use from a path
def load_mask(path):
    label_image = Image.open(path).convert('L')
    to_tensor = transforms.PILToTensor()
    mask_tensor = (to_tensor(label_image)/255).squeeze()
    mask_tensor[mask_tensor != 1] = 0
    return mask_tensor


# Apply masking of segmentation map
def mask_output(output, mask):
    output[mask.bool()] = 0
    return output


# Kinda hacky way to convert cityscape classes to woodscape classes
def convert_city2woodscape(pred, seg_type='pred'):
    if seg_type == 'pred':
        pred[pred == 1] = 1
        pred[pred == 2] = 0
        pred[pred == 3] = 0
        pred[pred == 4] = 0
        pred[pred == 5] = 0
        pred[pred == 6] = 0
        pred[pred == 7] = 0
        pred[pred == 9] = 0
        pred[pred == 8] = 9
        pred[pred == 10] = 0
        pred[pred == 11] = 0
        pred[pred == 12] = 4
        pred[pred == 13] = 5
        pred[pred == 14] = 6
        pred[pred == 15] = 6
        pred[pred == 16] = 6
        pred[pred == 17] = 6
        pred[pred == 18] = 8
        pred[pred == 19] = 7
    elif seg_type == 'gt':
        pred[pred == 2] = 1
        pred[pred == 3] = 0
    return pred
