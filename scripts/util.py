# --------------------------------------------------
# Author: Ryan Griffiths (r.griffiths@sydney.edu.au)
# Functions required use RectConv Layers
# --------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
import math

from scripts.projection import Camera
from conv.RectifyConv2d import conv2d_to_rectifyconv2d 


# Generate the offset field based on the camera parameters
def generate_offset(cam: Camera, max_kernel_size=7, interp_step=32, scaling=0.9):
    print('Generating Offset Field...')

    offsetmap = torch.zeros((max_kernel_size,max_kernel_size,2, cam.height//interp_step, cam.width//interp_step))

    for v in range(0, offsetmap.shape[-2], 1):
        for u in range(0, offsetmap.shape[-1], 1):
            kernel_points = kernel_grid(u*interp_step, v*interp_step, max_kernel_size)
            world_point = cam.project_2d_to_3d(kernel_points, norm=np.ones(kernel_points.shape[0])) - cam.translation
            patch_worldpoints = get_patch(world_point.reshape(max_kernel_size, max_kernel_size, 3), scaling)

            image_points = cam.project_3d_to_2d(patch_worldpoints + cam.translation)
            image_points_reshape = image_points.reshape(max_kernel_size,max_kernel_size, 2)

            image_points_tensor = torch.rot90(torch.from_numpy(image_points_reshape), k=0)
            
            grid_y, grid_x = torch.meshgrid(torch.arange(0, max_kernel_size)-max_kernel_size//2, 
                                            torch.arange(0, max_kernel_size)-max_kernel_size//2, indexing='ij')

            offsetmap[:,:,0, v, u] = torch.rot90((image_points_tensor[:,:,1] - v*interp_step - grid_y),k=0)
            offsetmap[:,:,1, v, u] = torch.rot90((image_points_tensor[:,:,0] - u*interp_step - grid_x),k=0)
            
    return offsetmap


# Calculated patch grid for a given position and size
def kernel_grid(x,y,kernel_size):
    kernel_min = math.floor(kernel_size/2)
    x_lin = np.linspace(x-kernel_min, x+kernel_min, kernel_size)
    y_lin = np.linspace(y-kernel_min, y+kernel_min, kernel_size)
    xv, yv = np.meshgrid(x_lin, y_lin)
    grid = np.concatenate((xv.reshape((-1, 1)), yv.reshape((-1,1))), axis=1)
    return grid


# Calculate rectified patch
def get_patch(distorted_worldpoints, scaling):

    centre = distorted_worldpoints[distorted_worldpoints.shape[0]//2, distorted_worldpoints.shape[1]//2, :]
    elev_centre, az_centre = cartesian2spherical(centre)
    rot = Rotation.from_euler('xyz', [(elev_centre-np.pi/2)*np.cos(az_centre-np.pi/2), (elev_centre-np.pi/2)*np.sin(az_centre-np.pi/2), -az_centre])
    rotated_worldpoints = distorted_worldpoints @ rot.as_matrix().T

    edge1 = rotated_worldpoints[0, 0, :]
    edge2 = rotated_worldpoints[-1, -1, :]
    edge3 = rotated_worldpoints[0, -1, :]
    edge4 = rotated_worldpoints[-1, 0, :]
    
    elev_edge1, az_edge1 = cartesian2spherical(edge1)
    elev_edge2, az_edge2 = cartesian2spherical(edge2)
    elev_edge3, az_edge3 = cartesian2spherical(edge3)
    elev_edge4, az_edge4 = cartesian2spherical(edge4)

    elev_edge_min = min([elev_edge1,elev_edge2,elev_edge3,elev_edge4])
    az_edge_min = min([az_edge1,az_edge2,az_edge3,az_edge4])
    elev_edge_max = max([elev_edge1,elev_edge2,elev_edge3,elev_edge4])
    az_edge_max = max([az_edge1,az_edge2,az_edge3,az_edge4])

    elev_centre_rot, az_centre_rot = cartesian2spherical(rotated_worldpoints[rotated_worldpoints.shape[0]//2, rotated_worldpoints.shape[1]//2, :])

    angle = ((abs(elev_edge_max-elev_edge_min))+(abs(az_edge_max-az_edge_min)))/2/2 * scaling

    elev_lin = np.linspace(elev_centre_rot-angle, elev_centre_rot+angle, rotated_worldpoints.shape[0])
    az_lin = np.linspace(az_centre_rot-angle, az_centre_rot+angle, rotated_worldpoints.shape[1])

    elev_v, az_v = np.meshgrid(elev_lin, az_lin)
    cart_points = polar2cart(1, np.rot90(elev_v, -1).flatten(), np.rot90(az_v, -1).flatten())

    cart_points_rotatedback = np.moveaxis(cart_points, 0, -1) @ rot.inv().as_matrix().T
    return cart_points_rotatedback


# Convert cartesian coordinates to elevation/azimuth
def cartesian2spherical(xyz):
    xy = xyz[0]**2 + xyz[1]**2
    elevation = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])
    return elevation, azimuth


# Convert polar coordinates to cartesian 
def polar2cart(r, theta, phi):
    return np.array([
         r * np.sin(theta) * np.cos(phi),
         r * np.sin(theta) * np.sin(phi),
         r * np.cos(theta)
    ])


# Calculate IOU metric
def metric_iou(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


# Calculate pixel accuracy metric
def metric_accuracy(pred, target):
    return torch.eq(pred, target).sum().item()/(pred.shape[0]*pred.shape[1])


# Recursive search for conv layers to be converted to rectconv
def convert_to_rectconv(model, distmap):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(model, child_name, conv2d_to_rectifyconv2d(child, distmap, conv_size=1))
        else:
            convert_to_rectconv(child, distmap)
