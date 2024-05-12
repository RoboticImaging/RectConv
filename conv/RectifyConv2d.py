# --------------------------------------------------
# Author: Ryan Griffiths (r.griffiths@sydney.edu.au)
# Custom RectConvs layer
# --------------------------------------------------

import torch
from torch import Tensor
from torch.nn.common_types import _size_2_t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Optional


class RectifyConv2d(_ConvNd):
    """ Custom RectifyConv2d layer based on deformable convolutions 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        distortion_map: Tensor = None
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(RectifyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.offsetfield = torch.zeros((kernel_size[0]*kernel_size[1]*2, distortion_map.shape[-2], distortion_map.shape[-1]))
        self.offsetfield[0::2, :, :] = distortion_map[distortion_map.shape[0]//2-kernel_size[0]//2:distortion_map.shape[0]//2+kernel_size[0]//2+1,
                                                      distortion_map.shape[1]//2-kernel_size[0]//2:distortion_map.shape[1]//2+kernel_size[0]//2+1,
                                                      0,:,:].reshape(-1, distortion_map.shape[-2], distortion_map.shape[-1])
        self.offsetfield[1::2, :, :] = distortion_map[distortion_map.shape[0]//2-kernel_size[0]//2:distortion_map.shape[0]//2+kernel_size[0]//2+1,
                                                      distortion_map.shape[1]//2-kernel_size[0]//2:distortion_map.shape[1]//2+kernel_size[0]//2+1,
                                                      1,:,:].reshape(-1, distortion_map.shape[-2], distortion_map.shape[-1])
        self.offsetconfig = False

    def calculate_output_size(self, input_size):
        h = int((input_size[0] + 2 * self.padding[0] - (((self.dilation[0]-1) * 2 + self.kernel_size[0])-1)-1)/self.stride[0] + 1)
        w = int((input_size[1] + 2 * self.padding[1] - (((self.dilation[1]-1) * 2 + self.kernel_size[1])-1)-1)/self.stride[1] + 1)
        return [h, w]

    def configure_offset(self, input_size):
        output_size = self.calculate_output_size(input_size[-2:])
        offsetfield = F.interpolate(self.offsetfield.unsqueeze(0), size=(output_size[0], output_size[1]))
        if (self.dilation[0] != 1):
            offsetfield *= self.dilation[0]
        return offsetfield
    
    def _deformable_conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        offsetfield = self.configure_offset(input.shape)

        if self.padding_mode != 'zeros':
            return torchvision.ops.deform_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            offsetfield.to(input.device).repeat(input.shape[0],1,1,1), weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return torchvision.ops.deform_conv2d(input, offsetfield.to(input.device).repeat(input.shape[0],1,1,1), weight, bias=bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)

    def forward(self, input: Tensor) -> Tensor:
        return self._deformable_conv_forward(input, self.weight, self.bias)


# Convert a Con2d layer to the Rectconv version, if kernel size is large enough
def conv2d_to_rectifyconv2d(conv: nn.Conv2d, distortion_map: Tensor, conv_size=1):
    if conv.bias is not None:
        bias = True
    else:
        bias = False
    
    if conv.kernel_size[0] <= conv_size:
        return conv

    rectconv = RectifyConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, 
                         conv.stride, conv.padding, conv.dilation, conv.groups, 
                         bias, conv.padding_mode, distortion_map)
    rectconv.weight = conv.weight
    if bias:
        rectconv.bias = conv.bias
    return rectconv