# *
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
# *

import math
import numpy as np
from torch.autograd import Function, Variable
import torch


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def linear_quantize_channel(input, scale, zero_point, k, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        zero_point = zero_point.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        raise NotImplementedError(
            "inplace quantization is not supported for tile quantization"
        )
        input_tile.mul_(scale).sub_(zero_point).round_()
        return input_tile
    result = torch.round(scale * input - zero_point)
    n = 2 ** (k - 1)
    if type(n) == torch.Tensor:
        n = n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    result = torch.clamp(result, -n, n - 1)
    return result


def linear_dequantize_channel(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        zero_point = zero_point.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def linear_quantize_tile(input, scale, zero_point, k, elements_per_tile, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    input_flatten = input.view(input.shape[0], -1)
    padding_elements = elements_per_tile - (input_flatten.shape[1] % elements_per_tile)

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        input_padded = input_flatten
        if padding_elements != elements_per_tile:
            input_padded = torch.nn.functional.pad(
                input_flatten, (0, padding_elements), "constant", 0
            )
        input_padded = input_padded.view(input_padded.shape[0], -1, elements_per_tile)
        input_tile = input_padded
        scale = scale.unsqueeze(-1)
        zero_point = zero_point.unsqueeze(-1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        input_tile = input
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        raise NotImplementedError(
            "inplace quantization is not supported for tile quantization"
        )
        input_tile.mul_(scale).sub_(zero_point).round_()
        return input_tile
    result_tile = torch.round(scale * input_tile - zero_point)
    n = 2 ** (k - 1)
    if type(n) == torch.Tensor:
        n = n.unsqueeze(-1)
    result_tile = torch.clamp(result_tile, -n, n - 1)
    # reshape back to original shape
    if len(input.shape) == 4:
        result_tile = result_tile.view(result_tile.shape[0], -1)
        if padding_elements != elements_per_tile:
            result_tile = result_tile[:, :-padding_elements]
        result_tile = result_tile.view(input.shape)
    return result_tile


def linear_dequantize_tile(input, scale, zero_point, elements_per_tile, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    input_flatten = input.view(input.shape[0], -1)
    padding_elements = elements_per_tile - (input_flatten.shape[1] % elements_per_tile)

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        input_padded = input_flatten
        if padding_elements != elements_per_tile:
            input_padded = torch.nn.functional.pad(
                input_flatten, (0, padding_elements), "constant", 0
            )
        input_padded = input_padded.view(input_padded.shape[0], -1, elements_per_tile)
        input_tile = input_padded
        scale = scale.unsqueeze(-1)
        zero_point = zero_point.unsqueeze(-1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        input_tile = input
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        raise NotImplementedError(
            "inplace quantization is not supported for tile quantization"
        )
        input_tile.add_(zero_point).div_(scale)
        return input_tile
    result_tile = (input_tile + zero_point) / scale
    # reshape back to original shape
    if len(input.shape) == 4:
        result_tile = result_tile.view(result_tile.shape[0], -1)
        if padding_elements != elements_per_tile:
            result_tile = result_tile[:, :-padding_elements]
        result_tile = result_tile.view(input.shape)
    return result_tile


def asymmetric_linear_quantization_params(
    num_bits, saturation_min, saturation_max, integral_zero_point=True, signed=True
):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """

    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        if (
            x_min is None
            or x_max is None
            or (sum(x_min == x_max) == 1 and x_min.numel() == 1)
        ):
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2 ** (k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AsymmetricQuantPerChannelFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """

    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # compute scale and zeropoint per input channel
        # feature: NCHW, weight: OIHW
        if (
            x_min is None
            or x_max is None
            or (torch.equal(x_min, x_max) and x_min.numel() == 1)
        ):
            x_channel_flatten = x.view(x.shape[0] - 1)
            x_min, x_max = (
                x_channel_flatten.min(dim=1).value,
                x_channel_flatten.max(dim=1).value,
            )
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
        new_quant_x = linear_quantize_channel(x, scale, zero_point, k, inplace=False)
        # n = 2 ** (k - 1)
        # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_channel(
            new_quant_x, scale, zero_point, inplace=False
        )
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AsymmetricQuantPerTileFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """

    @staticmethod
    def forward(
        ctx, x, k, x_min=None, x_max=None, tile_size=None, elements_per_tile=None
    ):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        tile_size: tile size for quantization (bytes)
        """

        # compute scale and zeropoint per tile per ifm/ofm/weight
        # feature: NCHW, weight: OIHW
        assert tile_size is not None
        x_flatten = x.view(x.shape[0], -1)
        if elements_per_tile is not None:
            _elements_per_tile = elements_per_tile
        elif type(k) == torch.Tensor:
            _elements_per_tile = tile_size * 8 // torch.max(k).item()
        else:
            _elements_per_tile = tile_size * 8 // k
        padding_elements = _elements_per_tile - (
            x_flatten.shape[1] % _elements_per_tile
        )
        if (
            x_min is None
            or x_max is None
            or (torch.equal(x_min, x_max) and x_min.numel() == 1)
        ):
            x_padded = x_flatten
            if padding_elements != _elements_per_tile:
                x_padded = torch.nn.functional.pad(
                    x_flatten, (0, padding_elements), "constant", 0
                )
            x_padded = x_padded.view(x_padded.shape[0], -1, _elements_per_tile)

            x_min, x_max = (
                x_padded.min(dim=2).value,
                x_padded.max(dim=2).value,
            )
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
        new_quant_x = linear_quantize_tile(
            x, scale, zero_point, k, _elements_per_tile, inplace=False
        )
        # n = 2 ** (k - 1)
        # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_tile(
            new_quant_x, scale, zero_point, _elements_per_tile, inplace=False
        )
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
