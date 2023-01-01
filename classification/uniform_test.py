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

import argparse
import random

from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

from distill_data import *
from utils import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description="This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar10"],
        help="type of dataset",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="distill",
        choices=["distill", "random", "train"],
        help="whether to use distill data, this will take some minutes",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet50",
            "inceptionv3",
            "mobilenetv2_w1",
            "shufflenet_g1_w1",
            "resnet20_cifar10",
            "sqnxt23_w2",
        ],
        help="model to be quantized",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of distilled data"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size of test data"
    )
    parser.add_argument(
        "--percentile", type=float, default=None, help="percentile for quantization"
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="whether to use per channel quantization",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="tile size (Byte) for per tile quantization",
    )
    parser.add_argument(
        "--sensitivity_constraint",
        type=float,
        default=1e-2,
        help="sensitivity constraint for quantization (1 + input)",
    )
    args = parser.parse_args()
    return args


def kl_divergence(P, Q):
    return (P * (P / Q).log()).sum() / P.size(0)  # batch size
    # F.kl_div(Q.log(), P, None, None, 'sum')


def symmetric_kl(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2


def plot_sen(sen, arch, tile_size=None, per_channel=False):
    if per_channel:
        xaxis = "{} channel id".format(arch)
        title = "{} per channel".format(arch, tile_size)
    elif tile_size is not None:
        xaxis = "{} tile id".format(arch)
        title = "{} tile size = {}".format(arch, tile_size)
    else:
        xaxis = "{} layer id".format(arch)
        title = "{}".format(arch)
    trace = []
    for i in range(len(bits_candidate)):
        trace.append(
            go.Scatter(
                y=sen[i], mode="lines + markers", name="{}bit".format(bits_candidate[i])
            )
        )
    data = trace

    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis),
        yaxis=dict(title="sensitivity of quantization", type="log"),
    )
    fig = go.Figure(data, layout)
    if not os.path.exists("workspace/images"):
        os.makedirs("workspace/images")
    if per_channel:
        file_name = "workspace/images/{}_sen_per_channel.png".format(arch)
    elif tile_size is not None:
        file_name = "workspace/images/{}_sen_{}B.png".format(arch, tile_size)
    else:
        file_name = "workspace/images/{}_sen.png".format(arch)
    fig.write_image(file_name)


def random_sample(sen_result, quan_weight, weight_num):
    bit_ = [2, 4, 8]
    random_code = [random.randint(0, 2) for i in range(len(quan_weight))]
    sen = 0
    size = 0
    for i, bit in enumerate(random_code):
        sen += sen_result[bit][i]
    size = sum(
        weight_num[l] * bit_[i] / 8 / 1024 / 1024 for (l, i) in enumerate(random_code)
    )
    return size, sen


class Node:
    def __init__(
        self,
        cost=0,
        profit=0,
        bit=None,
        parent=None,
        left=None,
        middle=None,
        right=None,
        position="middle",
    ):
        self.parent = parent
        self.left = left
        self.middle = middle
        self.right = right
        self.position = position
        self.cost = cost
        self.profit = profit
        self.bit = bit

    def __str__(self):
        return "cost: {:.2f} profit: {:.2f}".format(self.cost, self.profit)

    def __repr__(self):
        return self.__str__()


def get_FrontierFrontier(sen_result, layer_num, weight_num, constraint=1000):
    bits = [2, 4, 8]
    cost = [2, 4, 8]
    prifits = []
    for line in sen_result:
        prifits.append([-i for i in line])
    root = Node(cost=0, profit=0, parent=None)
    current_list = [root]
    for layer_id in range(layer_num):
        # 1. split
        next_list = []
        for n in current_list:
            n.left = Node(
                n.cost + cost[0] * weight_num[layer_id] / 8 / 1024 / 1024,
                n.profit + prifits[0][layer_id],
                bit=bits[0],
                parent=n,
                position="left",
            )
            n.middle = Node(
                n.cost + cost[1] * weight_num[layer_id] / 8 / 1024 / 1024,
                n.profit + prifits[1][layer_id],
                bit=bits[1],
                parent=n,
                position="middle",
            )
            n.right = Node(
                n.cost + cost[2] * weight_num[layer_id] / 8 / 1024 / 1024,
                n.profit + prifits[2][layer_id],
                bit=bits[2],
                parent=n,
                position="right",
            )
            next_list.extend([n.left, n.middle, n.right])
        # 2. sort
        next_list.sort(key=lambda x: x.cost, reverse=False)
        # 3. prune
        pruned_list = []
        for node in next_list:
            if (
                len(pruned_list) == 0 or pruned_list[-1].profit < node.profit
            ) and node.cost <= constraint:
                pruned_list.append(node)
            else:
                node.parent.__dict__[node.position] = None
        # 4. loop
        current_list = pruned_list
    return current_list


def sensitivity_anylysis(
    quan_act, quan_weight, dataloader, quantized_model, args, weight_num
):
    # 1. get the ground truth output
    for l in quan_act:
        l.full_precision_flag = True
    for l in quan_weight:
        l.full_precision_flag = True
    inputs = None
    gt_output = None
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            if isinstance(inputs, list):
                inputs = inputs[0]
            inputs = inputs.cuda()
            gt_output = quantized_model(inputs)
            gt_output = F.softmax(gt_output, dim=1)
            break
    # 2. change bitwidth layer by layer and get the sensitivity
    if args.per_channel:
        assert args.tile_size == None
        total_channels = 0
        conv_layers = []
        for l in quan_weight:
            if type(l) == Quant_Conv2d:
                total_channels += l.out_channels
                conv_layers.append(l)
        sen_result = [
            [0 for i in range(total_channels)] for k in range(len(bits_candidate))
        ]
        channel_start_id = 0
        with tqdm(total=total_channels) as pbar:
            for l in conv_layers:
                for out_channel_idx in range(l.out_channels):
                    bit_array = torch.full((l.out_channels,), 32, device="cuda")
                    for k, bit in enumerate(bits_candidate):
                        l.full_precision_flag = False
                        bit_array[out_channel_idx] = bit
                        l.bit = bit_array
                        with torch.no_grad():
                            tmp_output = quantized_model(inputs)
                            tmp_output = F.softmax(tmp_output, dim=1)
                            l2_loss = F.mse_loss(tmp_output, gt_output)
                        sen_result[k][
                            channel_start_id + out_channel_idx
                        ] = l2_loss.item()
                        l.full_precision_flag = True
                    pbar.update(1)
                channel_start_id += l.out_channels
        plot_sen(sen_result, args.model, per_channel=args.per_channel)
    elif args.tile_size != None:
        assert args.per_channel == False
        total_tiles = 0
        conv_layers = []
        for l in quan_weight:
            if type(l) == Quant_Conv2d:
                total_tiles += l.tile_nums
                conv_layers.append(l)
        sen_result = [
            [0 for i in range(total_tiles)] for k in range(len(bits_candidate))
        ]
        tile_start_id = 0
        with tqdm(total=total_tiles) as pbar:
            for l in conv_layers:
                for tile_offset in range(l.tile_nums):
                    bit_array = torch.full((l.tile_nums,), 32, device="cuda")
                    for k, bit in enumerate(bits_candidate):
                        l.full_precision_flag = False
                        bit_array[tile_offset] = bit
                        l.bit = bit_array
                        with torch.no_grad():
                            tmp_output = quantized_model(inputs)
                            tmp_output = F.softmax(tmp_output, dim=1)
                            l2_loss = F.mse_loss(tmp_output, gt_output)
                        sen_result[k][tile_start_id + tile_offset] = l2_loss.item()
                        l.full_precision_flag = True
                    pbar.update(1)
                tile_start_id += l.tile_nums
        plot_sen(sen_result, args.model, tile_size=args.tile_size)
    else:
        sen_result = [
            [0 for i in range(len(quan_weight))] for j in range(len(bits_candidate))
        ]
        for i in range(len(quan_weight)):
            for j, bit in enumerate(bits_candidate):
                quan_weight[i].full_precision_flag = False
                quan_weight[i].bit = bit
                with torch.no_grad():
                    tmp_output = quantized_model(inputs)
                    tmp_output = F.softmax(tmp_output, dim=1)
                    l2_loss = F.mse_loss(tmp_output, gt_output)
                sen_result[j][i] = l2_loss.item()
                quan_weight[i].full_precision_flag = True
        plot_sen(sen_result, args.model)
    # 3. Heruistic, minimize the total memory size under sensitivity constraint
    total_sensority_8bit = sum(sen_result[-1])
    sensitivity_constraint = total_sensority_8bit * (1.0 + args.sensitivity_constraint)
    current_sensitivity = total_sensority_8bit
    current_bits = [(len(sen_result) - 1) for i in range(len(sen_result[-1]))]
    # from least sensitive to most sensitive
    # from less bits to more bits
    begin = time.time()
    with tqdm(total=(len(sen_result) * len(sen_result[-1]))) as pbar:
        for i in range(len(bits_candidate)):
            # sort the sensitivity from large to small
            sort_sensority_idx = np.argsort(sen_result[i])[::-1]
            for j in sort_sensority_idx:
                current_select = current_bits[j]
                if (
                    current_sensitivity
                    - sen_result[current_select][j]
                    + sen_result[i][j]
                    < sensitivity_constraint
                ):
                    current_sensitivity -= sen_result[current_select][j]
                    current_sensitivity += sen_result[i][j]
                    current_bits[j] = i
                pbar.update(1)
    node_list = []
    for i in current_bits:
        node_list.append(bits_candidate[i])
    print("Heruistic cost: {:.2f}s".format(time.time() - begin))
    return node_list


def plot_bits(bits, name, tile_size=None, per_channel=False):
    if per_channel:
        xaxis = "channel id"
        title = "{} per channel".format(name, tile_size)
        file_name = "workspace/images/{}_bit_per_channel".format(name)
    elif tile_size is not None:
        xaxis = "tile id"
        title = "{} tile size = {}".format(name, tile_size)
        file_name = "workspace/images/{}_{}B".format(name, tile_size)
    else:
        xaxis = "layer id"
        title = "{}".format(name)
        file_name = "workspace/images/{}_bit".format(name)
    trace = go.Scatter(y=bits, mode="markers+lines")
    layout = go.Layout(
        title=title, xaxis=dict(title=xaxis), yaxis=dict(title="bits of weight")
    )
    data = [trace]
    fig = go.Figure(data, layout)
    fig.write_image(file_name + ".png")
    fig.write_image(file_name + ".pdf")


if __name__ == "__main__":
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print("****** Full precision model loaded ******")

    # Load validation data
    test_loader = getTestData(
        args.dataset,
        batch_size=args.test_batch_size,
        path="./data/imagenet/",
        for_inception=args.model.startswith("inception"),
    )
    # Generate distilled data
    begin = time.time()
    if args.data_source == "distill":
        print("distill data ...")
        dataloader = getDistilData(
            model.cuda(),
            args.dataset,
            batch_size=args.batch_size,
            for_inception=args.model.startswith("inception"),
        )
    elif args.data_source == "random":
        print("Get random data ...")
        dataloader = getRandomData(
            dataset=args.dataset,
            batch_size=args.batch_size,
            for_inception=args.model.startswith("inception"),
        )
    elif args.data_source == "train":
        print("Get train data")
        dataloader = getTrainData(
            args.dataset,
            batch_size=args.batch_size,
            path="./data/imagenet/",
            for_inception=args.model.startswith("inception"),
        )
    print("****** Data loaded ****** cost {:.2f}s".format(time.time() - begin))
    # print("FP model test ...")
    # parallel_model = nn.DataParallel(model).cuda()
    # test(parallel_model, test_loader)
    begin = time.time()
    # Quantize single-precision model to 8-bit model
    quan_tool = QuanModel(
        percentile=args.percentile,
        per_channel=args.per_channel,
        tile_size=args.tile_size,
    )
    quantized_model = quan_tool.quantize_model(model, init_a_bit=32, init_w_bit=32)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    bits_candidate = [2, 3, 4, 5, 6, 7, 8]

    node_list = sensitivity_anylysis(
        quan_tool.quan_act_layers,
        quan_tool.quan_weight_layers,
        dataloader,
        quantized_model,
        args,
        quan_tool.weight_num,
    )

    config = {
        "resnet18": [
            (8, 8),
        ],  # representing MP6 for weights and 6bit for activation
        "resnet50": [(6, 6), (4, 8)],
        "mobilenetv2_w1": [(6, 6), (4, 8)],
        "shufflenet_g1_w1": [(6, 6), (4, 8)],
    }
    for (bit_w, bit_a) in config[args.model]:
        begin = time.time()

        bits = node_list
        plot_bits(
            bits,
            "{}_MP{}A{}".format(args.model, bit_w, bit_a),
            tile_size=args.tile_size,
            per_channel=args.per_channel,
        )

        # set weight bits
        start_idx = 0
        for i, l in enumerate(quan_tool.quan_weight_layers):
            if type(l) != Quant_Conv2d:
                continue
            l.full_precision_flag = False
            if args.per_channel:
                l.bit = bits[start_idx : start_idx + l.out_channels]
                start_idx += l.out_channels
            elif args.tile_size is not None:
                l.bit = bits[start_idx : start_idx + l.tile_nums]
                start_idx += l.tile_nums
            else:
                l.bit = bit_w
        # set activation bits
        for l in quan_tool.quan_act_layers:
            l.full_precision_flag = False
            l.bit = bit_a

        total_weight_bits = 0
        max_act_bits = 0
        # summation total bits
        for i, l in enumerate(quan_tool.quan_weight_layers):
            if type(l) != Quant_Conv2d:
                continue
            if args.per_channel:
                elements_per_out_channel = l.weight[0].numel()
                for i in range(l.out_channels):
                    total_weight_bits += l.bit[i] * elements_per_out_channel
            elif args.tile_size is not None:
                elements_per_tile = l.elements_per_tile
                for i in range(l.tile_nums):
                    total_weight_bits += l.bit[i] * elements_per_tile
            else:
                total_weight_bits += l.bit * quan_tool.weight_num[i]
        for i, l in enumerate(quan_tool.quan_act_layers):
            max_act_bits = max(max_act_bits, l.bit * l.output_nums)

        # setup for inference
        for i, l in enumerate(quan_tool.quan_weight_layers):
            if type(l) != Quant_Conv2d:
                continue
            if args.per_channel:
                l.bit = torch.tensor(l.bit, dtype=torch.float32, device="cuda")
            elif args.tile_size is not None:
                l.bit = torch.tensor(l.bit, dtype=torch.float32, device="cuda")
            else:
                l.bit = l.bit
        for i, l in enumerate(quan_tool.quan_act_layers):
            max_act_bits = max(max_act_bits, l.bit * l.output_nums)
        # Update activation range according to distilled data
        unfreeze_model(quantized_model)
        update(quantized_model, dataloader)
        print(
            "****** Zero Shot Quantization Finished ****** cost {:.2f}s".format(
                time.time() - begin
            )
        )

        # Freeze activation range during test
        freeze_model(quantized_model)
        # parallel_quantized_model = nn.DataParallel(quantized_model).cuda()

        # Test the final quantized model
        print(
            "size: W(tot):{:.2f} KiB A(max):{:.2F} KiB Wmp{}A{}".format(
                total_weight_bits // 8 // 1024, max_act_bits // 8 // 1024, bit_w, bit_a
            )
        )
        # test(parallel_quantized_model, test_loader)
        test(quantized_model, test_loader)
