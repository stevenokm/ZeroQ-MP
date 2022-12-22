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
    args = parser.parse_args()
    return args


def kl_divergence(P, Q):
    return (P * (P / Q).log()).sum() / P.size(0)  # batch size
    # F.kl_div(Q.log(), P, None, None, 'sum')


def symmetric_kl(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2


def plot_sen(sen, arch):
    trace0 = go.Scatter(y=sen[0], mode="lines + markers", name="2bit")
    trace1 = go.Scatter(y=sen[1], mode="lines + markers", name="4bit")
    trace2 = go.Scatter(y=sen[2], mode="lines + markers", name="8bit")
    data = [trace0, trace1, trace2]

    layout = go.Layout(
        title="{}".format(arch),
        xaxis=dict(
            title="{} layer id".format(arch),
        ),
        yaxis=dict(title="sensitivity of quantization", type="log"),
    )
    fig = go.Figure(data, layout)
    if not os.path.exists("workspace/images"):
        os.makedirs("workspace/images")
    fig.write_image("workspace/images/{}_sen.png".format(arch))


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
    sen_result = [[0 for i in range(len(quan_weight))] for j in range(3)]
    for i in range(len(quan_weight)):
        for j, bit in enumerate([2, 4, 8]):
            quan_weight[i].full_precision_flag = False
            quan_weight[i].bit = bit
            with torch.no_grad():
                tmp_output = quantized_model(inputs)
                tmp_output = F.softmax(tmp_output, dim=1)
                kl_div = symmetric_kl(tmp_output, gt_output)
            sen_result[j][i] = kl_div.item()
            quan_weight[i].full_precision_flag = True
    plot_sen(sen_result, args.model)
    # 3. Pareto Frontier
    ## random
    sizes = []
    sens = []
    for i in range(1000):
        size, sen = random_sample(sen_result, quan_weight, weight_num)
        sizes.append(size)
        sens.append(sen)
    trace_random = go.Scatter(x=sizes, y=sens, mode="markers", name="random")
    layout = go.Layout(
        title="{}".format(args.model),
        xaxis=dict(
            title="{} size (MB)".format(args.model),
        ),
        yaxis=dict(title="sensitivity", type="log"),
    )
    begin = time.time()
    ## DP
    node_list = get_FrontierFrontier(sen_result, len(quan_weight), weight_num)
    print("dp cost: {:.2f}s".format(time.time() - begin))
    sizes = [x.cost for x in node_list]
    sens = [-x.profit for x in node_list]
    trace = go.Scatter(
        x=sizes,
        y=sens,
        mode="markers+lines",
        name="Frontier Frontier",
        marker={"size": 3},
    )
    data = [trace, trace_random]
    fig = go.Figure(data, layout)
    fig.write_image("workspace/images/{}_Pareto.png".format(args.model))
    fig.write_image("workspace/images/{}_Pareto.pdf".format(args.model))
    return node_list


def plot_bits(bits, name):
    trace = go.Scatter(y=bits, mode="markers+lines")
    layout = go.Layout(
        title=name, xaxis=dict(title="size (MB)"), yaxis=dict(title="bits of weight")
    )
    data = [trace]
    fig = go.Figure(data, layout)
    fig.write_image("workspace/images/{}_bit.png".format(name))
    fig.write_image("workspace/images/{}_bit.pdf".format(name))


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
    # test(model, test_loader)
    begin = time.time()
    # Quantize single-precision model to 8-bit model
    quan_tool = QuanModel(
        percentile=args.percentile,
        per_channel=args.per_channel,
        tile_size=args.tile_size,
    )
    quantized_model = quan_tool.quantize_model(model)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()
    config = {
        "resnet18": [
            (8, 8),
            (6, 6),
            (5, 5),
            (4, 4),
            (3, 3),
            (2, 2),
        ],  # representing MP6 for weights and 6bit for activation
        "resnet50": [(6, 6), (4, 8)],
        "mobilenetv2_w1": [(6, 6), (4, 8)],
        "shufflenet_g1_w1": [(6, 6), (4, 8)],
    }
    for (bit_w, bit_a) in config[args.model]:
        begin = time.time()
        for l in quan_tool.quan_act_layers:
            l.full_precision_flag = False
            l.bit = bit_a
        for i, l in enumerate(quan_tool.quan_weight_layers):
            l.full_precision_flag = False
            l.bit = bit_w
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
        parallel_quantized_model = nn.DataParallel(quantized_model).cuda()

        # Test the final quantized model
        print("W{}A{}".format(bit_w, bit_a))
        test(parallel_quantized_model, test_loader)
