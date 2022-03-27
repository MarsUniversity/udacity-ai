#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MIT License Kevin J. Walchko 2022
#

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import json
from PIL import Image
import os
import argparse

from common import *

def handleArgs():
    parser = argparse.ArgumentParser(description='use a neural network')
    parser.add_argument('path', type = str, help = 'path to image')
    parser.add_argument('checkpoint', type = str, help = 'model to use')
    parser.add_argument('--gpu', action='store_true', help = 'set GPU use')
    parser.add_argument('--top_k', type = int, default = 5, help = 'return the K most likely classes')
    parser.add_argument('--category_names', type = str, default = "", help = 'json file with catagory names')
    return vars(parser.parse_args())

def predict(args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_path = args["path"]
    checkpoint = args["checkpoint"]
    topk = args["top_k"]
    gpu = args["gpu"]
    cat_to_names = args["category_names"]

    model = loadCheckpoint(checkpoint)

    if cat_to_names != "":
        cat_to_name = getCatagories(cat_to_names)
        model.idx_to_class = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
        idx_to_class = model.idx_to_class
    else:
        idx_to_class = model.idx_to_class

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path).type(torch.FloatTensor)

    model.eval()
    # device = torch.device("cuda" if gpu is True else "cpu")
    device = set_gpu(gpu)
    model.to(device)

#     image = torch.from_numpy(np.array([image])).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        logps = model(image)

        # Top k classifications
        ps = torch.exp(logps)
        topPer, topClass = ps.topk(topk, dim=1)

    c = np.array(topClass).squeeze()
    topclsnames = [idx_to_class[x] for x in c]

    return image, np.array(topPer).squeeze(), np.array(topClass).squeeze(), topclsnames

if __name__ == "__main__":
    # ./predict.py flowers/test/59/image_05052.jpg vgg19-2-works-checkpoint.pth
    args = handleArgs()
    # print(args)

    img, prob, classIdx, clsNames = predict(args)
    print(prob)
    print(clsNames)
