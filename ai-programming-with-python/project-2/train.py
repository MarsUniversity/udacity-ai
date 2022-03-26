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

desc = """train a neural network

Ex: ./train.py flowers --learning_rate 0.01 --hidden_units 512 256 --epochs 20
"""

def handleArgs():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('data_dir', type = str, help = 'image data training, test, and validation folder')
    parser.add_argument('--gpu', action='store_true', help = 'enable GPU use')
    parser.add_argument('--arch', type = str, default = 'vgg',  choices=['resnet', 'vgg'], help = 'select neural network to use')
    parser.add_argument('--hidden_units', type = int, default = (1024,102),  nargs='+', help = 'set hidden layers')
    parser.add_argument('--learning_rate', type = float, default = 0.0003, help = 'set learning rate')
    parser.add_argument('--save_dir', type = str, default = './', help = 'checkpoint save location')
    parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs to train for')
    return vars(parser.parse_args())

# modified from course material
def trainVal(model, trainloader, validloader, optimizer, epochs, device):
    model.to(device);
    model.train()

    criterion = nn.NLLLoss()

    steps = 0
    valLoop = len(validloader)
    trainLoop = len(trainloader)
    train_losses, val_losses, accuracy = [], [], []

    for e in keep_awake(range(1,epochs+1)):
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f">> Epoch: {e}/{epochs} Train: {i}/{trainLoop}     ")

        else:
            val_loss = 0
            acc = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for i, (images, labels) in enumerate(validloader):
                    # Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc += torch.mean(equals.type(torch.FloatTensor))

                    print(f">> Epoch: {e}/{epochs} Val: {i}/{valLoop}    ")

            model.train()

            train_losses.append(running_loss/trainLoop)
            val_losses.append(val_loss/valLoop)
            accuracy.append(acc/valLoop)

            print(f">> Epoch: {e+1}/{epochs}              Train Loss: {train_losses[-1]:.3f}"
                  f" Val Loss: {val_losses[-1]:.3f} Accuracy: {accuracy[-1]:.3f}")

    return train_losses, val_losses, accuracy


if __name__ == "__main__":
    args = handleArgs()
    # print(args)
    # print("wtf", args["hidden_units"])

    learningRate = args["learning_rate"]
    epochs = args["epochs"]
    arch = args["arch"]
    drop = 0.2
    hidden = args["hidden_units"]
    gpu = args["gpu"]
    chkpt = args["save_dir"]+"/checkpoint.pth"
    data = args["data_dir"]

    model = buildNN(arch, hidden, drop)

    opt = optim.Adam
    # opt = optim.SGD
    # learningRate = 0.3 # resent
    if arch == "vgg":
        optimizer = opt(model.classifier.parameters(), lr=learningRate)
    elif arch == "resnet":
        optimizer = opt(model.fc.parameters(), lr=learningRate)
    else:
        raise Exception(f"Invalid model type: {type(model)}")

    device = set_gpu(gpu)
    trainloader, _, validloader = getData(data)

    model = trainVal(model, trainloader, validloader, optimizer, epochs, device)
    saveCheckpoint(model, opt, chkpt)
