# -*- coding: utf-8 -*-
# MIT License Kevin J. Walchko 2022
#
# Most of this code is modified course work
import json
import PIL
import numpy as np
import random
from os import listdir

try:
    from workspace_utils import keep_awake
except ImportError:
    def keep_awake(a):
        return a

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# taken from course material
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

def getData(filepath):
    """
    Given the directory, grabs the training, validation, and testing data.
    """
    data_dir = filepath
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # really, this is used by both test and validation data sets
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    return trainloader, testloader, validloader

def getCatagories(filepath, test_data):
    """
    Given a json file of indexs to class names and pyTorch data, returns
    the decoder ring.
    """
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)

    # torch id to class name
    idx_to_class = {v: cat_to_name[k] for k, v in test_data.class_to_idx.items()}
    return idx_to_class

def set_gpu(val):
    """
    Given True or False, will enable cude (True) or enable cpu (False) and
    return the device object.
    """
    device = torch.device("cuda" if val is True else "cpu")
    return device

def buildNN(nn, hidden, drop=0.2):
    """
    Given the type of NN to build (vgg or resent) and the hidden layer sizes
    ([1000, 300] for 2 hidden layers of size 1000 and 300), this will return
    the NN with a new classifier based on the hidden layers.
    """
    if nn == "vgg":
        model = models.vgg19(pretrained=True)
    elif nn == "resnet":
        model = models.resnet18(pretrained=True)
    else:
        raise Exception(f"Invalid NN type: {nn}")
    # print(type(model))
    print(f"Building a {type(model)} type network")

    # attach label decoders to model
    # model.class_to_idx = class_to_idx
    # model.idx_to_class = idx_to_class

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # create a new classifier and attach to loaded trained model
    # since I am looking at different models, there is some if/else code
    # drop = 0.2
    if isinstance(model, models.vgg.VGG):
    #     model.classifier = Network(25088, 102, [8*1024, 4*1024, 512],drop) # 1.2GB
        # model.classifier = Network(25088, 102, [4*1024, 512],drop) # 410MB
    #     model.classifier = Network(25088, 102, [2*1024, 512],drop) # 210MB
        classifier = classifier = Network(25088, 102, hidden, drop)
        model.classifier = classifier
    elif isinstance(model, models.resnet.ResNet):
        # model.fc = Network(512, 102, [256, 175, 128],drop)
        classifier = Network(512, 102, hidden, drop)
        model.fc = classifier
    else:
        raise Exception(f"Invalid model type: {type(model)}")

    print("New classifier NN:\n", classifier)

    return model

def loadCheckpoint(filepath):
    """
    Given a checkpoint file, this will load it and return the NN model.
    """
    checkpoint = torch.load(filepath)
    m = Network(
            checkpoint['input_size'],
            checkpoint['output_size'],
            checkpoint['hidden_layers'])

    m.load_state_dict(checkpoint['state_dict'])

    mt = checkpoint['modelType']

    if mt == "VGG":
        model = models.vgg19(pretrained=True)
        model.classifier = m
    elif mt == "ResNet":
        model = models.resnet18(pretrained=True)
        model.fc = m
    else:
        raise Exception(f"Invalid model: {mt}")

    print(f"Loading a {type(model)} type network with"
        f" {len(checkpoint['hidden_layers'])} hidden layers")

    # model.class_to_idx = checkpoint['cls']
    model.idx_to_class = checkpoint['idx']

    if "opt" in checkpoint:
        opt = checkpoint['opt']
    else:
        opt = None

    return model

def saveCheckpoint(model, optimizer, filename='checkpoint.pth'):
    """
    Given a trained model, optimizer used to train the model and a filename (
    default it "checkpoint.pth"), this will save the NN and nothing is returned.
    """
    model.to("cpu")
    name = type(model).__name__
    m = model.classifier if name == "VGG" else model.fc
    checkpoint = {
        'modelType': name,
        'input_size': m.hidden_layers[0].in_features,
        'output_size': m.output.out_features,
        'hidden_layers': [each.out_features for each in m.hidden_layers],
        'state_dict': m.state_dict(),
        # 'cls': model.class_to_idx,
        'idx': model.idx_to_class,
    }
    if optimizer:
        checkpoint['opt']: optimizer.state_dict

    torch.save(checkpoint, filename)


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = PIL.Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    image = transform(image)

    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
#     print("numpy",np.min(image),np.max(image))

    ax.imshow(image)

    return ax

def getRandomImage(kind):
    ranClass = f"flowers/{kind}/" + random.choice(listdir(f'flowers/{kind}/'))
    ranImg = ranClass + "/" + random.choice(listdir(f'{ranClass}'))
    print(ranImg)
    return ranImg
