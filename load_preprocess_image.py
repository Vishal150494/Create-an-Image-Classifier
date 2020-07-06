import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
from PIL import Image

import load_preprocess_image
import function_and_classes_model
import os

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Defining the transforms for training, validation and testing sets
def image_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
    
    return train_transforms, validation_transforms, train_transforms 

#To Load the datasets with the Image Folder
def load_datasets(train_dir, train_transforms, validation_transforms, test_transforms):
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    return train_data, validation_data, test_data

#import json
def imp_json(json_file):
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

#Processing a PIL image for using it in our PyTorch model
def process_image(image_path):
    pil_image = Image.open(image_path)
    
    #Resize using thumbnail
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000,256))
    else:
        pil_image.thumbnail((256,5000))
    
    #Cropping
    left_margin = (pil_image.width - 224)/2
    right_margin = left_margin + 224
    bottom_margin = (pil_image.height - 224) / 2
    top_margin = bottom_margin + 224
    
    pil_image= pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    np_image = np.array(pil_image)/255 # Divided by 255 because imshow() expects integers 

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2,0,1))

    return np_image

#converting PyTorch tensor and displaying it
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#Displaying the image along with top 5 classes
def image_display (image_path, cat_to_name, classes):
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    # Set up title
    flower_num = image_path.split('/')[2]
    flower_name = cat_to_name[flower_num]

    # Plot flower
    image = process_image(image_path)
    imshow(image, ax, title = flower_name);

    #Converting the class integer encoding to real flower names 
    flower_names = [cat_to_name[i] for i in classes] 

    # Plot bar chart
    plt.subplot(2,1,2)
    sb.barplot(x=probability, y=flower_names, color=sb.color_palette()[0]);
    plt.show()
    