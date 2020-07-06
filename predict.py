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
import argparse
import os

import load_preprocess_image
import function_and_classes_model

# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description = 'Image Classifier Prediction')

#https://docs.python.org/3/library/argparse.html---REFERENCE
#Argument 1: --gpu with default value 'cuda'(our gpu)
#Argument 2: --image_path with default value 'flowers/test/1/image_06764.jpg'
#Argument 3: --checkpoint, default as checkpoint.pth
#Argument 4: --topk, top k classes and probabilities
#Argument 5: --json with default as cat_to_name.json
parser.add_argument('--gpu', type=str, default='cuda', help='Can be GPU if available or can be CPU')
parser.add_argument('--image_path', type=str, default='flowers/test/1/image_06764.jpg', help='The path of our image')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='tHE PATH TO THE checkpoint')
parser.add_argument('--topk', type=int, default=5, help='tHE TOP K probabilities and classes') 
parser.add_argument('--json', type=str, default='cat_to_name.json', help='json_file')                    
                    
#Assigning variable 'in_args' to parse_args()
in_args = parser.parse_args()  

#defining device: either cuda or cpu
if in_args.gpu == 'cuda':
    device = 'cuda'
else:
    device = 'cpu'
                    
# Load in a mapping from category label to category name
class_to_name_dict = load_preprocess_image.imp_json(in_args.json)

# Loading the checkpoint
model = function_and_classes_model.load_checkpoint(in_args.checkpoint)
print(model)  
checkpoint = torch.load(in_args.checkpoint)

# Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
image = load_preprocess_image.process_image(in_args.image_path)

# Display image
load_preprocess_image.imshow(image)

load_preprocess_image.image_display(in_args.image_path, cat_to_name, classes)

#Top k probabilities and the classes(converted idx_to_class)
probabilities, classes = function_and_classes_model.predict(in_args.image_path, model, in_args.topk, in_args.gpu)  

print(probabilities)
print(classes)                   
