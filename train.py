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
parser = argparse.ArgumentParser(description = 'Train the Image Classifier')

#https://docs.python.org/3/library/argparse.html---REFERENCE
#Argument 1: --arch with default value 'vgg' is the architecture use here
#argument 2: --hidden units with default value 5000 is the number of hidden layers in the network.(Try with 10000 as well)
#Argument 3: --learning_rate with default value 0.01 is our learn rate for the network
#Argument 4: --epochs with default value 20 is the number of Epoch
#Argument 5: --gpu with default value 'cuda'(our gpu)
#Argument 6: --save_dir with default value 'checkpoint.pth'(from the first part where we saved our model
parser.add_argument('--arch', type=str, default='vgg', help='CNN Model Architecture')
parser.add_argument('--hidden_units', type=int, default=5000, help='Number of neurons in our network')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learn rATE OF OUR MODEL')
parser.add_argument('--epochs', type=int, default=3, help='Epoch')
parser.add_argument('--gpu', type=str, default='cuda', help='Can be GPU if available or can be CPU')                    
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='This is the path to the checkpoint file from first part')

#Assigning variable 'in_args' to parse_args()
in_args = parser.parse_args()                    

#Adding the flowers directory (with train,valid and test folders)
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
                    
#Transforms
train_transforms, validation_transforms, test_transforms = load_preprocess_image.image_transforms()   
                    
#Loading the datasets with ImageFolder
train_data, validation_data, test_data = load_preprocess_image.load_datasets(train_dir, train_transforms, validation_transforms, test_transforms)
                    
#Defining the data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)                    
                    
#Build and train the network
if in_args.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif in_args.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Freeze pretrained model parameters to avoid backpropogating through them
for param in model.parameters():    
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),                    
                          ('fc2', nn.Linear(5000, 102)),             
                          ('output', nn.LogSoftmax(dim=1))])) 

model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)                   
                    
function_and_classes_model.train_model(model, optimizer, criterion, in_args.epochs, trainloader, validloader, in_args.gpu)
function_and_classes_model.test_model_accuracy(model, criterion,testloader, in_args.gpu)
function_and_classes_model.save_checkpoint(model, train_data, in_args.arch, in_args.epochs, in_args.hidden_units)    
                   