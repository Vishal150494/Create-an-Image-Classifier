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
import time
import os

import load_preprocess_image
from workspace_utils import active_session

#Function to train and validate the model
def train_model(model, optimizer, criterion, arg_epochs, trainloader, validloader, gpu):
    with active_session():

        epochs = arg_epochs
        steps = 0
        print_every = 5
        start_time = time.time()
        model.to(gpu)

        for epoch in range(epochs):
            
            model.train()
            running_loss = 0
            for inputs, labels in iter(trainloader):
        
                steps += 1
                inputs, labels = inputs.to(gpu), labels.to(gpu)
                optimizer.zero_grad()
        
                log_ps = model.forward(inputs)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        for inputs, labels in iter(validloader):
                            inputs, labels = inputs.to(gpu), labels.to(gpu)
                    
                            log_ps = model.forward(inputs)#log probability(log_ps)
                            batch_loss = criterion(log_ps, labels)
                            valid_loss += batch_loss.item()
                    
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    
                    print("Epoch {}/{}..".format(epoch+1, epochs),
                          "Training Loss: {:.3f}".format(running_loss/print_every),
                          "Validation Loss: {:.3f}".format(valid_loss/len(validloader)),
                          "Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    
            
            time_elapsed = time.time() - start_time
            print("Time elapsed: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))    
            
            running_loss = 0
            model.train()
            

#function to test the accuracy of the model
def test_model_accuracy(model, criterion,testloader, gpu):
    model.eval()
    model.to(gpu)
    
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in iter(testloader):
            inputs, labels = inputs.to(gpu), labels.to(gpu)
            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print("Test accuracy: {:.3f}".format(accuracy/len(testloader))) 
        running_loss = 0 
        model.train()
            
        
#Save the checkpoint
def save_checkpoint(model, train_data, arch, epochs, hidden_units):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'hidden_layer': hidden_units,
                  'output_size': 102,
                  'structure': arch,'learning_rate': 0.003,
                  'classifier': model.classifier, 
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
#Function that Loads the checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
            
    #Now we have to freeze some parameters 
    #which we don't backpropogate through
    for param in model.parameters():
        param.requires_grad = False
        
        model.class_to_idx = checkpoint['class_to_idx']
        
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, checkpoint['hidden_layer'])),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),                    
                          ('fc2', nn.Linear(checkpoint['hidden_layer'], 102)),             
                          ('output', nn.LogSoftmax(dim=1))])) 

        
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
       
                
    return model 

#Class pREDICTION
def predict(image_path, model, topk=5, device='cuda'):
    
    image = load_preprocess_image.process_image(image_path)
    
    # TODO: Implement the code to predict the class from an image file
    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)
    
    im = im.unsqueeze_(0)
    model.to(device)
    im.to(device)
    
    with torch.no_grad():    
        output=model.forward(im)

    probabilities = torch.exp(output)
    
    #Probabilities and indices corresponding to the respective classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes
    