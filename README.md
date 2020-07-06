# Developing an Image Classifier

In this project, I have trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice I have trained this classifier, then exported it for use in my application. For the first part of the project, I have worked through Jupyter notebook to implement the image classifier with PyTorch (Image Classifier Project.ipynb).

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset 
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

At the end of this project (Image Classifier Project.ipynb), you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. 

### Requirements
- Python 3.6
- Pytorch
- CUDA 

### Steps to run the code
- First download the datasets of images with flowers. You can use the followung link to download tha datasets. 
  (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- Once downloaded, load the datasets using `torchvision` and run the `Image Classifier Project`. This is the part one of this project. 
- The second part includes building the command line application. 
- This project includes two main files `train.py` and `predict.py`.
- To make it look more sophisticated, the project consists of two more files `load_preprocess_image.py` and `function_and_classes_model.py`.
- `load_preprocess_image.py` act towards utility functions like loading data and preprocessing images where as `function_and_classes_model.py`
  is used to define functions and classes related to the model such as train, test, validate, save and load checkpoint and prediction. 
- Run `train.py` to train a new network on the dataset and save the model as a checkpoint. 
- Finally, run `predict.py` to use the trained network to predict the class for an input image.  


# NOTE: 
This repo doesn't contain any datasets of images. You have to download them manually to your current working directory from the link mentioned above
in the README section. The datasets has to be divided into three sub datasets each one for training, validating and testing before you run the python files. 
