import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict 
import torchvision
import argparse



def setup_network(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='cpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        n_input_features = model.classifier[0].in_features
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        n_input_features = model.classifier.in_features
    
    for para in model.parameters():
        para.requires_grad = False
  

    model.classifier = nn.Sequential(nn.Linear(n_input_features , hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    print(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    model.to(device)

    return model, criterion, optimizer


def load_checkpoint(path = 'checkpoint.pth', device='cpu'):
    if device == 'cpu' or device == 'mps':
      checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
      checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    model, _, _ = setup_network(structure, dropout, hidden_units, lr, device)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image


def predict(image_path, model, topk, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    
    with torch.no_grad():
        if device!='gpu':
            logps = model.forward(img.to('cpu'))
        else:
            logps = model.forward(img.cuda())
        
        
    probability = torch.exp(logps).data
    
    return probability.topk(topk)