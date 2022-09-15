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
from utils_p2 import setup_network

parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('--data_dir', action="store", default="flowers")
parser.add_argument('--save_dir', action="store", default="checkpoint_vgg16_1.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")
args = parser.parse_args()
print(args)
print('setting device')

if args.gpu == 'gpu':
    if getattr(torch,'has_mps',False):
        device = 'mps'
    elif torch.cuda.is_available():
        device = "cuda"
else:
    device = "cpu"

print("device set to ", device)


train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def setup_network(structure='vgg16',dropout=0.1,hidden_units=512, lr=0.001, device='cpu'):
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


model, criterion, optimizer = setup_network(args.arch, args.dropout, args.hidden_units, args.learning_rate, device)
print(model)


epochs = args.epochs
print_every = 5
steps = 0
loss_show = []

for e in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    
                
                    log_ps = model.forward(inputs)
                    batch_loss = criterion(log_ps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Loss: {running_loss/print_every:.3f}.. "
                  f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                  f"Accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()



# TODO: Save the checkpoint
if args.arch == 'vgg16':
  model = models.vgg16(pretrained=True)
  n_input_features = model.classifier[0].in_features
elif args.arch == 'densenet121':
  model = models.densenet121(pretrained=True)
  n_input_features = model.classifier.in_features

model.class_to_idx = train_data.class_to_idx
torch.save({'input_size': n_input_features,
            'output_size': 102,
            'structure': args.arch,
            'learning_rate': args.learning_rate,
            'dropout': args.dropout,
            'hidden_units':args.hidden_units,
            'classifier': model.classifier,
            'epochs': args.epochs,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}, args.save_dir)