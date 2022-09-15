# Imports here
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
from utils_p2 import load_checkpoint, predict


parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('input', default='image_06743.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="flowers")
parser.add_argument('checkpoint', default='checkpoint_vgg16_1.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()

if args.gpu == 'gpu':
    if getattr(torch,'has_mps',False):
        device = 'mps'
    elif torch.cuda.is_available():
        device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device set to ", device)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


def main():
    model = load_checkpoint(args.checkpoint, device)
    
    probabilities = predict(args.input, model, args.top_k, device)
    probability = np.array(probabilities[0][0].to('cpu'))
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].to('cpu'))]
    
    i = 0
    while i < args.top_k:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Finished Predicting!")

main()
  