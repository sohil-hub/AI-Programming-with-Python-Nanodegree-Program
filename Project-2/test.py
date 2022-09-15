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
from utils_p2 import setup_network, load_checkpoint, process_image, predict
device='cpu'
model = load_checkpoint('checkpoint_vgg16_2.pth', device)

# model = load_checkpoint(args.checkpoint, device)
with open('/Users/mohammedsohilshaikh/Desktop/workspace/Project-2/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
probabilities = predict('image_06734.jpg', model, 5, device)
probability = np.array(probabilities[0][0].to('cpu'))
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].to('cpu'))]

i = 0
while i < 5:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1
print("Finished Predicting!")