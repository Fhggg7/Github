# from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# from models.conv import Net
# from models.rnn_conv import ImageRNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datetime

print("done importing")