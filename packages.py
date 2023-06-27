import numpy as np

import torch
import torchvision
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.autograd.functional import jacobian
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D


import os
import time

import polytope 