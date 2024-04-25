import os
import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA, FastICA
import copy
import json
from Utilities.Timer import Timers
import warnings
from tqdm import tqdm
import matplotlib.patches as patches
import pdb
from Utilities.Plotter import plotReachability, SOTAPlotCMP
import sys
import argparse
import matplotlib.patches as patches
from matplotlib import animation

torch.set_printoptions(precision=8)
warnings.filterwarnings("ignore")

from run import setArgs, calculateDirectionsOfOptimization, calculateDirectionsOfHigherDimProjections
from run import solveSingleStepReachability, main


t1, t2, t3, splittingMethod = main('secondOrder')
plt.show()

data = torch.load('Output/PlotData/quadrotorS')


plt.figure()
ax = plt.gca()
fig = plt.gcf()
finalHorizon = list(data.keys())[-1]
for i in range(1, finalHorizon):
    pcaDirections = data[i]['A']
    calculatedLowerBoundsforpcaDirections = data[i]['d']
    indexToStartReadingBoundsForPlotting = 12
    
    AA = -np.array(pcaDirections[indexToStartReadingBoundsForPlotting:])
    AA = AA[:, :2]
    bb = []
    for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsforpcaDirections)):
        bb.append(calculatedLowerBoundsforpcaDirections[i])

    bb = np.array(bb)
    pltp = polytope.Polytope(AA, bb)
    ax = pltp.plot(ax, alpha = 0.3, color='grey', edgecolor='black')


x_init = data[0]['exactSet'][0, 0]
y_init = data[0]['exactSet'][0, 1]
patch = patches.Rectangle((x_init, y_init), 0.1, 0.1, fc='b')

def init():
    ax.add_patch(patch)
    return patch,

def animate(i):
    patch.set_xy([data[i]['exactSet'][0, 0], data[i]['exactSet'][0, 1]])
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=finalHorizon,
                               interval=500,
                               blit=True)





plt.xlim(2, 5)
plt.ylim(1.5, 5)

plt.show()
