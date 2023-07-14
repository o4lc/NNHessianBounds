import os

if 'capsule-8237552' in os.getcwd():
    os.chdir('../../../ReachLipBnB/')

from packages import *
from run import solveSingleStepReachability

import polytope 
from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA, FastICA
import copy
import json
import sys

def main(A, b):
    Method = 'secondOrder'
    configFolder = "Config/"
    fileName = ["RobotArmS", "DoubleIntegratorS", "quadrotorS", "MnistS" , "ACASXU", 'nonLinear', 'RandomNet' ,"test"]
    fileName = fileName[5]
    if fileName == "nonLinear":
        fileName = ["B2", "B4", "B5", 'TORA', 'ACC']
        fileName = fileName[2]
        configFolder += "nonLinear/"


    configFileToLoad = configFolder + fileName + ".json"

    with open(configFileToLoad, 'r') as file:
        config = json.load(file)

    eps = config['eps']
    verboseMultiHorizon = config['verboseMultiHorizon']
    normToUseLipschitz = config['normToUseLipschitz']
    useSdpForLipschitzCalculation = config['useSdpForLipschitzCalculation']
    finalHorizon = config['finalHorizon']
    performMultiStepSingleHorizon = config['performMultiStepSingleHorizon']
    plotProjectionsOfHigherDims = config['plotProjectionsOfHigherDims']
    onlyPcaDirections = config['onlyPcaDirections']
    pathToStateDictionary = config['pathToStateDictionary']
    fullLoop = config['fullLoop']
    try:
        initialZonotope = config['InitialZonotope']
    except:
        initialZonotope = False
    try:
        activation = config['activation']
    except:
        activation = 'relu'
    try:
        isLinear = config['isLinear']
    except:
        isLinear = True
    try:
        splittingMethod = config['splittingMethod']
    except:
        splittingMethod = 'length'

    if Method == None:
        boundingMethod = config['boundingMethod']
    else:
        boundingMethod = Method
    
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    lowerCoordinate = b - np.sum(A, 1)
    upperCoordinate = b + np.sum(A, 1)
    lowerCoordinate = torch.Tensor(lowerCoordinate)
    upperCoordinate = torch.Tensor(upperCoordinate)
    lowerCoordinate = lowerCoordinate.to(device)
    upperCoordinate = upperCoordinate.to(device)


    if finalHorizon > 1 and performMultiStepSingleHorizon and\
            (normToUseLipschitz != 2 or not useSdpForLipschitzCalculation):
        raise ValueError

    isLinear = True
    network = NeuralNetwork(pathToStateDictionary, None, None, None, activation=activation, loadOrGenerate=True, isLinear=isLinear)

    # @TODO: move this
    if initialZonotope and True:
        zonotopeMatrix = torch.Tensor([[1, 1, 1], 
                                        [-1, 0, 1]]) 
        # zonotopeMatrix /=  torch.linalg.norm(zonotopeMatrix, 2, 0, True)
        zonotopeMatrix /= 10

        lowerCoordinate = torch.Tensor([-1, -1, -1])
        upperCoordinate = torch.Tensor([1, 1, 1])
        zCenter = (upperCoordinate + lowerCoordinate) / 2
        xCenter = torch.Tensor([2.5, 0.])
    else:
        zonotopeMatrix = torch.Tensor([[1, 0], [0, 1]])

    horizonForLipschitz = 1
    originalNetworkZonotope = None
    if performMultiStepSingleHorizon:
        horizonForLipschitz = finalHorizon
        network.setRepetition(finalHorizon)
        finalHorizon = 1

    dimZ = lowerCoordinate.shape[0]
    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)

    if dim < 3:
        plotProjectionsOfHigherDims = False

    plottingData = {}

    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(10000, dimZ, device=device) \
                                                        + lowerCoordinate
    if initialZonotope:
        inputPlotData = (inputData - zCenter) @ zonotopeMatrix.T + xCenter
    else:
        inputPlotData = inputData

    if verboseMultiHorizon:
        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
            plotInitandHorizon = False
        else:
            fig, ax = plt.subplots()
            plotInitandHorizon = True

        if "robotarm" not in configFileToLoad.lower() and 'random' not in configFileToLoad.lower()  and plotInitandHorizon:
            plt.scatter(inputPlotData[:, 0], inputPlotData[:, 1], marker='.', label='Initial', alpha=0.5)
    plottingData[0] = {"exactSet": inputData}

    iteration = 0
    inputDataVariable = Variable(inputData, requires_grad=False)
    # @TODO: move this
    if initialZonotope and iteration == 0:
        with torch.no_grad():
            networkZonotope = copy.deepcopy(network)
            originalWeight0 = networkZonotope.Linear[0].weight

            networkZonotope.Linear[0].weight = \
                            torch.nn.parameter.Parameter((originalWeight0 @ zonotopeMatrix).float().to(device))
            networkZonotope.Linear[0].bias += \
                            torch.nn.parameter.Parameter((originalWeight0 @ (xCenter - zonotopeMatrix @ zCenter)).float().to(device))
            
            networkZonotope.c += networkZonotope.A @ xCenter
            networkZonotope.A @= zonotopeMatrix

            originalNetworkZonotope = copy.deepcopy(networkZonotope)

    else:
        if iteration == 0:
            networkZonotope = copy.deepcopy(network)
        else:
            rotation =  networkZonotope.rotation
            networkZonotope = copy.deepcopy(network)
            networkZonotope.rotation = rotation

        originalNetworkZonotope = copy.deepcopy(networkZonotope)

    
    with no_grad():
        imageData = networkZonotope.forward(inputDataVariable)

    plottingData[iteration + 1] = {"exactSet": imageData}
    # pcaDirections, data_comp, data_mean, inputData = calculateDirectionsOfOptimization(onlyPcaDirections, imageData,
    #                                                                                     label_data if 'MnistS' in fileName else None)
    pcaDirections = np.array([[1.], [-1.]])


    plottingData[iteration + 1]["A"] = pcaDirections
    plottingConstants = np.zeros((len(pcaDirections), 1))
    plottingData[iteration + 1]['d'] = plottingConstants
    pcaDirections = torch.Tensor(np.array(pcaDirections))
    # pcaDirections = torch.from_numpy(np.array(pcaDirections))
    calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))


    t1, timers = solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, networkZonotope,
                                plottingConstants, calculatedLowerBoundsforpcaDirections,
                                originalNetworkZonotope, horizonForLipschitz, lowerCoordinate, upperCoordinate, boundingMethod, splittingMethod)

    u = -calculatedLowerBoundsforpcaDirections[0]
    l = calculatedLowerBoundsforpcaDirections[1]

    return [1, -1], [u, l]



if __name__ == "__main__":
    if 'capsule-8237552' in os.getcwd():
        path = './'
    else:
        path = '../#Other Methods/capsule-8237552/code/'
    f = open(path + 'inputZonotope.txt', "r")
    txt = np.array(list(map(float, f.read().splitlines())))
    # f.close()

    dim = int(txt[0])
    A = txt[1:dim**2+1].reshape((dim, dim))
    b = txt[dim**2+1:].reshape((dim, ))
    dir, val = main(A, b)

    f = open(path + "outZonotope.txt", "w")
    for item in val:
        f.write(str(item.numpy())+"\n")
    # f.close()
    # sys.exit('a')