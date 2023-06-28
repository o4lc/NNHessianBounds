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

from Utilities.Plotter import plotReachability



torch.set_printoptions(precision=8)
warnings.filterwarnings("ignore")



def calculateDirectionsOfOptimization(onlyPcaDirections, imageData, label_data = None):
    data_mean = 0
    inputData = None
    if onlyPcaDirections:
        if True:
            pca = PCA()
            pcaData = pca.fit_transform(imageData)
        else:
            pca = FastICA()
            pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_


        inputData = torch.from_numpy(data_comp @ (imageData.cpu().numpy() - data_mean).T).T.float()
        # print(np.linalg.norm(data_comp, 2, 1))

        pcaDirections = []
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

    elif label_data == None:
        pcaDirections = []
        numDirections = 30

        data_comp = np.array(
            [[np.cos(i * np.pi / numDirections), np.sin(i * np.pi / numDirections)] for i in range(numDirections)])
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

    else:
        pcaDirections = []
        numDirections = 9  

        data_comp = np.zeros((numDirections, numDirections + 1))
        data_comp[:, label_data] = 1
        for i in range(len(data_comp)):
            if i < label_data:
                data_comp[i, i] = -1
            else:
                data_comp[i, i+1] = -1
        for direction in data_comp:
            pcaDirections.append(direction / np.linalg.norm(direction))
    return pcaDirections, data_comp, data_mean, inputData


def calculateDirectionsOfHigherDimProjections(currentPcaDirections, imageData):
    indexToStartReadingBoundsForPlotting = len(currentPcaDirections)
    projectedImageData = imageData.clone()
    projectedImageData[:, 2:] = 0
    pca2 = PCA()
    _ = pca2.fit_transform(projectedImageData)
    plottingDirections = pca2.components_
    for direction in plottingDirections[:2]:
        currentPcaDirections.append(-direction)
        currentPcaDirections.append(direction)
    return indexToStartReadingBoundsForPlotting


def solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, network,
                                plottingConstants, calculatedLowerBoundsforpcaDirections,
                                originalNetwork, horizonForLipschitz, lowerCoordinate, upperCoordinate,
                                boundingMethod, splittingMethod):
    eps = config['eps']
    verbose = config['verbose']
    verboseEssential = config['verboseEssential']
    scoreFunction = config['scoreFunction']
    virtualBranching = config['virtualBranching']
    numberOfVirtualBranches = config['numberOfVirtualBranches']
    maxSearchDepthLipschitzBound = config['maxSearchDepthLipschitzBound']
    normToUseLipschitz = config['normToUseLipschitz']
    useTwoNormDilation = config['useTwoNormDilation']
    useSdpForLipschitzCalculation = config['useSdpForLipschitzCalculation']
    lipschitzSdpSolverVerbose = config['lipschitzSdpSolverVerbose']
    initialGD = config['initialGD']
    nodeBranchingFactor = config['nodeBranchingFactor']
    branchNodeNum = config['branchNodeNum']
    pgdIterNum = config['pgdIterNum']
    pgdNumberOfInitializations = config['pgdNumberOfInitializations']
    pgdStepSize = config['pgdStepSize']
    spaceOutThreshold = config['spaceOutThreshold']
    dim = network.Linear[0].weight.shape[1]
    totalNumberOfBranches = 0

    timers = Timers(["lowerBound",
                        "lowerBound:lipschitzForwardPass", "lowerBound:lipschitzCalc",
                        "lowerBound:lipschitzSearch",
                        "lowerBound:virtualBranchPreparation", "lowerBound:virtualBranchMin",
                        "upperBound",
                        "bestBound",
                        "branch", "branch:prune", "branch:maxFind", "branch:nodeCreation",
                        "LipSDP",
                        ])


    for i in range(len(pcaDirections)):
        previousLipschitzCalculations = []
        if i % 2 == 1 and torch.allclose(pcaDirections[i], -pcaDirections[i - 1]):
            previousLipschitzCalculations = BB.lowerBoundClass.calculatedLipschitzConstants
        c = pcaDirections[i]
        if True:
            print('** Solving Horizon: ', iteration, 'dimension: ', i)
        initialBub = torch.min(imageData @ c)
        # initialBub = None
        BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, verboseEssential=verboseEssential,
                            inputDimension=dim,
                            eps=eps, network=network, queryCoefficient=c, currDim=i, device=device,
                            nodeBranchingFactor=nodeBranchingFactor, branchNodeNum=branchNodeNum,
                            scoreFunction=scoreFunction,
                            pgdIterNum=pgdIterNum, pgdNumberOfInitializations=pgdNumberOfInitializations, pgdStepSize=pgdStepSize,
                            virtualBranching=virtualBranching,
                            numberOfVirtualBranches=numberOfVirtualBranches,
                            maxSearchDepthLipschitzBound=maxSearchDepthLipschitzBound,
                            normToUseLipschitz=normToUseLipschitz, useTwoNormDilation=useTwoNormDilation,
                            useSdpForLipschitzCalculation=useSdpForLipschitzCalculation,
                            lipschitzSdpSolverVerbose=lipschitzSdpSolverVerbose,
                            initialGD=initialGD,
                            previousLipschitzCalculations=previousLipschitzCalculations,
                            originalNetwork=originalNetwork,
                            horizonForLipschitz=horizonForLipschitz,
                            initialBub=initialBub,
                            spaceOutThreshold=spaceOutThreshold,
                            boundingMethod=boundingMethod,
                            splittingMethod=splittingMethod,
                            timers=timers
                            )
        lowerBound, upperBound, space_left = BB.run()
        plottingConstants[i] = -lowerBound
        calculatedLowerBoundsforpcaDirections[i] = lowerBound
        totalNumberOfBranches += BB.numberOfBranches

        if False:
            print('Best lower/upper bounds are:', lowerBound, '->', upperBound)
    return totalNumberOfBranches, timers.timers


def main(Method = None):
    configFolder = "Config/"
    fileName = ["RobotArmS", "DoubleIntegratorS", "quadrotorS", "MnistS" , "ACASXU" ,"test"]
    fileName = fileName[2]

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
        splittingMethod = config['splittingMethod']
    except:
        splittingMethod = 'length'

    if Method == None:
        boundingMethod = config['boundingMethod']
    else:
        boundingMethod = Method
    if config['A'] and not fullLoop:
        A = torch.Tensor(config['A'])
        B = torch.Tensor(config['B'])
        c = torch.Tensor(config['c'])
    else:
        A = B = c = None
    try:
        lowerCoordinate = torch.Tensor(config['lowerCoordinate'])
        upperCoordinate = torch.Tensor(config['upperCoordinate'])
    except:
        trainTransform = transforms.ToTensor()
        trainSet = torchvision.datasets.MNIST('data', train=True, transform=trainTransform, download=True)
    
        X_train = trainSet[0][0].reshape(28*28)
        y_train = trainSet[0][1]
        lowerCoordinate = torch.ones((784, )) / 20000 * -1 + torch.Tensor(X_train)
        upperCoordinate = torch.ones((784, )) / 20000      + torch.Tensor(X_train)
        label_data = y_train


    if not verboseMultiHorizon:
        plotProjectionsOfHigherDims = False

    if finalHorizon > 1 and performMultiStepSingleHorizon and\
            (normToUseLipschitz != 2 or not useSdpForLipschitzCalculation):
        raise ValueError

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    if False:
        print(device)
        print(' ')
    
    lowerCoordinate = lowerCoordinate.to(device)
    upperCoordinate = upperCoordinate.to(device)

    network = NeuralNetwork(pathToStateDictionary, A, B, c, activation=activation, loadOrGenerate=True)

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

        if "robotarm" not in configFileToLoad.lower() and plotInitandHorizon:
            plt.scatter(inputPlotData[:, 0], inputPlotData[:, 1], marker='.', label='Initial', alpha=0.5)
    plottingData[0] = {"exactSet": inputData}

    
    startTime = time.time()
    totalNumberOfBranches = 0
    totalLipSDPTime = 0
    for iteration in tqdm(range(finalHorizon)):
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
        pcaDirections, data_comp, data_mean, inputData = calculateDirectionsOfOptimization(onlyPcaDirections, imageData,
                                                                                           label_data if 'MnistS' in fileName else None)
        if verboseMultiHorizon and plotInitandHorizon:
            plt.scatter(imageData[:, 0], imageData[:, 1], marker='.', label='Horizon ' + str(iteration + 1), alpha=0.5)


        numberOfInitialDirections = len(pcaDirections)
        indexToStartReadingBoundsForPlotting = 0
        plottingDirections = pcaDirections
        if plotProjectionsOfHigherDims:
            indexToStartReadingBoundsForPlotting = calculateDirectionsOfHigherDimProjections(pcaDirections, imageData) 

        plottingData[iteration + 1]["A"] = pcaDirections
        plottingConstants = np.zeros((len(pcaDirections), 1))
        plottingData[iteration + 1]['d'] = plottingConstants
        pcaDirections = torch.Tensor(np.array(pcaDirections))
        calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))


        t1, timers = solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, networkZonotope,
                                    plottingConstants, calculatedLowerBoundsforpcaDirections,
                                    originalNetworkZonotope, horizonForLipschitz, lowerCoordinate, upperCoordinate, boundingMethod, splittingMethod)
        
        totalNumberOfBranches += t1
        totalLipSDPTime += timers['LipSDP'].totalTime

        if finalHorizon > 1:
            rotation = nn.Linear(dim, dim)
            rotation.weight = torch.nn.parameter.Parameter(torch.linalg.inv(torch.from_numpy(data_comp).float().to(device)))
            rotation.bias = torch.nn.parameter.Parameter(torch.from_numpy(data_mean).float().to(device))
            # print(rotation.weight, '\n', rotation.weight.T @ rotation.weight)
 
            networkZonotope.rotation = rotation

            centers = []
            for i, component in enumerate(data_comp):
                u = -calculatedLowerBoundsforpcaDirections[2 * i]
                l = calculatedLowerBoundsforpcaDirections[2 * i + 1]
                # center = (u + l) / 2
                center = component @ data_mean
                centers.append(center)
                upperCoordinate[i] = u - center
                lowerCoordinate[i] = l - center
            
            if initialZonotope:
                upperCoordinate = upperCoordinate[:dim]
                lowerCoordinate = lowerCoordinate[:dim]


        if verboseMultiHorizon:
            plotReachability(configFileToLoad, pcaDirections, indexToStartReadingBoundsForPlotting, 
                                calculatedLowerBoundsforpcaDirections, Method, finalIter = (iteration == (finalHorizon - 1)))

    
    endTime = time.time()

    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps, ', LipSDP time (s):', totalLipSDPTime)
    print("Total number of branches: {}".format(totalNumberOfBranches))
    torch.save(plottingData, "Output/reachCurv" + fileName)
    plt.savefig("Output/reachCurv" + fileName + '.png')
    return endTime - startTime, totalNumberOfBranches, totalLipSDPTime, splittingMethod


if __name__ == '__main__':
    for Method in ['secondOrder']:
        runTimes = []
        numberOfBrancehs = []
        lipSDPTimes = []
        for i in range(1):
            t1, t2, t3, splittingMethod = main(Method)
            runTimes.append(t1)
            numberOfBrancehs.append(t2)
            lipSDPTimes.append(t3)
        print('-----------------------------------')
        print('Average run time: {}, std {}'.format(np.mean(runTimes), np.std(runTimes)))
        print('Average LipSDP time: {}, std {}'.format(np.mean(lipSDPTimes), np.std(lipSDPTimes)))
        print('Average branches: {}, std {}'.format(np.mean(numberOfBrancehs), np.std(numberOfBrancehs)), ', splitting method: ', splittingMethod)
        print(' ')

    if plt.get_fignums():
        plt.show()