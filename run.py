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

torch.set_printoptions(precision=8)
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)



def setArgs(args, configFile, Method=None):
    configFolder = "Config/"
    if configFile == None:
        args.fileName = ["RobotArmS", "DoubleIntegratorS", "quadrotorS", "MnistS" ,
                     "ACASXU", 'nonLinear', 'RandomNetTanh', 'RandomNetSig' ,"test"]
        args.fileName = args.fileName[5]
        if args.fileName == "nonLinear":
            args.fileName = ["B1", "B2", "B3", "B4", "B5","ACC", "TORA"]
            args.fileName = args.fileName[4]
            configFolder += "nonLinear/"
    else:
        args.fileName = configFile
        if 'B' in configFile or 'TORA' in configFile:
            configFolder += "nonLinear/"


    configFileToLoad = configFolder + args.fileName + ".json"

    with open(configFileToLoad, 'r') as file:
        config = json.load(file)

        args.eps = [config['eps'] if args.eps is None else args.eps][0]
        args.splittingMethod = [config['splittingMethod'] if args.splittingMethod is None else args.splittingMethod][0]
        args.lipMethod = [config['lipMethod'] if args.lipMethod is None else args.lipMethod][0]
        args.verboseMultiHorizon = [config['verboseMultiHorizon'] if args.verboseMultiHorizon is None else args.verboseMultiHorizon][0]
        args.normToUseLipschitz = config['normToUseLipschitz']
        args.useSdpForLipschitzCalculation = config['useSdpForLipschitzCalculation']
        args.finalHorizon = [config['finalHorizon'] if args.finalHorizon is None else args.finalHorizon][0]
        args.performMultiStepSingleHorizon = config['performMultiStepSingleHorizon']
        args.plotProjectionsOfHigherDims = config['plotProjectionsOfHigherDims']
        args.onlyPcaDirections = [config['onlyPcaDirections'] if args.onlyPcaDirections is None else args.onlyPcaDirections][0]
        args.pathToStateDictionary = config['pathToStateDictionary']
        args.fullLoop = config['fullLoop']
        try:
            args.initialZonotope = config['InitialZonotope']
        except:
            args.initialZonotope = False
        try:
            args.activation = config['activation']
        except:
            args.activation = 'tanh'
        try:
            args.isLinear = config['isLinear']
        except:
            args.isLinear = True
        if Method == None:
            args.boundingMethod = config['boundingMethod']
        else:
            args.boundingMethod = Method
        if config['A'] and not args.fullLoop:
            A = torch.Tensor(config['A'])
            B = torch.Tensor(config['B'])
            c = torch.Tensor(config['c'])
        else:
            A = B = c = None
        
        if torch.cuda.is_available():
            args.device = torch.device("cuda", 0)
        else:
            args.device = torch.device("cpu")

        args.device = torch.device("cpu")

        lowerCoordinate = torch.Tensor(config['lowerCoordinate'])
        upperCoordinate = torch.Tensor(config['upperCoordinate'])
        lowerCoordinate = lowerCoordinate.to(args.device)
        upperCoordinate = upperCoordinate.to(args.device)

        if not args.verboseMultiHorizon:
            plotProjectionsOfHigherDims = False

        if args.finalHorizon > 1 and args.performMultiStepSingleHorizon and\
                (args.normToUseLipschitz != 2 or not args.useSdpForLipschitzCalculation):
            raise ValueError

        network = NeuralNetwork(args.pathToStateDictionary, A, B, c, activation=args.activation, loadOrGenerate=True, isLinear=args.isLinear)

    return args, network, lowerCoordinate, upperCoordinate, configFileToLoad, config




def calculateDirectionsOfOptimization(onlyPcaDirections, imageData, label_data = None, network = None):
    data_mean = None
    inputData = None
    pcaDirections = []

    if onlyPcaDirections:
        if True:
            pca = PCA()
            pcaData = pca.fit_transform(imageData)
        else:
            pca = FastICA()
            pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_


        inputData = torch.from_numpy(data_comp @ (imageData.cpu().numpy()).T).T.float()
        # print(np.linalg.norm(data_comp, 2, 1))
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

    elif label_data == None:
        if network.isLinear:
            numDirections = 8
            data_comp = np.array(
                [np.array([np.cos(i * np.pi / numDirections), np.sin(i * np.pi / numDirections)]) for i in range(numDirections)])
        else:
            numDirections = imageData.shape[1]
            data_comp = np.eye(numDirections)
        
        data_mean = np.mean(imageData.numpy(), 0)
        inputData = torch.from_numpy(data_comp @ (imageData.cpu().numpy()).T).T.float()
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

    else:
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


def solveSingleStepReachability(pcaDirections, imageData, config, iteration, device, network, eps,
                                plottingConstants, calculatedLowerBoundsforpcaDirections,
                                originalNetwork, horizonForLipschitz, lowerCoordinate, upperCoordinate,
                                boundingMethod, splittingMethod, lipMethod):
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
        c = pcaDirections[i].float()
        if False:
            print('** Solving Horizon: ', iteration, 'dimension: ', i)
        # pdb.set_trace()
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
                            timers=timers,
                            lipMethod=lipMethod,
                            )
        lowerBound, upperBound, space_left = BB.run()
        plottingConstants[i] = -lowerBound
        calculatedLowerBoundsforpcaDirections[i] = lowerBound
        totalNumberOfBranches += BB.numberOfBranches

        if False:
            print('Best lower/upper bounds are:', lowerBound, '->', upperBound)
    return totalNumberOfBranches, timers.timers


def main(Method = None, args=None):
    if args.config is not None:
        configFile = args.config
        args, network, lowerCoordinate, upperCoordinate, configFileToLoad, configDict = setArgs(args, configFile, Method)
    

    # @TODO: move this
    if args.initialZonotope and True:
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
    if args.performMultiStepSingleHorizon:
        horizonForLipschitz = args.finalHorizon
        network.setRepetition(args.finalHorizon)
        args.finalHorizon = 1

    dimZ = lowerCoordinate.shape[0]
    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(args.device)

    if dim < 3:
        args.plotProjectionsOfHigherDims = False

    plottingData = {}

    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(10000, dimZ, device=args.device) \
                                                        + lowerCoordinate
    if args.initialZonotope:
        inputPlotData = (inputData - zCenter) @ zonotopeMatrix.T + xCenter
    else:
        inputPlotData = inputData

    if args.verboseMultiHorizon:
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

    startTime = time.time()
    totalNumberOfBranches = 0
    totalLipSDPTime = 0
    allPcaDirectionsList = []
    allLBsList = []
    for iteration in tqdm(range(args.finalHorizon)):
        inputDataVariable = Variable(inputData, requires_grad=False)
        # @TODO: move this
        if args.initialZonotope and iteration == 0:
            with torch.no_grad():
                networkZonotope = copy.deepcopy(network)
                originalWeight0 = networkZonotope.Linear[0].weight

                networkZonotope.Linear[0].weight = \
                                torch.nn.parameter.Parameter((originalWeight0 @ zonotopeMatrix).float().to(args.device))
                networkZonotope.Linear[0].bias += \
                                torch.nn.parameter.Parameter((originalWeight0 @ (xCenter - zonotopeMatrix @ zCenter)).float().to(args.device))
                
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
        pcaDirections, data_comp, data_mean, inputData = calculateDirectionsOfOptimization(args.onlyPcaDirections, imageData, None, network=networkZonotope)
                                                                                           
        if args.verboseMultiHorizon and plotInitandHorizon:
            plt.scatter(imageData[:, 0],
                         imageData[:, 1], marker='.', label='Horizon ' + str(iteration + 1), alpha=0.5)


        numberOfInitialDirections = len(pcaDirections)
        indexToStartReadingBoundsForPlotting = 0
        plottingDirections = pcaDirections
        if args.plotProjectionsOfHigherDims and network.isLinear:
            indexToStartReadingBoundsForPlotting = calculateDirectionsOfHigherDimProjections(pcaDirections, imageData) 

        plottingData[iteration + 1]["A"] = pcaDirections
        plottingConstants = np.zeros((len(pcaDirections), 1))
        plottingData[iteration + 1]['d'] = plottingConstants
        pcaDirections = torch.Tensor(np.array(pcaDirections))
        # pcaDirections = torch.from_numpy(np.array(pcaDirections))
        calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))

        t1, timers = solveSingleStepReachability(pcaDirections, imageData, configDict, iteration, args.device, networkZonotope, args.eps,
                                    plottingConstants, calculatedLowerBoundsforpcaDirections,
                                    originalNetworkZonotope, horizonForLipschitz, lowerCoordinate, upperCoordinate, args.boundingMethod, 
                                    args.splittingMethod, args.lipMethod)
        
        totalNumberOfBranches += t1
        totalLipSDPTime += timers['LipSDP'].totalTime

        if args.finalHorizon > 1:
            rotation = nn.Linear(dim, dim)
            rotation.weight = torch.nn.parameter.Parameter(torch.linalg.inv(torch.from_numpy(data_comp).float().to(args.device)))
            # rotation.bias = torch.nn.parameter.Parameter(torch.from_numpy(data_mean).float().to(args.device))
            rotation.bias = torch.nn.parameter.Parameter(torch.zeros(dim).float().to(args.device))
            # print(rotation.weight, '\n', rotation.weight.T @ rotation.weight)
 
            networkZonotope.rotation = rotation


            for i, component in enumerate(data_comp):
                u = -calculatedLowerBoundsforpcaDirections[2 * i]
                l = calculatedLowerBoundsforpcaDirections[2 * i + 1]
                # center = (u + l) / 2
                center = component @ data_mean
                center=0
                upperCoordinate[i] = u - center
                lowerCoordinate[i] = l - center
            
            if args.initialZonotope:
                upperCoordinate = upperCoordinate[:dim]
                lowerCoordinate = lowerCoordinate[:dim]

        if args.verboseMultiHorizon:
            plotReachability(configFileToLoad, pcaDirections, indexToStartReadingBoundsForPlotting, 
                                calculatedLowerBoundsforpcaDirections, Method, finalIter = (iteration == (args.finalHorizon - 1)), 
                                finalHorizon=args.finalHorizon)
            
        allPcaDirectionsList.append(pcaDirections)
        allLBsList.append(calculatedLowerBoundsforpcaDirections)
    
    endTime = time.time()
    if args.fileName in ['B1', 'B2', 'B3', 'B4', 'B5', 'TORA']:
        SOTAPlotCMP(args.fileName, verisig=False, numHorizons=args.finalHorizon)
    print('The algorithm took (s):', endTime - startTime, 'with eps =', args.eps, ', Lipschitz time (s):', totalLipSDPTime)
    print("Total number of branches: {}".format(totalNumberOfBranches))
    torch.save(plottingData, "Output/PlotData/" + args.fileName)
    plt.savefig("Output/" + args.fileName + '.png')
    plt.savefig("Output/" + args.fileName + '.pdf', dpi=600)
    return endTime - startTime, totalNumberOfBranches, totalLipSDPTime, args.splittingMethod, args.lipMethod


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Name of Benchmark')
    parser.add_argument('--eps', type=float, default=None, help='Accuracy of Method')
    parser.add_argument('--verboseMultiHorizon', type=bool, default=None)
    parser.add_argument('--finalHorizon', type=int, default=None, help='Number of Iterations')
    parser.add_argument('--onlyPcaDirections', type=int, default=None)
    parser.add_argument('--lipMethod', type=int, default=None, help='Use LipSDP or LipLT')
    parser.add_argument('--splittingMethod', type=str, default=None, help='Splitting Method')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    args = parser.parse_args()

    Methods = ['secondOrder']
    for Method in Methods:
        runTimes = []
        numberOfBrancehs = []
        lipSDPTimes = []
        for i in range(5):
            set_seed(args.seed + i)
            t1, t2, t3, splittingMethod, lipMethod = main(Method, args)
            runTimes.append(t1)
            numberOfBrancehs.append(t2)
            lipSDPTimes.append(t3)
        if Method == 'firstOrder':
            lipMethod = 1

        print('-----------------------------------')
        print('Average run time: {}, std {}'.format(np.mean(runTimes), np.std(runTimes)))
        print('Average Lipschitz time: {}, std {}'.format(np.mean(lipSDPTimes), np.std(lipSDPTimes)))
        print('Average branches: {}, std {}'.format(np.mean(numberOfBrancehs), np.std(numberOfBrancehs)), ', splitting method: ', splittingMethod,
                ', Lip method: ', ['navie', 'LipSDP', 'LipLT'][lipMethod])
        print(' ')

    if plt.get_fignums():
        plt.show(block=False)
        plt.pause(1)
        plt.close()