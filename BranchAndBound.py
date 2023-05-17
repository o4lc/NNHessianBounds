from packages import *
from Utilities.Plotter import Plotter
from BranchAndBoundNode import BB_node
from Bounding.LipschitzBound import LipschitzBounding
from Bounding.PgdUpperBound import PgdUpperBound



class BranchAndBound:
    def __init__(self, coordUp=None, coordLow=None, verbose=False, verboseEssential=False, pgdStepSize=1e-3,
                 inputDimension=2, eps=0.1, network=None, queryCoefficient=None, currDim = 0,
                 pgdIterNum=5, pgdNumberOfInitializations=2, device=torch.device("cuda", 0),
                 maximumBatchSize=256,  nodeBranchingFactor=2, branchNodeNum = 1,
                 scoreFunction='length',
                 virtualBranching=False, numberOfVirtualBranches=4,
                 maxSearchDepthLipschitzBound=10,
                 normToUseLipschitz=2, useTwoNormDilation=False, useSdpForLipschitzCalculation=False,
                 lipschitzSdpSolverVerbose=False, initialGD=False, previousLipschitzCalculations=[],
                 originalNetwork=None,
                 horizonForLipschitz=1,
                 initialBub=None,
                 initialBubPoint=None,
                 spaceOutThreshold=10000,
                 boundingMethod='firstOrder',
                 splittingMethod='length',
                 timers=None
                 ):

        self.lowerBoundClass = LipschitzBounding(network, device, virtualBranching, maxSearchDepthLipschitzBound,
                                                 normToUseLipschitz, useTwoNormDilation, useSdpForLipschitzCalculation,
                                                 numberOfVirtualBranches, lipschitzSdpSolverVerbose,
                                                 previousLipschitzCalculations,
                                                 originalNetwork=originalNetwork,
                                                 horizon=horizonForLipschitz,
                                                 boundingMethod=boundingMethod
                                                 )
        self.queryCoefficient = queryCoefficient
        self.calculateLipschitzBeforeNodeCreation = not (normToUseLipschitz == float("inf")
                                                         or (normToUseLipschitz == 2 and useTwoNormDilation))
        if self.calculateLipschitzBeforeNodeCreation:
            lipschitzConstant =\
                self.lowerBoundClass.calculateLipschitzConstant(self.queryCoefficient, coordLow.unsqueeze(0), coordUp.unsqueeze(0))
            self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow, scoreFunction=scoreFunction,
                                       depth=0, lipschitzConstant=lipschitzConstant)]
            self.initialLipschitz = lipschitzConstant
        else:
            self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow, scoreFunction=scoreFunction)]
        self.bestUpperBound = initialBub
        self.initialBubPoint = initialBubPoint
        self.bestLowerBound = None
        self.initCoordUp = coordUp
        self.initCoordLow = coordLow
        self.verbose = verbose
        self.verboseEssential = verboseEssential
        self.pgdIterNum = pgdIterNum
        self.pgdNumberOfInitializations = pgdNumberOfInitializations
        self.inputDimension = inputDimension
        self.eps = eps
        self.network = network
        self.currDim = currDim


        self.upperBoundClass = PgdUpperBound(network, pgdNumberOfInitializations, pgdIterNum, pgdStepSize,
                                             inputDimension, device, maximumBatchSize)
        self.nodeBranchingFactor = nodeBranchingFactor
        self.scoreFunction = scoreFunction
        self.branchNodeNum = branchNodeNum
        self.device = device
        self.maximumBatchSize = maximumBatchSize
        self.initialGD = initialGD
        self.timers = timers
        self.numberOfBranches = 0
        self.spaceOutThreshold = spaceOutThreshold
        self.lipschitzUpdateDepths = [0]
        self.boundingMethod = boundingMethod
        self.splittingMethod = splittingMethod

    def prune(self):
        for i in range(len(self.spaceNodes) - 1, -1, -1):
            if self.spaceNodes[i].lower > self.bestUpperBound and len(self.spaceNodes) > 1:
                self.spaceNodes.pop(i)


    def lowerBound(self, indices):
        lowerBounds = torch.vstack([self.spaceNodes[index].coordLower for index in indices])
        upperBounds = torch.vstack([self.spaceNodes[index].coordUpper for index in indices])
        lipschitzConstants = None
        if self.calculateLipschitzBeforeNodeCreation:
            lipschitzConstants = torch.hstack([self.spaceNodes[index].lipschitzConstant for index in indices])

        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds, timer=self.timers,
                                               extractedLipschitzConstants=lipschitzConstants)

    def upperBound(self, indices):
        return self.upperBoundClass.upperBound(indices, self.spaceNodes, self.queryCoefficient)
    
    def chooseCoordToSplit(self, node):
        if self.splittingMethod == 'length':
            coordToSplitSorted = torch.argsort(node.coordUpper - node.coordLower)
            coordToSplit = coordToSplitSorted[len(coordToSplitSorted) - 1]

        elif self.splittingMethod == 'BestLB':
            temp_lb =  -1e8 * torch.ones(len(node.coordUpper))
            for coord in range(len(node.coordUpper)):
                parentNodeUpperBound = node.coordUpper
                parentNodeLowerBound = node.coordLower

                newIntervals = torch.linspace(parentNodeLowerBound[coord],
                                                        parentNodeUpperBound[coord],
                                                        self.nodeBranchingFactor + 1)
                temp = []
                with torch.no_grad():
                    for i in range(self.nodeBranchingFactor):
                        tempLow = parentNodeLowerBound.clone()
                        tempHigh = parentNodeUpperBound.clone()

                        tempLow[coord] = newIntervals[i]
                        tempHigh[coord] = newIntervals[i+1]
                        temp.append(self.lowerBoundClass.lowerBound(self.queryCoefficient, tempLow.unsqueeze(0), tempHigh.unsqueeze(0), timer=self.timers,
                                               extractedLipschitzConstants=None))
                    temp = torch.Tensor(temp)
                    temp_lb[coord] = torch.min(temp)
            coordToSplit = torch.argmax(temp_lb)
        return coordToSplit

    def branch(self):
        # Prunning Function
        self.timers.start("branch:prune")
        self.prune()
        self.timers.pause("branch:prune")
        numNodesAfterPrune = len(self.spaceNodes)


        self.timers.start("branch:maxFind")
        scoreArray = torch.Tensor([self.spaceNodes[i].score for i in range(len(self.spaceNodes))])
        scoreArraySorted = torch.argsort(scoreArray)
        if len(self.spaceNodes) > self.branchNodeNum:
            maxIndices = scoreArraySorted[len(scoreArraySorted) - self.branchNodeNum : len(scoreArraySorted)]
        else:
            maxIndices = scoreArraySorted[:]

        deletedUpperBounds = []
        deletedLowerBounds = []
        nodes = []
        maxIndices, __ = torch.sort(maxIndices, descending=True)
        for maxIndex in maxIndices:
            node = self.spaceNodes.pop(maxIndex)
            nodes.append(node)
            for i in range(self.nodeBranchingFactor):
                deletedUpperBounds.append(node.upper)
                deletedLowerBounds.append(node.lower)
        deletedLowerBounds = torch.Tensor(deletedLowerBounds).to(self.device)
        deletedUpperBounds = torch.Tensor(deletedUpperBounds).to(self.device)
        self.timers.pause("branch:maxFind")
        for j in range(len(nodes) - 1, -1, -1):
            self.timers.start("branch:nodeCreation")
            coordToSplit = self.chooseCoordToSplit(nodes[j])
            node = nodes[j]
            parentNodeUpperBound = node.coordUpper
            parentNodeLowerBound = node.coordLower

            newIntervals = torch.linspace(parentNodeLowerBound[coordToSplit],
                                                    parentNodeUpperBound[coordToSplit],
                                                    self.nodeBranchingFactor + 1)
            for i in range(self.nodeBranchingFactor):
                tempLow = parentNodeLowerBound.clone()
                tempHigh = parentNodeUpperBound.clone()

                tempLow[coordToSplit] = newIntervals[i]
                tempHigh[coordToSplit] = newIntervals[i+1]
                lipschitzConstant = node.lipschitzConstant
                depth = node.depth + 1
                if self.calculateLipschitzBeforeNodeCreation and depth in self.lipschitzUpdateDepths:
                    lipschitzConstant =\
                        self.lowerBoundClass.calculateLipschitzConstant(self.queryCoefficient,
                                                                        tempLow.unsqueeze(0), tempHigh.unsqueeze(0))

                self.spaceNodes.append(
                    BB_node(np.infty, -np.infty, tempHigh, tempLow, scoreFunction=self.scoreFunction,
                            depth=depth, lipschitzConstant=lipschitzConstant))

                if torch.any(tempHigh - tempLow < 1e-8):
                    self.spaceNodes[-1].score = -1
            self.timers.pause("branch:nodeCreation")
        
        numNodesAfterBranch = len(self.spaceNodes)
        numNodesAdded = numNodesAfterBranch - numNodesAfterPrune + len(maxIndices)
        self.numberOfBranches += numNodesAdded

        return [len(self.spaceNodes) - j for j in range(1, numNodesAdded + 1)], deletedUpperBounds, deletedLowerBounds

    def bound(self, indices, parent_lb):
        self.timers.start("lowerBound")
        tempLowerBound = self.lowerBound(indices)
        lowerBounds = torch.maximum(tempLowerBound, parent_lb)
        self.timers.pause("lowerBound")
        self.timers.start("upperBound")
        upperBounds = self.upperBound(indices)
        # if self.boundingMethod == "secondOrder":
        #     upperBounds = upperBounds
        self.timers.pause("upperBound")
        for i, index in enumerate(indices):
            self.spaceNodes[index].upper = upperBounds[i]
            self.spaceNodes[index].lower = lowerBounds[i]

    def run(self):
        if self.initialGD:
            pgdStartTime = time.time()
            initUpperBoundClass = PgdUpperBound(self.network, 10 if self.initialBubPoint is None else 1, 100, 0.001,
                                                self.inputDimension, self.device, self.maximumBatchSize)

            pgdUpperBound = torch.Tensor(initUpperBoundClass.upperBoundPerIndexWithPgd(0, self.spaceNodes,
                                                                                       self.queryCoefficient,
                                                                                       self.initialBubPoint))
            if self.bestUpperBound:
                # if pgdUpperBound < self.bestUpperBound:
                if self.verbose:
                    print("Improvement percentage of initial PGD over current Best Upper Bound: {}"
                          .format((pgdUpperBound - self.bestUpperBound) / self.bestUpperBound * 100))
                self.bestUpperBound =\
                    torch.minimum(self.bestUpperBound, pgdUpperBound)
            else:
                self.bestUpperBound = pgdUpperBound

            timeForInitialGd = time.time() - pgdStartTime
            # print("Time used in PGD: {}".format(timeForInitialGd))
            if self.verboseEssential:
                print(self.bestUpperBound)
        elif self.bestUpperBound is None:
            self.bestUpperBound = torch.Tensor([torch.inf]).to(self.device)
        self.bestLowerBound = torch.Tensor([-torch.inf]).to(self.device)

        if self.verbose:
            plotter = Plotter()
        self.bound([0], self.bestLowerBound)
        if self.scoreFunction in ["worstLowerBound", "bestLowerBound", "bestUpperBound", "worstUpperBound",
                                  "averageBounds", "weightedGap"]:
            self.spaceNodes[0].score = self.spaceNodes[0].calc_score()

        while self.bestUpperBound - self.bestLowerBound >= self.eps:
            if len(self.spaceNodes) > self.spaceOutThreshold:
                break
            if self.verboseEssential:
                print(len(self.spaceNodes))
            self.timers.start("branch")
            indices, deletedUb, deletedLb = self.branch()
            self.timers.pause("branch")
            self.bound(indices, deletedLb)

            if self.scoreFunction in ["worstLowerBound", "bestLowerBound", "bestUpperBound", "worstUpperBound",
                                      "averageBounds", "weightedGap"]:
                minimumIndex = len(self.spaceNodes) - self.branchNodeNum * self.nodeBranchingFactor
                if minimumIndex < 0:
                    minimumIndex = 0
                maximumIndex = len(self.spaceNodes)
                for i in range(minimumIndex, maximumIndex):
                    self.spaceNodes[i].score = self.spaceNodes[i].calc_score()

            self.timers.start("bestBound")

            self.bestUpperBound =\
                torch.minimum(self.bestUpperBound,
                              torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))])))
            self.bestLowerBound = torch.min(
                torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            self.timers.pause("bestBound")
            if self.verboseEssential:
                print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound, "diff", self.bestUpperBound - self.bestLowerBound)

            if self.verbose:
                print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                print('----------' * 10)
        if self.verbose:
            print("Number of created nodes: {}".format(self.numberOfBranches))
            plotter.showAnimation(self.spaceNodes, self.currDim)
        self.timers.pauseAll()
        if self.verboseEssential:
            self.timers.print()
            print(self.lowerBoundClass.calculatedLipschitzConstants)
            print("number of calculated lipschitz constants ", len(self.lowerBoundClass.calculatedLipschitzConstants))

        return self.bestLowerBound, self.bestUpperBound, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string


        