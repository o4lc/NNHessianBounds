from typing import List

import numpy as np
import time
import torch
import torch.nn as nn
from copy import deepcopy
import cvxpy as cp
from scipy.linalg import block_diag
import copy

from Bounding.Utils4Curvature import power_iteration, MatrixNorm
from torch.autograd.functional import jacobian

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class LipschitzBounding:
    def __init__(self,
                 network: nn.Module,
                 device=torch.device("cuda", 0),
                 virtualBranching=False,
                 maxSearchDepth=10,
                 normToUse=2,
                 useTwoNormDilation=False,
                 useSdpForLipschitzCalculation=False,
                 numberOfVirtualBranches=4,
                 sdpSolverVerbose=False,
                 calculatedLipschitzConstants=[],
                 originalNetwork=None,
                 horizon=1, 
                 activation='softplus',
                 boundingMethod='firstOrder',
                 lipMethod=1,
                 coordLow = None,
                 coordUp = None):
        
        self.network = network
        self.device = device
        if originalNetwork:
            self.weights, self.biases = self.extractWeightsFromNetwork(originalNetwork)
        else:
            self.weights, self.biases = self.extractWeightsFromNetwork(self.network)
        self.calculatedLipschitzConstants = calculatedLipschitzConstants
        self.maxSearchDepth = maxSearchDepth
        self.performVirtualBranching = virtualBranching
        self.normToUse = normToUse
        self.useTwoNormDilation = useTwoNormDilation
        self.useSdpForLipschitzCalculation = useSdpForLipschitzCalculation
        self.numberOfVirtualBranches = numberOfVirtualBranches
        if normToUse == 2:
            assert (not(self.useSdpForLipschitzCalculation and self.useTwoNormDilation))
        self.sdpSolverVerbose = sdpSolverVerbose
        self.horizon = horizon
        self.activation = activation
        self.boundingMethod = boundingMethod
        self.calculatedCurvatureConstants = []
        self.lipMethod = lipMethod
        self.coordLow=coordLow
        self.coordUp=coordUp

    # def calculateboundsCROWN(self, 
    #                         inputLowerBound: torch.Tensor,
    #                         inputUpperBound: torch.Tensor,
    #                         queryCoefficient: torch.Tensor):
        
    #     # Creating c^T @ Network
    #     model = copy.deepcopy(self.network)
    #     model.A = (queryCoefficient.unsqueeze(0) @ model.A)
    #     model.B = (queryCoefficient.unsqueeze(0) @ model.B)
    #     model.c = (queryCoefficient @ model.c)

    #     my_input = torch.Tensor((inputLowerBound + inputUpperBound) / 2)
    #     # Wrap the model with auto_LiRPA.
    #     model = BoundedModule(model, my_input)
    #     # Define perturbation. Here we add Linf perturbation to input data.
    #     ptb = PerturbationLpNorm(norm=np.Inf, eps=torch.Tensor(torch.Tensor((-inputLowerBound + inputUpperBound) / 2)).to(self.device))
    #     # Make the input a BoundedTensor with the pre-defined perturbation.
    #     my_input = BoundedTensor(my_input, ptb)
    #     with torch.no_grad():
    #         lb, ub = model.compute_bounds(x=(my_input,), method="CROWN")
    #     return lb.reshape(-1, ), ub.reshape(-1, )                                                       

    def calculateCurvatureConstantGeneral(self,
                                            queryCoefficient: torch.Tensor,
                                            g: float,
                                            h: float,
                                            inputLowerBound: torch.Tensor,
                                            inputUpperBound: torch.Tensor,
                                            timer,
                                            lipMethod=1,
                                            local=True):
        ''' 
        In this section, haveing a closed loop system is meaning-less during calculations of subs-networks
        Knowing that the hessian of the Ax+Bu is the same as Bu ... therefore, we can calculate the curvature
        without having the closed loop system
        We concider BW^L as the final layer weight.
        '''
        calculateFinalLayerLip = False
        if self.network.isLinear == False or lipMethod == 2:
            calculateFinalLayerLip = True

        with torch.no_grad():
            params = list(self.network.Linear.parameters()) 
            numLayers = len(params)//2
            summationTerm = 0

            r = torch.zeros((numLayers, 1)).to(self.device)
            r[0] = torch.linalg.norm(params[0], ord=2)
            for i in range(1, numLayers - (1 * (1 - calculateFinalLayerLip))):
                temp_weight = self.weights[:i+1]
                temp_biases = self.biases[:i+1]
                alpha, beta = self.calculateMinMaxSlopes(temp_weight ,
                                                            inputLowerBound, inputUpperBound,
                                                            self.network.activation, 
                                                            temp_biases)
                
                cc = np.eye((temp_weight[-1]).shape[0])
                AA = None
                BB = None
                if (i == numLayers - 1) and self.network.isLinear:
                    # Note that Lip of whole system is not used for Hessian bounds: calculateFinalLayerLip
                    cc = queryCoefficient.unsqueeze(0).cpu().numpy()
                    BB = self.network.B.cpu().numpy()
                    # A is passed here just so that the Lipschitz is valid for the calculation of ICRA
                    AA = self.network.A.cpu().numpy()

                self.startTime(timer, "LipSDP")
                if lipMethod == 1:
                    # LipSDP
                    if local:
                        r[i] = torch.Tensor([lipSDP(temp_weight, alpha, beta, cc, AA, BB,
                                                verbose=self.sdpSolverVerbose)]).to(self.device)
                    else:
                        r[i] = torch.Tensor([lipSDP(temp_weight, np.zeros((len(alpha), 1)), np.ones((len(alpha), 1)),  cc, AA, BB,
                                                verbose=self.sdpSolverVerbose)]).to(self.device)
                    
                elif lipMethod == 2:
                    # LipLT
                    if local:
                        r[i] = torch.Tensor([self.calculateLocalLipLT(temp_weight, alpha, beta, cc, AA, BB,
                                                                    normToUse=2)[-1]]).to(self.device)
                    else:
                        r[i] = torch.Tensor([self.calculateLocalLipLT(temp_weight, np.zeros((len(alpha), 1)), np.ones((len(alpha), 1)), cc, AA, BB,
                                                                    normToUse=2)[-1]]).to(self.device)
                        # r[i] = torch.Tensor([self.calculateLocalLipschitzConstantSingleBatchNumpy(temp_weight, cc, AA, BB,
                        #                                                         normToUse=2)[-1]]).to(self.device)
                        
                else:
                    # Naive Lip
                    if local:
                        r[i] = torch.Tensor([self.calculateNaiveLip(temp_weight, beta, 
                                                                    cc, AA, BB, normToUse=2)[-1]]).to(self.device)
                    else:
                        r[i] = torch.Tensor([self.calculateNaiveLip(temp_weight, np.ones((len(alpha), 1)), 
                                                                    cc, AA, BB, normToUse=2)[-1]]).to(self.device)

                # print(lipMethod, r[-1])
                self.pauseTime(timer, "LipSDP")

            S = []  
            if lipMethod in [0, 1, 2]:
                # Using simple recursion for S
                for i in range(numLayers - 1):
                    SS = []
                    for j in range(i+1, numLayers): 
                        # print('i,j', i, j)
                        if i == j - 1:
                            if self.network.isLinear and j == numLayers - 1:
                                SS.append(torch.abs(queryCoefficient @ self.network.B  @ params[2*j]).unsqueeze(0))
                            else:
                                SS.append(torch.abs(params[2*j]))
                        else:
                            if self.network.isLinear and j == numLayers - 1:
                                SS.append(g * torch.abs(queryCoefficient @ self.network.B  @ params[2*j]).unsqueeze(0) @ SS[-1])   
                            else:
                                SS.append(g * torch.abs(params[2*j]) @ SS[-1])

                    S.append(torch.max(torch.Tensor(SS[-1][0])))
                    
            else:
                # raise Exception("Not implemented")
                # Using LipLT for S
                for i in range(numLayers - 1):
                    temp_weight = self.weights[i+1:]
                    temp_biases = self.biases[i+1:]
                    # alpha, beta = self.calculateMinMaxSlopes(temp_weight ,torch.zeros((1, temp_weight[0].shape[1])), 
                    #                                             g * torch.ones((1, temp_weight[0].shape[1])), self.network.activation, temp_biases)
                    # A doesn't matter for Hessian bounds, but c^TB is important
                    cc = queryCoefficient.unsqueeze(0).cpu().numpy()
                    AA = None
                    BB = self.network.B.cpu().numpy()
                    self.startTime(timer, "LipSDP")
                    if local:
                        S.append(torch.Tensor([self.calculateLocalLipLT(temp_weight, alpha, 
                                                            beta, cc, AA, BB, normToUse=np.inf)[-1]]).to(self.device))
                    else:
                        S.append(torch.Tensor([self.calculateLocalLipLT(temp_weight, None, 
                                                            None, cc, AA, BB, normToUse=np.inf)[-1]]).to(self.device))
                    self.pauseTime(timer, "LipSDP")
            

            for l in range(numLayers-1):
                summationTerm += r[l]**2 * S[l]
            
        if calculateFinalLayerLip:
            return h * summationTerm, r[-1]
        else:
            return h * summationTerm, torch.Tensor([-1]).to(self.device)
    
    def calculateCurvatureConstant(self,
                                   queryCoefficient: torch.Tensor,
                                   g: float,
                                   h: float
                                   ):
        with torch.no_grad():
            params = list(self.network.Linear.parameters()) 
            if len(params) == 4:
                W1 = params[0]
                W2 = params[2]

                W2_eff = queryCoefficient @ self.network.B.cpu() @ W2

                if self.network.activation == 'softplus':
                    W2_pos = ((W2_eff > 0).float()*W2_eff)
                    W2_neg = ((W2_eff < 0).float()*W2_eff)
                elif (self.network.activation == 'sigmoid') or (self.network.activation == 'tanh'):
                    W2_pos = torch.abs(W2_eff)
                    W2_neg = -torch.abs(W2_eff)

                # Check if W2_pos and W2_neg are both zero, set m and M to zero
                # This should be considered
                if torch.all(W2_pos == 0) and torch.all(W2_neg == 0):
                    m = torch.Tensor([0]).to(self.device)
                    M = torch.Tensor([0]).to(self.device)
                else:
                    # print(W2_pos, W2_neg)
                    m = h*power_iteration(W1, W2_neg)
                    M = h*power_iteration(W1, W2_pos)

            elif len(params) == 6:
                W1 = params[0]
                W2 = params[2]
                W3 = params[4]

                W3_eff = queryCoefficient @ self.network.B.cpu() @ W3

                W1_sigma = MatrixNorm.apply(W1)*g
                if self.network.activation == 'softplus':
                    left_m = power_iteration(W2, W3_eff*(W3_eff<0).float())*W1_sigma*W1_sigma
                    left_M = power_iteration(W2, W3_eff*(W3_eff>0).float())*W1_sigma*W1_sigma
                elif (self.network.activation == 'sigmoid') or (self.network.activation == 'tanh'):
                    W3_diag = torch.abs(W3_eff)
                    left_m = power_iteration(W2, W3_diag)*W1_sigma*W1_sigma
                    left_M = left_m.clone()

                W2_tensor = g*W3_eff.unsqueeze(1)*W2.unsqueeze(0)
                W2_neg = ((W2_tensor < 0).float()*W2_tensor).sum(1)
                W2_pos = ((W2_tensor > 0).float()*W2_tensor).sum(1)
                W2_diag = torch.max(W2_neg.abs(), W2_pos.abs())

                if torch.all(W2_pos == 0) and torch.all(W2_neg == 0):
                    m = torch.Tensor([0]).to(self.device)
                    M = torch.Tensor([0]).to(self.device)
                else:
                    if self.network.activation == 'softplus':
                        right_m = power_iteration(W1, W2_neg)
                        right_M = power_iteration(W1, W2_pos)
                    elif (self.network.activation == 'sigmoid') or (self.network.activation == 'tanh'):
                        right_m = power_iteration(W1, W2_diag)
                        right_M = right_m.clone()

                    m = h*(left_m + right_m)
                    M = h*(left_M + right_M)

            elif len(params) == 8:
                W1 = params[0]
                W2 = params[2]
                W3 = params[4]
                W4 = params[6]

                W4_eff = queryCoefficient @ self.network.B.cpu() @ W4
        
                W3_tensor = g*W4_eff.unsqueeze(1)*W3.unsqueeze(0)
                W3_neg = ((W3_tensor < 0).float()*W3_tensor).sum(1)
                W3_pos = ((W3_tensor > 0).float()*W3_tensor).sum(1)
                W3_diag = torch.max(W3_neg.abs(), W3_pos.abs())
                W2_diag = g*(W3_diag.mm(torch.abs(W2)))

                if torch.all(W3_pos == 0) and torch.all(W3_neg == 0):
                    m = torch.Tensor([0]).to(self.device)
                    M = torch.Tensor([0]).to(self.device)

                else:
                    left_m = power_iteration(W1, W2_diag)
                    left_M = left_m.clone()
            
                    W1_sigma = MatrixNorm.apply(W1)*g
                    if self.network.activation == 'softplus':
                        middle_m = power_iteration(W2, W3_neg)*W1_sigma*W1_sigma
                        middle_M = power_iteration(W2, W3_pos)*W1_sigma*W1_sigma
                    elif (self.network.activation == 'sigmoid') or (self.network.activation == 'tanh'):
                        middle_m = power_iteration(W2, W3_diag)*W1_sigma*W1_sigma
                        middle_M = middle_m.clone()
            
                    W1_W2_sigma = W1_sigma*MatrixNorm.apply(W2)*g
                    if self.network.activation == 'softplus':
                        right_m = power_iteration(W3, W4_eff*(W4_eff<0).float())*W1_W2_sigma*W1_W2_sigma
                        right_M = power_iteration(W3, W4_eff*(W4_eff>0).float())*W1_W2_sigma*W1_W2_sigma
                    elif (self.network.activation == 'sigmoid') or (self.network.activation == 'tanh'):
                        W4_diag = torch.abs(W4_eff)
                        right_m = power_iteration(W3, W4_diag)*W1_W2_sigma*W1_W2_sigma
                        right_M = right_m.clone()
            
                    m = h*(left_m + middle_m + right_m)
                    M = h*(left_M + middle_M + right_M)

        # print('M', M)
        return m[0], M[0]
    
    def calculateLipschitzConstant(self,
                                   queryCoefficient: torch.Tensor,
                                   inputLowerBound: torch.Tensor,
                                   inputUpperBound: torch.Tensor,
                                   timer=None
                                   ):

        if (self.normToUse == 2 and not self.useTwoNormDilation) or self.normToUse == 1:
            if self.useSdpForLipschitzCalculation and self.normToUse == 2:
                alpha, beta = self.calculateMinMaxSlopes(None, inputLowerBound, inputUpperBound, self.activation)

                # print(self.network)
                # print(queryCoefficient.unsqueeze(0).cpu().numpy())
                # print(self.network.A.cpu().numpy(), self.network.B.cpu().numpy())
                # raise
                self.startTime(timer, "LipSDP")
                if self.horizon == 1:
                    lipschitzConstant = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                                             queryCoefficient.unsqueeze(0).cpu().numpy(),
                                                             self.network.A.cpu().numpy(),
                                                             self.network.B.cpu().numpy(),
                                                             verbose=self.sdpSolverVerbose)]).to(self.device)
                else:
                    l1 = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                              queryCoefficient.unsqueeze(0).cpu().numpy(),
                                              self.network.A.cpu().numpy(),
                                              self.network.B.cpu().numpy(),
                                              verbose=self.sdpSolverVerbose)]).to(self.device)
                    l2 = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                              np.eye(self.network.A.shape[0]),
                                              self.network.A.cpu().numpy(),
                                              self.network.B.cpu().numpy(),
                                              verbose=self.sdpSolverVerbose)]).to(self.device)
                    lipschitzConstant = l1 * l2 ** (self.horizon - 1)
                self.pauseTime(timer, "LipSDP")
            else:
                lipschitzConstant = torch.from_numpy(
                    self.calculateLocalLipschitzConstantSingleBatchNumpy(self.weights, normToUse=self.normToUse))[-1].to(
                    self.device)
        else:
            raise ValueError
        return lipschitzConstant

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor,
                   virtualBranch=True,
                   timer=None,
                   extractedLipschitzConstants=None):
        
        if virtualBranch and self.performVirtualBranching:
            virtualBranchLowerBounds = \
                self.handleVirtualBranching(inputLowerBound, inputUpperBound,
                                            queryCoefficient, extractedLipschitzConstants, timer)
            
        centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
        if self.boundingMethod == 'firstOrder':
            additiveTerm = self.calculateAdditiveTerm(inputLowerBound, inputUpperBound, queryCoefficient,
                                                    extractedLipschitzConstants, timer)

            with torch.no_grad():
                self.startTime(timer, "lowerBound:lipschitzForwardPass")
                lowerBound = self.network(centerPoint) @ queryCoefficient - additiveTerm
                # upperBound = self.network(centerPoint) @ queryCoefficient + additiveTerm
                self.pauseTime(timer, "lowerBound:lipschitzForwardPass")

        elif self.boundingMethod == 'secondOrder':
            firstOrderAdditiveTerm = torch.Tensor([-1]).to(self.device)
            grad_x, dialation, dialation2, x_center = self.calculateAdditiveTermSecondOrder(inputLowerBound, inputUpperBound, \
                                                                                                         queryCoefficient, timer)
            self.startTime(timer, "lowerBound:lipschitzForwardPass")
            if self.network.isLinear:
                additiveTerm1 = torch.sum(grad_x * dialation2, axis=1)
                additiveTerm2 = self.calculatedCurvatureConstants / 2 * torch.linalg.norm(dialation, dim=1, ord=self.normToUse)**2

                secondOrderAdditiveTerm = additiveTerm1 + additiveTerm2
                firstOrderAdditiveTerm = self.LipCnt * torch.linalg.norm(dialation, dim=1, ord=self.normToUse)
                
            else:
                fullNLCurvatureConstant = self.calculateNLCurvature(queryCoefficient, inputLowerBound, inputUpperBound)
                additiveTerm1 = torch.sum(grad_x * dialation2, axis=1)
                additiveTerm2 = fullNLCurvatureConstant / 2 * torch.linalg.norm(dialation, dim=1, ord=self.normToUse)**2
                secondOrderAdditiveTerm = additiveTerm1 + additiveTerm2
                firstOrderAdditiveTerm = torch.Tensor([-1]).to(self.device) #@TODO: Fix this

            with torch.no_grad():
                temp1 = self.network(centerPoint) @ queryCoefficient - secondOrderAdditiveTerm
                if torch.any(firstOrderAdditiveTerm < 0) :
                    # Means that the lip constant was not calculated
                    temp2 = temp1
                else:
                    temp2 = self.network(centerPoint) @ queryCoefficient - firstOrderAdditiveTerm

                lowerBound = torch.maximum(temp1, temp2)
                # upperBound = self.network(centerPoint) @ queryCoefficient + additiveTerm1 + additiveTerm2
            self.pauseTime(timer, "lowerBound:lipschitzForwardPass")

        elif self.boundingMethod == 'CROWN':
            lowerBound, __ = self.calculateboundsCROWN(inputLowerBound, inputUpperBound, queryCoefficient)
        
        if virtualBranch and self.performVirtualBranching:
            lowerBound = torch.maximum(lowerBound, virtualBranchLowerBounds)

        return lowerBound

    def handleVirtualBranching(self, inputLowerBound, inputUpperBound, queryCoefficient,
                               extractedLipschitzConstants, timer):
        difference = inputUpperBound - inputLowerBound
        batchSize = inputUpperBound.shape[0]
        self.startTime(timer, "lowerBound:virtualBranchPreparation")
        maxIndices = torch.argmax(difference, 1)
        newLowers = [inputLowerBound[i, :].clone() for i in range(batchSize)
                     for _ in range(self.numberOfVirtualBranches)]
        newUppers = [inputUpperBound[i, :].clone() for i in range(batchSize)
                     for _ in range(self.numberOfVirtualBranches)]
        virtualBranchLipschitz = None
        if extractedLipschitzConstants is not None:
            virtualBranchLipschitz = \
                torch.zeros(self.numberOfVirtualBranches * extractedLipschitzConstants.shape[0]).to(self.device)
            for i in range(batchSize):
                virtualBranchLipschitz[self.numberOfVirtualBranches * i: self.numberOfVirtualBranches * (i + 1)] = \
                    extractedLipschitzConstants[i]
        for i in range(batchSize):
            for j in range(self.numberOfVirtualBranches):
                newUppers[self.numberOfVirtualBranches * i + j][maxIndices[i]] = \
                    newLowers[self.numberOfVirtualBranches * i + j][maxIndices[i]] + \
                    (j + 1) * difference[i, maxIndices[i]] / self.numberOfVirtualBranches
                newLowers[self.numberOfVirtualBranches * i + j][maxIndices[i]] += \
                    j * difference[i, maxIndices[i]] / self.numberOfVirtualBranches

        newLowers = torch.vstack(newLowers)
        newUppers = torch.vstack(newUppers)
        self.pauseTime(timer, "lowerBound:virtualBranchPreparation")

        virtualBranchLowerBoundsExtra = self.lowerBound(queryCoefficient, newLowers, newUppers, False, timer=timer,
                                                        extractedLipschitzConstants=virtualBranchLipschitz)
        self.startTime(timer, "lowerBound:virtualBranchMin")

        virtualBranchLowerBounds = torch.Tensor([torch.min(
            virtualBranchLowerBoundsExtra[i * self.numberOfVirtualBranches:(i + 1) * self.numberOfVirtualBranches])
            for i in range(0, batchSize)]).to(self.device)
        self.pauseTime(timer, "lowerBound:virtualBranchMin")
        return virtualBranchLowerBounds
    
    def compareSecondOrder(self, inputLowerBound, inputUpperBound, queryCoefficient, timer):
        if self.network.activation == 'softplus':
            g = 1.
            h = 0.25
        elif self.network.activation == 'sigmoid':
            g = 0.25
            h = 0.09623
        elif self.network.activation == 'tanh':
            g = 1.
            h = 0.7699

        M1, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=0,
                                                                        local=False)
        M2, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=0,
                                                                        local=True)
        M3, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=1,
                                                                        local=False)
        a = time.time()
        M4, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=1,
                                                                        local=True)
        b = time.time()
        M5, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=2,
                                                                        local=False)
        c = time.time()
        M6, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                        inputLowerBound, inputUpperBound, timer, lipMethod=2
                                                                        , local=True)
        d = time.time()
        return M1, M2, M3, M4, M5, M6, [b-a, d-c]


    def calculateAdditiveTermSecondOrder(self, inputLowerBound, inputUpperBound, queryCoefficient, timer):
        def J_c(x):
            return self.network(x) @ queryCoefficient
        
        if self.network.activation == 'softplus':
            g = 1.
            h = 0.25
        elif self.network.activation == 'sigmoid':
            g = 0.25
            h = 0.09623
        elif self.network.activation == 'tanh':
            g = 1.
            h = 0.7699

        x_center = (inputLowerBound + inputUpperBound) / 2.0

        curvatureMethod = [1]
        lipMethod = self.lipMethod
        if len(self.network.Linear) > 10:
            # print('Using Naive Lip')
            lipMethod = 1
        if self.calculatedCurvatureConstants == []:
            m, M, lipcnt = torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1])
            if 0 in curvatureMethod:
                m, M = self.calculateCurvatureConstant(queryCoefficient, g, h)
                # print('--', M)
            if 1 in curvatureMethod:
                M, lipcnt = self.calculateCurvatureConstantGeneral(queryCoefficient, g, h, 
                                                                    inputLowerBound, inputUpperBound, timer, 
                                                                    lipMethod=lipMethod)

            self.calculatedCurvatureConstants.append(torch.maximum(m, M))
            self.calculatedCurvatureConstants = torch.Tensor(self.calculatedCurvatureConstants).to(self.device)
            self.LipCnt = torch.Tensor([lipcnt]).to(self.device)

        # because it takes the jacobian w.r.t all input output pairs
        grad_x = torch.sum(jacobian(J_c, (x_center)), axis=1).reshape(x_center.shape)
        dialation = (inputUpperBound - inputLowerBound)/2
        dialation2 = dialation * torch.sign(grad_x)
        return grad_x, dialation, dialation2, x_center

    def calculateAdditiveTerm(self, inputLowerBound, inputUpperBound, queryCoefficient,
                              extractedLipschitzConstants,
                              timer):
        batchSize = inputUpperBound.shape[0]
        difference = inputUpperBound - inputLowerBound
        # this function is not optimal for cases in which an axis is cut into unequal segments
        dilationVector = difference / torch.tensor(2., device=self.device)
        if (self.normToUse == 2 and not self.useTwoNormDilation) or self.normToUse == 1:
            if extractedLipschitzConstants is None:
                if len(self.calculatedLipschitzConstants) == 0:
                    self.startTime(timer, "lowerBound:lipschitzCalc")
                    if self.useSdpForLipschitzCalculation and self.normToUse == 2:
                        # num_neurons = sum([newWeights[i].shape[0] for i in range(len(newWeights) - 1)])
                        # alpha = np.zeros((num_neurons, 1))
                        # beta = np.ones((num_neurons, 1))
                        alpha, beta = self.calculateMinMaxSlopes(None, inputLowerBound, inputUpperBound)
                        if self.horizon == 1:
                            self.startTime(timer, "LipSDP")
                            lipschitzConstant = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                                                     queryCoefficient.unsqueeze(0).cpu().numpy(),
                                                                     self.network.A.cpu().numpy(),
                                                                     self.network.B.cpu().numpy(),
                                                                     verbose=self.sdpSolverVerbose)]).to(self.device)
                            self.pauseTime(timer, "LipSDP")
                        else:
                            self.startTime(timer, "LipSDP")
                            l1 = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                                      queryCoefficient.unsqueeze(0).cpu().numpy(),
                                                      self.network.A.cpu().numpy(),
                                                      self.network.B.cpu().numpy(),
                                                      verbose=self.sdpSolverVerbose)]).to(self.device)
                            l2 = torch.Tensor([lipSDP(self.weights, alpha, beta,
                                                      np.eye(self.network.A.shape[0]),
                                                      self.network.A.cpu().numpy(),
                                                      self.network.B.cpu().numpy(),
                                                      verbose=self.sdpSolverVerbose)]).to(self.device)
                            lipschitzConstant = l1 * l2 ** (self.horizon - 1)
                            self.pauseTime(timer, "LipSDP")
                    else:
                        self.startTime(timer, "LipSDP")
                        lipschitzConstant = torch.from_numpy(
                            self.calculateLocalLipschitzConstantSingleBatchNumpy(self.weights,
                                                                                 normToUse=self.normToUse))[-1].to(
                            self.device)
                        self.startTime(timer, "LipSDP")
                    self.calculatedLipschitzConstants.append(lipschitzConstant)
                    self.pauseTime(timer, "lowerBound:lipschitzCalc")
                else:
                    lipschitzConstant = self.calculatedLipschitzConstants[0]
            else:
                lipschitzConstant = extractedLipschitzConstants
            multipliers = torch.linalg.norm(dilationVector, ord=self.normToUse, dim=1)
            additiveTerm = lipschitzConstant * multipliers
        elif self.normToUse == float("inf") or (self.normToUse == 2 and self.useTwoNormDilation):
            self.startTime(timer, "lowerBound:lipschitzSearch")
            batchesThatNeedLipschitzConstantCalculation = [i for i in range(batchSize)]
            lipschitzConstants = -torch.ones(batchSize, device=self.device)
            locationOfUnavailableConstants = {}
            for batchCounter in range(batchSize):  # making it reversed might just help a tiny amount.
                foundLipschitzConstant = False
                if not foundLipschitzConstant:
                    for i in range(len(self.calculatedLipschitzConstants) - 1,
                                   max(len(self.calculatedLipschitzConstants) - self.maxSearchDepth, -1), -1):
                        existingDilationVector, lipschitzConstant = self.calculatedLipschitzConstants[i]
                        # if torch.norm(dilationVector[batchCounter, :] - existingDilationVector) < 1e-8:
                        if torch.allclose(dilationVector[batchCounter, :], existingDilationVector, rtol=1e-3,
                                          atol=1e-7):
                            if lipschitzConstant == -1:
                                locationOfUnavailableConstants[batchCounter] = i
                            else:
                                lipschitzConstants[batchCounter] = lipschitzConstant
                            batchesThatNeedLipschitzConstantCalculation.remove(batchCounter)
                            foundLipschitzConstant = True
                            break
                if not foundLipschitzConstant:
                    locationOfUnavailableConstants[batchCounter] = len(self.calculatedLipschitzConstants)
                    # suppose we divide equally along an axes. Then the lipschitz constant of the two subdomains are gonna
                    # be the same. By adding the dilationVector of one of the sides, we are preventing the calculation of
                    # the lipschitz constant for both sides when they are exactly the same.
                    self.calculatedLipschitzConstants.append([dilationVector[batchCounter, :], -1])
            self.pauseTime(timer, "lowerBound:lipschitzSearch")
            self.startTime(timer, "lowerBound:lipschitzCalc")
            if len(batchesThatNeedLipschitzConstantCalculation) != 0:
                if self.normToUse == 2:
                    normalizerDilationVector = torch.sqrt(difference.shape[1]) * dilationVector
                else:
                    normalizerDilationVector = dilationVector
                """"""
                # Torch batch implementation
                # newWeights = [w.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1) for w in self.weights]
                # # w @ D is equivalent to w * dilationVector
                # # newWeights[0] = newWeights[0] @ dMatrix
                #
                # newWeights[0] = newWeights[0] * normalizerDilationVector[batchesThatNeedLipschitzConstantCalculation, :].unsqueeze(1)
                # # print(newWeights[0])
                # queryCoefficientRepeated = queryCoefficient.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1)
                # # newWeights[-1] = queryCoefficient @ newWeights[-1]
                #
                # newWeights[-1] = torch.bmm(queryCoefficientRepeated, newWeights[-1])
                # # print(newWeights)
                # newCalculatedLipschitzConstants = self.calculateLocalLipschitzConstantTorch(newWeights, self.device, self.normToUse)[:, -1]
                """"""
                # Numpy single batch implementation
                newCalculatedLipschitzConstants = []
                for i in range(len(batchesThatNeedLipschitzConstantCalculation)):
                    newWeights = [np.copy(w) for w in self.weights]
                    newWeights[0] = newWeights[0] * normalizerDilationVector[
                                                    batchesThatNeedLipschitzConstantCalculation[i]:
                                                    batchesThatNeedLipschitzConstantCalculation[i] + 1, :].cpu().numpy()
                    newWeights[-1] = queryCoefficient.unsqueeze(0).cpu().numpy() @ newWeights[-1]
                    # print(newWeights)
                    newCalculatedLipschitzConstants.append(torch.from_numpy(
                        self.calculateLocalLipschitzConstantSingleBatchNumpy(newWeights, normToUse=self.normToUse))[
                                                               -1].to(self.device))

                """"""
                for i in range(len(newCalculatedLipschitzConstants)):
                    self.calculatedLipschitzConstants[locationOfUnavailableConstants[
                        batchesThatNeedLipschitzConstantCalculation[i]]][1] = newCalculatedLipschitzConstants[i]
                for unavailableBatch in locationOfUnavailableConstants.keys():
                    lipschitzConstants[unavailableBatch] = \
                        self.calculatedLipschitzConstants[locationOfUnavailableConstants[unavailableBatch]][1]

            self.pauseTime(timer, "lowerBound:lipschitzCalc")
            if torch.any(lipschitzConstants < 0):
                print("error. lipschitz constant hasn't been calculated")
                raise
            additiveTerm = lipschitzConstants
        return additiveTerm


    @staticmethod
    def startTime(timer, timerName):
        try:
            timer.start(timerName)
        except:
            pass

    @staticmethod
    def pauseTime(timer, timerName):
        try:
            timer.pause(timerName)
        except:
            pass

    @staticmethod
    def calculateLocalLipschitzConstantSingleBatchNumpy(weights, coef=None, A=None, B=None, normToUse=float("inf")):
        numberOfWeights = len(weights)
        ms = np.zeros(numberOfWeights, dtype=np.float64)
        ms[0] = np.linalg.norm(weights[0], normToUse)


        dim_in = weights[0].shape[1]
        dim_out = weights[-1].shape[0]
        if B is None:
            B = np.eye(dim_in, dim_out)

        weights[-1] = coef @ B @ weights[-1]
            
        for i in range(1, numberOfWeights):
            multiplier = 1.
            temp = 0.
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = productMatrix @ weights[k]
                if j > 0:
                    multiplier *= 0.5
                    print("could be a bug!")
                    if (i == numberOfWeights - 1) and (A is not None):
                        temp += np.linalg.norm(multiplier * productMatrix + coef @ A, normToUse) * ms[j - 1]
                    else:
                        temp += multiplier * np.linalg.norm(productMatrix, normToUse) * ms[j - 1]
                        
                else:
                    temp += multiplier * np.linalg.norm(productMatrix, normToUse)
            ms[i] = temp
        # print(ms)
        return ms
    
    def calculateLocalLipLT(self, weights, alpha=None, beta=None, coef=None, A=None, B=None, normToUse=float("inf")):
        neurons = [weights[i].shape[0] for i in range(len(weights) - 1)]
        numberOfNeurons = sum(neurons)
        if alpha is None or beta is None:
            alpha = np.zeros((numberOfNeurons, 1))
            beta = np.ones((numberOfNeurons, 1))
        
        numberOfWeights = len(weights)
        # To handle closedLoop System, B only affects the final layer
        dim_in = weights[0].shape[1]
        dim_out = weights[-1].shape[0]
        if B is None:
            B = np.eye(dim_out, dim_out)
        weights[-1] = coef @ B @ weights[-1]

        ms = np.zeros(numberOfWeights, dtype=np.float64)
        # if normToUse == np.inf:
        #     print(numberOfWeights, self.network)
        #     print([weights[i].shape for i in range(numberOfWeights)])
            
        ms[0] = np.linalg.norm(weights[0], normToUse)

        # Calculating D and D'
        neuronCounter = 0
        D = []
        Dprime = []
        for i in range(numberOfWeights - 1):
            # print('code changed here!')
            # D.append(np.diag(np.squeeze(alpha[neuronCounter: neuronCounter + neurons[i]] + 
            #                             beta[neuronCounter: neuronCounter + neurons[i]] , axis=1)/2))
            # Dprime.append(np.diag(np.squeeze(-alpha[neuronCounter: neuronCounter + neurons[i]] + \
            #                                      beta[neuronCounter: neuronCounter + neurons[i]], axis=1)/2))
            D.append(np.diag(np.squeeze(beta[neuronCounter: neuronCounter + neurons[i]] , axis=1)/2))
            Dprime.append(np.diag(np.squeeze(beta[neuronCounter: neuronCounter + neurons[i]], axis=1)/2))
            neuronCounter += neurons[i]


        # print(self.network)
        # a = 1/4 * np.linalg.norm(weights[0].T @ weights[1].T @ weights[2].T, normToUse)
        # b = 3/4 * np.linalg.norm(np.abs(weights[0].T) @ np.abs(weights[1].T) @ np.abs(weights[2].T), normToUse)
        # c = np.linalg.norm(np.abs(weights[0].T) @ np.abs(weights[1].T) @ np.abs(weights[2].T), normToUse)
        # print(a)
        # print(b)
        # print('sol: ', a + b, c)
        # print('---------------------------------------------------------')
        # raise

        # if normToUse == float("inf"):
        #     absWeightProd = [np.abs(weights[0])]
        #     # print('--', absWeightProd[-1].shape)
        #     # print(np.linalg.norm(absWeightProd[0], normToUse))
        #     for i in range(1, numberOfWeights):
        #         absWeightProd.append(beta[i-1] * np.abs(weights[i]) @ absWeightProd[i-1])
        #         # print(np.linalg.norm(absWeightProd[i], normToUse))
        #         # print('---', absWeightProd[-1].shape)
        # print('---------------------------------------------------------')
        # calculating m[i] 
        for i in range(1, numberOfWeights):
            temp = 0.
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = productMatrix @ D[k] @ weights[k]
                if j > 0:
                    # if normToUse == float("inf"):
                    #     # print('$$', temp)
                    #     # print((Dprime[j-1] @ absWeightProd[j-1].T).T.shape)
                    #     # print(np.linalg.norm(Dprime[j-1] @ absWeightProd[j-1].T, normToUse))
                    #     # print(np.linalg.norm(absWeightProd[j-1] @ Dprime[j-1].T, normToUse))
                    #     # print(np.abs(productMatrix).shape,  (Dprime[j-1] @ absWeightProd[j-1]).shape)
                    #     # print((np.abs(productMatrix) @ Dprime[j-1] @ absWeightProd[j-1]).shape)
                    #     temp += np.linalg.norm((np.abs(productMatrix) @ Dprime[j-1] @ absWeightProd[j-1]), normToUse)
                    #     # print('$$', temp)

                    #     # print('+', j, '--', np.linalg.norm(np.abs(productMatrix)), np.linalg.norm((Dprime[j-1] @ absWeightProd[j-1].T).T, normToUse))
                    #     # print('+', ms)
                    # else:
                        temp += np.linalg.norm(productMatrix, normToUse) * np.linalg.norm(Dprime[j-1], normToUse) * ms[j - 1]   
                else:
                    if (i == numberOfWeights - 1) and (A is not None):
                        temp += np.linalg.norm((productMatrix + coef @ A), normToUse)
                    else:
                        temp += np.linalg.norm(productMatrix, normToUse)
                        # print('==', np.linalg.norm(productMatrix, normToUse))
                        # if normToUse == float("inf"):
                        #     print('$$', temp)
                        #     print(productMatrix.shape)
            ms[i] = temp
        
        # if normToUse == float("inf"):
        #     print(ms)
        #     print('-----------------')
        #     raise
        #     print(ms)
        #     print(1/2 * np.linalg.norm(np.abs(weights[0]).T @ np.abs(weights[1]).T, normToUse) + 
        #           1/2 * np.linalg.norm(weights[0].T @ weights[1].T, normToUse))
        #     print(np.linalg.norm(np.abs(weights[0]).T @ np.abs(weights[1]).T, normToUse))
        #     raise
        return ms


    @staticmethod
    def extractWeightsFromNetwork(network: nn.Module):
        weights = []
        biases = []
        for name, param in network.Linear.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone().cpu().numpy())
            elif "bias" in name:
                biases.append(param.detach().clone().cpu().numpy()[:, np.newaxis])
        return weights, biases

    @staticmethod
    def calculateBoundsAfterLinearTransformation(weight, bias, lowerBound, upperBound):
        """
        :param weight:
        :param bias: A (n * 1) matrix
        :param lowerBound: A vector and not an (n * 1) matrix
        :param upperBound: A vector and not an (n * 1) matrix
        :return:
        """
        outputLowerBound = (np.maximum(weight, 0) @ (lowerBound[np.newaxis].transpose())
                            + np.minimum(weight, 0) @ (upperBound[np.newaxis].transpose()) + bias).squeeze()
        outputUpperBound = (np.maximum(weight, 0) @ (upperBound[np.newaxis].transpose())
                            + np.minimum(weight, 0) @ (lowerBound[np.newaxis].transpose()) + bias).squeeze()

        return outputLowerBound, outputUpperBound


    def propagateBoundsInNetwork(self, l, u, weights, biases, activation='relu'):
        if activation == 'relu':
            activation = lambda x: np.maximum(x, 0)
        elif activation == 'tanh':
            activation = lambda x: np.tanh(x)
        elif activation == 'softplus':
            activation = lambda x: np.log(1 + np.exp(x))
        elif activation == 'sigmoid':
            activation = lambda x: 1 / (1 + np.exp(-x))

        s, t = [l], [u]
        for i, (W, b) in enumerate(zip(weights, biases)):
            val1, val2 = s[-1], t[-1]
            if 0 < i:
                val1, val2 = activation(val1), activation(val2)
            if val1.shape == ():
                val1 = np.array([val1])
                val2 = np.array([val2])
            sTemp, tTemp = self.calculateBoundsAfterLinearTransformation(W, b, val1, val2)
            if sTemp.shape == ():
                sTemp = np.array([sTemp])
                tTemp = np.array([tTemp])
            s.append(sTemp)
            t.append(tTemp)
        return s, t

    def calculateMinMaxSlopes(self, weights, inputLowerBound, inputUpperBound, activation, biases=None):
        if weights == None:
            weights = self.weights
            biases = self.biases
        numberOfNeurons = sum([weights[i].shape[0] for i in range(len(weights) - 1)])
        if activation == 'softplus' or activation == 'tanh' or activation == 'relu':
            alpha = np.zeros((numberOfNeurons, 1))
            beta = np.ones((numberOfNeurons, 1))
        elif activation == 'sigmoid':
            alpha = np.zeros((numberOfNeurons, 1))
            beta = 0.25 * np.ones((numberOfNeurons, 1))

        if True and (self.network.activation == 'relu' or self.network.activation == 'tanh'):
            # For Local Calculations
            inputLowerBound = torch.Tensor(inputLowerBound)
            assert inputLowerBound.shape[0] == 1
            lowerBounds, upperBounds = self.propagateBoundsInNetwork(inputLowerBound[0, :].cpu().numpy(),
                                                                    inputUpperBound[0, :].cpu().numpy(),
                                                                    weights, biases, 
                                                                    self.network.activation)
            lowerBounds = np.hstack(lowerBounds[1:-1]).T
            upperBounds = np.hstack(upperBounds[1:-1]).T
            if self.network.activation == 'relu':
                alpha[lowerBounds >= 0] = 1
                beta[upperBounds <= 0] = 0
            elif self.network.activation == 'tanh':
                # If upperbound and lowerbound are both negative
                alpha[upperBounds <= 0] = (1 - np.tanh(lowerBounds[upperBounds <= 0]) ** 2)[:, np.newaxis]
                beta[upperBounds <= 0] = (1 - np.tanh(upperBounds[upperBounds <= 0]) ** 2)[:, np.newaxis]
                #If upperbound and lowerbound are both positive
                alpha[lowerBounds >= 0] = (1 - np.tanh(lowerBounds[lowerBounds >= 0]) ** 2)[:, np.newaxis]
                beta[lowerBounds >= 0] = (1 - np.tanh(upperBounds[lowerBounds >= 0]) ** 2)[:, np.newaxis]
                #If upperbound is positive and lowerbound is negative
                indexes = [a and b for a, b in zip(lowerBounds<= 0, upperBounds>= 0)]
                alpha[indexes] = np.minimum((1 - np.tanh(lowerBounds[indexes]) ** 2)[:, np.newaxis],
                                            (1 - np.tanh(upperBounds[indexes]) ** 2)[:, np.newaxis])

        return alpha, beta
    
    def calculateNaiveLip(self, weights, beta, coef, Asys, Bsys, normToUse=np.infty):
        numLayers = len(weights)
        if (beta == None).any():
            beta = np.ones((numLayers - 1, 1))
        neurons = [weights[i].shape[0] for i in range(numLayers - 1)]

        # To handle closedLoop System, B only affects the final layer
        dim_in = weights[0].shape[1]
        dim_out = weights[-1].shape[0]
        if Bsys is None:
            Bsys = np.eye(dim_in, dim_out)

        g = np.zeros((numLayers - 1, 1))
        neuronCounter = 0
        for i in range(numLayers-1):
            g[i] = np.max(beta[neuronCounter : neuronCounter + neurons[i]])
            neuronCounter += neurons[i]

        r = np.zeros((numLayers, 1))
        r[0] = np.linalg.norm(weights[i], ord=normToUse)
        for i in range(1, numLayers):
            if i < numLayers - 1:
                r[i] = g[i] * np.linalg.norm(weights[i], ord=normToUse) * r[i - 1]
            else:
                if self.network.isLinear and Asys is not None:
                    r[i] = g[i-1] * np.linalg.norm(coef @ Bsys  @ weights[i], ord=normToUse) * r[i - 1] \
                                    + np.linalg.norm(coef @ Asys, ord=normToUse)
                else:
                    r[i] = g[i-1] * np.linalg.norm(weights[i], ord=normToUse) * r[i - 1]
        return r



def lipSDP(weights, alpha, beta, coef, Asys=None, Bsys=None, verbose=False):
    # @TODO: Possible bug in weights input
    num_layers = len(weights) - 1
    dim_in = weights[0].shape[1]
    dim_out = weights[-1].shape[0]
    dim_last_hidden = weights[-1].shape[1]
    hidden_dims = [weights[i].shape[0] for i in range(0, num_layers)]
    dims = [dim_in] + hidden_dims + [dim_out]
    num_neurons = sum(hidden_dims)

    if Asys is None:
        Asys = np.zeros((dim_out, dim_in))
    if Bsys is None:
        Bsys = np.eye(dim_out, dim_out)
    # decision vars
    Lambda = cp.Variable((num_neurons, 1), nonneg=True)
    T = cp.diag(Lambda)
    rho = cp.Variable((1, 1), nonneg=True)
    A = weights[0]
    # C = np.bmat([np.zeros((weights[-1].shape[0], dim_in + num_neurons - dim_last_hidden)), weights[-1]])
    E0 = np.bmat([np.eye(weights[0].shape[1]), np.zeros((weights[0].shape[1], dim_in + num_neurons - dim_in))])
    El = np.bmat([np.zeros((weights[-1].shape[1], dim_in + num_neurons - dim_last_hidden)), np.eye(weights[-1].shape[1])])
    # print(Bsys.shape, weights[-1].shape, El.shape)
    Asys = coef @ Asys
    Bsys = coef @ Bsys
    Cnew = Asys @ E0 + Bsys @ weights[-1] @ El
    C = Cnew

    D = np.bmat([np.eye(dim_in), np.zeros((dim_in, num_neurons))])

    for i in range(1, num_layers):
        A = block_diag(A, weights[i])

    A = np.bmat([A, np.zeros((A.shape[0], weights[num_layers].shape[1]))])
    B = np.eye(num_neurons)
    B = np.bmat([np.zeros((num_neurons, weights[0].shape[1])), B])
    A_on_B = np.bmat([[A], [B]])

    cons = [A_on_B.T @ cp.bmat(
        [[-2 * np.diag(alpha[:, 0]) @ np.diag(beta[:, 0]) @ T, np.diag(alpha[:, 0] + beta[:, 0]) @ T],
         [np.diag(alpha[:, 0] + beta[:, 0]) @ T, -2 * T]]) @ A_on_B + C.T @ C - rho * D.T @ D << 0]

    prob = cp.Problem(cp.Minimize(rho), cons)

    prob.solve(solver=cp.MOSEK, verbose=verbose)

    return np.sqrt(rho.value)[0][0]




