from typing import List

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import cvxpy as cp
from scipy.linalg import block_diag

from Bounding.Utils4Curvature import power_iteration
from torch.autograd.functional import jacobian



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
                 boundingMethod='firstOrder'):
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

    def calculateCurvatureConstant(self,
                                   queryCoefficient: torch.Tensor,
                                   ):
        if self.network.activation == 'softplus':
            g = 1.
            h = 0.25
        elif self.network.activation == 'sigmoid':
            g = 0.25
            h = 0.09623
        elif self.network.activation == 'tanh':
            g = 1.
            h = 0.7699
        with torch.no_grad():
            # if model.num_layers == 2:
                params = list(self.network.parameters())
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
                    m = h*power_iteration(W1, W2_neg)
                    M = h*power_iteration(W1, W2_pos)

                print('m: ', m)
        
        return m[0], M[0]
    
    def calculateLipschitzConstant(self,
                                   queryCoefficient: torch.Tensor,
                                   inputLowerBound: torch.Tensor,
                                   inputUpperBound: torch.Tensor,
                                   ):

        if (self.normToUse == 2 and not self.useTwoNormDilation) or self.normToUse == 1:
            if self.useSdpForLipschitzCalculation and self.normToUse == 2:
                alpha, beta = self.calculateMinMaxSlopes(inputLowerBound, inputUpperBound, self.activation)

                # print(self.network)
                # print(queryCoefficient.unsqueeze(0).cpu().numpy())
                # print(self.network.A.cpu().numpy(), self.network.B.cpu().numpy())
                # raise
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

        if self.boundingMethod == 'firstOrder':
            additiveTerm = self.calculateAdditiveTerm(inputLowerBound, inputUpperBound, queryCoefficient,
                                                    extractedLipschitzConstants, timer)

            centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
            with torch.no_grad():
                self.startTime(timer, "lowerBound:lipschitzForwardPass")
                lowerBound = self.network(centerPoint) @ queryCoefficient - additiveTerm
                self.pauseTime(timer, "lowerBound:lipschitzForwardPass")
        elif self.boundingMethod == 'secondOrder':
            additiveTerm1, additiveTerm2 = self.calculateAdditiveTermSecondOrder(inputLowerBound, inputUpperBound, queryCoefficient)
            centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
            with torch.no_grad():
                self.startTime(timer, "lowerBound:lipschitzForwardPass")
                lowerBound = self.network(centerPoint) @ queryCoefficient - additiveTerm1 - additiveTerm2
                self.pauseTime(timer, "lowerBound:lipschitzForwardPass")

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

    def calculateAdditiveTermSecondOrder(self, inputLowerBound, inputUpperBound, queryCoefficient):
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

        params = list(self.network.parameters())
        W1 = params[0].data
        W2 = params[2].data

        x_center = (inputLowerBound + inputUpperBound) / 2.0
        if self.calculatedCurvatureConstants == []:
            m, M = self.calculateCurvatureConstant(queryCoefficient)
            self.calculatedCurvatureConstants.append(torch.maximum(m, M))
            self.calculatedCurvatureConstants = torch.Tensor(self.calculatedCurvatureConstants).to(self.device)

        # because it takes the jacobian w.r.t all input output pairs
        grad_x = torch.sum(jacobian(J_c, (x_center)), axis=1).reshape(x_center.shape)
        dialation = (inputUpperBound - inputLowerBound)/2
        dialation2 = dialation * torch.sign(grad_x)

        return torch.sum(grad_x * dialation2, axis=1), self.calculatedCurvatureConstants / 2 * torch.linalg.norm(dialation, dim=1)**2

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
                        alpha, beta = self.calculateMinMaxSlopes(inputLowerBound, inputUpperBound)
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
                    else:
                        lipschitzConstant = torch.from_numpy(
                            self.calculateLocalLipschitzConstantSingleBatchNumpy(self.weights,
                                                                                 normToUse=self.normToUse))[-1].to(
                            self.device)
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
    def calculateLocalLipschitzConstantSingleBatchNumpy(weights, normToUse=float("inf")):
        numberOfWeights = len(weights)
        ms = np.zeros(numberOfWeights, dtype=np.float64)
        ms[0] = np.linalg.norm(weights[0], normToUse)
        for i in range(1, numberOfWeights):
            multiplier = 1.
            temp = 0.
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = productMatrix @ weights[k]
                    # if j == 0:
                    #     print(weights[k])
                if j > 0:
                    multiplier *= 0.5
                    temp += multiplier * np.linalg.norm(productMatrix, normToUse) * ms[j - 1]
                else:
                    temp += multiplier * np.linalg.norm(productMatrix, normToUse)
            ms[i] = temp
        # print(ms)
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


    def propagateBoundsInNetwork(self, l, u, weights, biases):
        relu = lambda x: np.maximum(x, 0)

        s, t = [l], [u]
        for i, (W, b) in enumerate(zip(weights, biases)):
            val1, val2 = s[-1], t[-1]
            if 0 < i:
                val1, val2 = relu(val1), relu(val2)
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

    def calculateMinMaxSlopes(self, inputLowerBound, inputUpperBound, activation):
        numberOfNeurons = sum([self.weights[i].shape[0] for i in range(len(self.weights) - 1)])
        if activation == 'softplus' or activation == 'tanh':
            alpha = np.zeros((numberOfNeurons, 1))
            beta = np.ones((numberOfNeurons, 1))
        elif activation == 'sigmoid':
            alpha = np.zeros((numberOfNeurons, 1))
            beta = 0.25 * np.ones((numberOfNeurons, 1))

        assert inputLowerBound.shape[0] == 1
        # lowerBounds, upperBounds = self.propagateBoundsInNetwork(inputLowerBound[0, :].cpu().numpy(),
        #                                                          inputUpperBound[0, :].cpu().numpy(),
        #                                                          self.weights, self.biases)
        # lowerBounds = np.hstack(lowerBounds[1:-1]).T
        # upperBounds = np.hstack(upperBounds[1:-1]).T
        # alpha[lowerBounds >= 0] = 1
        # beta[upperBounds <= 0] = 0
        return alpha, beta



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
        Asys = np.zeros((dim_in, dim_in))
        Bsys = np.eye(dim_in)
    # decision vars
    Lambda = cp.Variable((num_neurons, 1), nonneg=True)
    T = cp.diag(Lambda)
    rho = cp.Variable((1, 1), nonneg=True)
    A = weights[0]
    # C = np.bmat([np.zeros((weights[-1].shape[0], dim_in + num_neurons - dim_last_hidden)), weights[-1]])
    E0 = np.bmat([np.eye(weights[0].shape[1]), np.zeros((weights[0].shape[1], dim_in + num_neurons - dim_in))])
    El = np.bmat([np.zeros((weights[-1].shape[1], dim_in + num_neurons - dim_last_hidden)), np.eye(weights[-1].shape[1])])
    # print(Asys.shape, E0.shape)
    # print(Bsys.shape, weights[-1].shape, El.shape)
    Asys = coef @ Asys
    Bsys = coef @ Bsys
    # print(Asys.shape, Bsys.shape)
    # print((Asys @ E0).shape, (Bsys).shape, (weights[-1] @ El).shape)
    # print('--')
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
