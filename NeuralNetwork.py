from packages import *

class NeuralNetwork(nn.Module):
    def __init__(self, path, A=None, B=None, c=None, activation='softplus', loadOrGenerate=True, isLinear=True):
        super().__init__()
        self.activation = activation
        self.isLinear = isLinear
        if activation == 'softplus':
            activationF = nn.Softplus()
        elif activation == 'sigmoid':
            activationF = nn.Sigmoid()
        elif activation == 'tanh':
            activationF = nn.Tanh()
        else:
            activationF = nn.ReLU()

        if loadOrGenerate:
            if self.isLinear == False:
                tmp = path.split(('.'))[-2]
                self.NLBench = tmp.split('/')[-1]
            stateDictionary = torch.load(path, map_location=torch.device("cpu"))
            layers = []
            for keyEntry in stateDictionary:
                if "weight" in keyEntry:
                    layers.append(nn.Linear(stateDictionary[keyEntry].shape[1], stateDictionary[keyEntry].shape[0]))
                    layers.append(activationF)
            layers.pop()
            self.Linear = nn.Sequential(
                *layers
            )
            self.load_state_dict(stateDictionary)

        else:
            self.Linear = nn.Sequential(
                            nn.Linear(784, 100),
                            activationF,
                            nn.Linear(100, 100),
                            activationF,
                            nn.Linear(100, 100),
                            activationF,
                            nn.Linear(100, 100),
                            activationF,
                            nn.Linear(100, 10))
        
        self.rotation = nn.Identity()
        self.A = A
        self.B = B
        self.c = c
        if self.A is None:
            dimInp = self.Linear[0].weight.shape[1]
            dimOut = self.Linear[-1].weight.shape[0]
            if self.isLinear:
                self.A = torch.zeros((dimOut, dimInp)).float()
                self.B = torch.eye(dimOut, dimOut).float()
                self.c = torch.zeros(dimOut).float()
            else:
                self.A = torch.zeros((dimInp, dimInp)).float()
                self.B = torch.zeros((dimInp, dimOut)).float()
                self.c = torch.zeros(dimOut).float()
                # TEMP @TODO: MOVE THIS TO A BETTER PLACE
                if self.NLBench == 'B2':
                    self.B[1] = 1
                elif self.NLBench in ['B4', 'B5']:
                    self.B[2] = 1
                elif self.NLBench == 'TORA':
                    self.B[3] = 1
                elif self.NLBench == 'ACC':
                    self.B[5] = 2
        self.repetition = 1

    def load(self, path):
        stateDict = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(stateDict)

    def nonLinFunc(self, x, u):
        if self.NLBench == 'B2':
            self.deltaT = 0.1
            x0 = x[:, 0] + self.deltaT * (x[:, 1] - x[:, 0]**3)
            x1 = x[:, 1] + self.deltaT * u[:, 0]
            return torch.stack((x0, x1)).T
        elif self.NLBench == 'B4':
            self.deltaT = 0.05
            x0 = x[:, 0] + self.deltaT * (-x[:, 0] + x[:, 1] - x[:, 2])
            x1 = x[:, 1] + self.deltaT * (-x[:, 0] * (x[:, 2] + 1) - x[:, 1])
            x2 = x[:, 2] + self.deltaT * (-x[:, 0] + u[:, 0])
            return torch.stack((x0, x1, x2)).T
        elif self.NLBench == 'B5':
            self.deltaT = 0.01
            x0 = x[:, 0] + self.deltaT * (x[:, 0]**3 - x[:, 1])
            x1 = x[:, 1] + self.deltaT * (x[:,2])
            x2 = x[:, 2] + self.deltaT * (u[:, 0])
            return torch.stack((x0, x1, x2)).T
        elif self.NLBench == 'TORA':
            self.deltaT = 0.05
            x0 = x[:, 0] + self.deltaT * (x[:, 1])
            x1 = x[:, 1] + self.deltaT * (-x[:, 0] + 0.1*torch.sin(x[:, 2]))
            x2 = x[:, 2] + self.deltaT * (x[:, 3])
            x3 = x[:, 3] + self.deltaT * (u[:, 0])
            return torch.stack((x0, x1, x2, x3)).T
        elif self.NLBench == 'ACC':
            self.deltaT = 0.1
            x0 = x[:, 0] + self.deltaT * (x[:, 1])
            x1 = x[:, 1] + self.deltaT * (x[:, 2])
            x2 = x[:, 2] + self.deltaT * (-4 - 2 * x[:, 2] - x[:, 1]**2 / 1000)
            x3 = x[:, 3] + self.deltaT * (x[:, 4])
            x4 = x[:, 4] + self.deltaT * (x[:, 5])
            x5 = x[:, 5] + self.deltaT * (2 * u[:, 0] - 2 * x[:, 5] - x[:, 4]**2 / 1000)
            return torch.stack((x0, x1, x2, x3, x4, x5)).T    

    def setRepetition(self, repetition):
        self.repetition = repetition

    def forward(self, x):
        x = self.rotation(x)
        if self.isLinear:
            for i in range(self.repetition):
                x = x @ self.A.T + self.Linear(x) @ self.B.T + self.c
        else:
            x = self.nonLinFunc(x, self.Linear(x))
        
        return x


