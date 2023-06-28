from packages import *


class NeuralNetwork(nn.Module):
    def __init__(self, path, A=None, B=None, c=None, activation='softplus', loadOrGenerate=True):
        super().__init__()
        self.activation = activation
        if activation == 'softplus':
            activationF = nn.Softplus()
        elif activation == 'sigmoid':
            activationF = nn.Sigmoid()
        elif activation == 'tanh':
            activationF = nn.Tanh()
        else:
            activationF = nn.ReLU()

        if loadOrGenerate:
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
            self.A = torch.zeros((dimOut, dimInp)).float()
            self.B = torch.eye(dimOut, dimOut).float()
            self.c = torch.zeros(dimOut).float()
        self.repetition = 1

    def load(self, path):
        stateDict = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(stateDict)

    def setRepetition(self, repetition):
        self.repetition = repetition

    def forward(self, x):
        x = self.rotation(x)
        for i in range(self.repetition):
            x = x @ self.A.T + self.Linear(x) @ self.B.T + self.c
        return x


