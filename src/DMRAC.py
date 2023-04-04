import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from Systems import *
from ReplayBuffer import *
import scipy.linalg as sp

import matplotlib.pyplot as plt


class DMRAC():
    def __init__(self, timeStep = 1e-2, nf = 0.1, damping = 1, n_dim = 2, m_dim = 1):
        
        self.layers = [2, 10, 10, 30]
        # last layer parameters; design with assumption that basis are fixed
        
        # self.P = np.eye(n_dim)
        self.B = np.array([0, 1])
        self.Gamma = np.eye(self.layers[-1])*2
        self.timeStep = timeStep
        
        naturalFreq = nf
        damping = damping
        self.feedbackGainP = -(naturalFreq*naturalFreq)
        self.feedbackGainD = -2*damping*naturalFreq
        
        A = np.reshape([0,1,self.feedbackGainP,self.feedbackGainD], (2,2))
        Q = 1*np.eye(2)
        # self.B = np.reshape([0,1],(2,1))
        self.P = sp.solve_lyapunov(A.transpose(),Q)
        
        self.W = np.random.random((self.layers[-1],m_dim))
        self.phi = np.random.random(self.layers[-1]).reshape((self.layers[-1],1))
        
        ## known case(A)
        self.kx = np.array([self.feedbackGainP, self.feedbackGainD]).reshape((2,1))
        self.kr = 1
        
        # internal network parameter

        self.network = Net(self.layers)
        self.buffer = ReplayBuffer(self)
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.network.parameters(), lr = 0.01)


        # -------------------
        self.delta_estimate = 0

    
    def getPhi(self, state):
        with torch.no_grad():
            state = torch.tensor(state.reshape((1,2)), dtype =torch.float)
            # print(state.dtype)
            # return self.network(state).numpy().reshape((1,1))
            return self.network.getPhi(state).detach().numpy().reshape((30,1))




    
    def getCntrl(self, state, ref_state, ref_signal):
        ## linear part
        # print("00")
        e = state - ref_state
        # print(self.phi.shape)
        self.updatePhi(state)
        # print(self.phi.shape)
        w_dot = -np.matmul(self.Gamma, np.matmul(self.phi, np.matmul(e.T, np.matmul(self.P, self.B))))
        w_dot = w_dot.reshape(self.W.shape)
        # print(w_dot.shape,self.W.shape)
        # print("01")
        
        self.W = self.W + self.timeStep*w_dot
        delta_estimate = np.matmul(self.W.T, self.phi)
        self.delta_estimate = delta_estimate
        self.buffer.add(state, delta_estimate)
        u = np.matmul(self.kx.T,state) + ( - self.feedbackGainP)*ref_signal  - delta_estimate

        if (self.buffer.getSize()%100 == 0 and self.buffer.getSize() > 0 ):
            self.updateNetwork()
            # update network
            
        return u

    def updateNetwork(self):
        print("update Called-------------------")
        s_batch, d_batch = self.buffer.sample_batch()

        s_batch = torch.tensor(s_batch).float()
        d_batch = torch.tensor(d_batch).float()


        self.network.updateLastLayer(self.W.T)

        outs = self.network(s_batch)
        loss = self.criterion(outs, d_batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def updatePhi(self, state):
        self.phi = self.getPhi(state)




        
class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()

        
        layer =[]
        for i in range(len(layers)-1):
            layer.append(nn.Linear(layers[i],layers[i+1]))

        self.linear = nn.Sequential(*layer)
        self.last = nn.Linear(layers[-1], 1)

    def forward(self, x):
        return self.last(self.linear(x)) ## make sure of this

    def getPhi(self,x):
        return self.linear(x)

    def updateLastLayer(self, WT):
        i = 0
        for params in self.parameters():
            i = i+1
            if i == 7:
                params.data = nn.parameter.Parameter(torch.tensor(WT)).float()


if __name__ == "__main__":
    model = Net([2, 10, 10, 30])
    # for params in model.parameters():
    print(model.state_dict()['last.weight'])
    i = 0 
    for params in model.parameters():
        i = i + 1
        if i == 7 :
            params.data = nn.parameter.Parameter(torch.tensor(np.random.random((1,30))))
            print(params.shape )
        print(i)
        print(params.shape,"-")






        
