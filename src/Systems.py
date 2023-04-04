import numpy as np


class Model():
    def __init__(self, init_state, timeStep = 1e-2):
        self.state = init_state
        self.timeStep = timeStep
        self.trueWeights = np.array([0.2314, 0.06918, -0.6245, 0.0095, 0.0214])
        # self.trueWeights = np.array([0, 0, 0, 0, 0])
        self.lDelta = 1
        self.substeps = 1

        self.A = np.array([0, 1, 0, 0]).reshape((2,2))
        self.B = np.array([0, self.lDelta]).reshape((2,1))

        self.delta = 0

    def dynamics(self, state, action):
        delta = self.trueWeights[0]*state[0] + self.trueWeights[1]*state[1] + self.trueWeights[2]*np.abs(state[0])*state[1] + self.trueWeights[3]*np.abs(state[1])*state[1] + self.trueWeights[4]*state[1]**3


        self.delta = delta
        # delta  = 0
        action = np.array(action).reshape((1,1))
        delta = np.array(delta).reshape((1,1))

        xdot = (self.A@state + self.B@(action + delta)).reshape((2,1))

        return xdot

    def simModel(self, action):
        xstep = self.euler(self.dynamics, self.state, action, self.timeStep, self.substeps)
        return xstep

    def applyCntrl(self, action):
        self.state = self.simModel(action)

    def euler(self, sys, state, action, timeStep, substeps):
        k1 = sys(state, action)
        xstep = state + timeStep*k1
        return xstep


class RefModel():
    def __init__(self, init_state, timeStep = 1e-2, nf = 0.1, damping =1):
        self.state = init_state
        self.timeStep = timeStep
        self.naturalFreq = nf
        self.damping = damping
        self.lDelta = 1
        self.substeps = 1

        self.Am = np.array([0 , 1, -self.naturalFreq*self.naturalFreq, -2*self.damping*self.naturalFreq ]).reshape((2,2))
        self.Bm = np.array([0 , self.naturalFreq*self.naturalFreq]).reshape((2,1))



    def dynamics(self, state, ref_signal):
        ref_signal = np.array(ref_signal).reshape((1,1))
        # print(self.Am.shape, self.Bm.shape, state.shape, ref_signal.shape,"--")
        xdot = (self.Am@state  + self.Bm @ ref_signal).reshape((2,1))
        return xdot

    def simModel(self, action):
        xstep = self.euler(self.dynamics, self.state, action, self.timeStep, self.substeps)
        return xstep

    def applyCntrl(self, action):
        self.state = self.simModel(action)

    def euler(self, sys, state, action, timeStep, substeps):
        k1 = sys(state, action)
        xstep = state + timeStep*k1
        return xstep




