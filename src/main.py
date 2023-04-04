import numpy as np
# import torch
# import torch.nn.functional as F
from Systems import *
from DMRAC import *
import scipy.linalg as sp


import matplotlib.pyplot as plt


init_state = np.array([0, 0]).reshape((2,1))



endTime = 20
startTime = 0
timeStep = 1e-2
nf = 3
damping = 1

model = Model(init_state, timeStep = timeStep)
refModel = RefModel(init_state, timeStep = timeStep, nf = nf , damping = damping)
agent = DMRAC(timeStep=timeStep, nf = nf, damping = damping)


time_grid = np.arange(startTime, endTime, timeStep)
refSignal = np.sin(time_grid)


ref_state_rec = []
state_rec = []
cntrl_rec = []
error_rec = []
true_delta_rec = []
estimate_delta_rec = []


for i in range(len(time_grid)):
    print(i)
    state = model.state
    ref_state = refModel.state
    ref_sig = refSignal[i]
    true_delta = float(model.delta)
    estimate_delta = float(agent.delta_estimate)

   
    action = agent.getCntrl(state, ref_state, ref_sig)
    model.applyCntrl(action)
    refModel.applyCntrl(ref_sig)
    
    state_rec.append(state)
    ref_state_rec.append(ref_state)
    cntrl_rec.append(action)
    error_rec.append(state - ref_state)
    true_delta_rec.append(true_delta)
    estimate_delta_rec.append(estimate_delta)



plt.subplots_adjust(top=0.95,
bottom=0.08,
left=0.065,
right=0.97,
hspace=0.33,
wspace=0.205)
plt.subplot(2,2,1)
plt.plot(time_grid,[_[0] for _ in state_rec ])
plt.plot(time_grid, [_[0] for _ in ref_state_rec ])
plt.legend(['s','r'])
plt.title("Roll")
plt.ylabel("Roll(rad)")
plt.xlabel('Time(s)')

plt.subplot(2,2,2)
plt.plot(time_grid,[_[1] for _ in state_rec ])
plt.plot(time_grid, [_[1] for _ in ref_state_rec ])
plt.legend(['s','r'])
plt.title("Roll Rate")
plt.ylabel("Roll(\dot{rad})")
plt.xlabel('Time(s)')

plt.subplot(2,2,3)
plt.plot(time_grid, [_[0] for _ in cntrl_rec])
plt.title("Control Effort")
plt.ylabel("Magnitude")
plt.xlabel('Time(s)')

plt.subplot(2,2,4)
plt.plot(time_grid, true_delta_rec)
plt.plot(time_grid, estimate_delta_rec)
plt.legend(["true","estimate"])
plt.title("Uncertainity")
plt.ylabel("Magnitude")
plt.xlabel('Time(s)')

plt.savefig('plot_2.png',dpi = 300, bbox_inches="tight")




plt.show()