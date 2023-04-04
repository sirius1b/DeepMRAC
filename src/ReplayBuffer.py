import numpy as np
from collections import deque
import random
import math

class ReplayBuffer():
	def __init__(self, agent, gamma = 1e-4, batch_size= 50, size = 2000):

	
		self.pairs = deque()
		self.size = size
		self.batch_size = batch_size
		self.agent = agent
		self.threshold = gamma

		# self.agent = agent


	def simpleAdd(self, state, delta):
		self.pairs.append((state.reshape(len(state)), delta))
		if (len(self.pairs) > self.size):
			self.pairs.popleft()

	def add(self, state, delta):
		# self.simpleAdd(state, delta)
		self.smartAdd(state, delta)
		print(self.getSize())


	def smartAdd(self, state, delta):
		if (len(self.pairs)> 1):
			state_p = self.pairs[-1][0]
			phi_i = self.agent.getPhi(state)
			phi_p = self.agent.getPhi(state_p)

			gamma = math.pow(np.linalg.norm(phi_i - phi_p),2)/np.linalg.norm(phi_i)
			print(gamma)
			if (gamma >= self.threshold):
				if (len(self.pairs) < self.size):
					self.pairs.append((state.reshape(len(state)), delta))
				else:
					self.pairs.popleft()
		else:
			self.pairs.append((state.reshape(len(state)), delta))









	def sample_batch(self):
		batch = random.sample(self.pairs, min(len(self.pairs), self.batch_size))
		s_batch = np.array([_[0] for _ in batch])
		d_batch = np.array([_[1] for _ in batch])

		return s_batch, d_batch

	def getSize(self):
		return len(self.pairs)





		



