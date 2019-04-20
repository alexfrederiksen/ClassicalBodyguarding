import pickle
import random
import numpy as np

class QController():
	gamma = 0.1
	exploration = 0.1


	def __init__(self, state_size = 0, action_size = 0, linked_controller = None, 
			load_file = None, gamma = None, exploration = None, 
			follow_reward = True):
		self.gamma = gamma
		self.exploration = exploration
		self.follow_reward = follow_reward
		self.state_size = state_size
		self.action_size = action_size
		if load_file is not None:
			# load table from file
			self.load(load_file)

		elif linked_controller is not None:
			# use the shared table
			self.state_size = linked_controller.state_size
			self.action_size = linked_controller.action_size
			if gamma is None: self.gamma = linked_controller.gamma
			if exploration is None: self.exploration = linked_controller.exploration
			self.q_table = linked_controller.q_table

		else:
			# create Q table
			self.q_table = np.zeros(state_size + action_size)

		if self.gamma is None:
			self.gamma = 0.1
		if self.exploration is None:
			self.exploration = 0.1



	""" 
	Computes action with e-greedy algorithm

	@param s state 
	"""
	def get_action(self, s):
		# update the Q table
		if random.random() > self.exploration:
			qs = self.q_table[s]

			if self.follow_reward:
				# choose best action
				a = random.choice(np.argwhere(qs == np.amax(qs)))
			else:
				# choose worst action
				a = random.choice(np.argwhere(qs == np.amin(qs)))

			return tuple(a)
		else:
			# choose random action
			a = [random.randint(0, b - 1) for b in self.action_size]
			return tuple(a)

	"""
	Updates Q table with trajectory

	@param s  state
	@param a  action done in state "s"
	@param r  reward for doing action "a" in state "s"
	@param s_ new state after doing action "a" in state "s"
	"""
	def update_trajectory(self, s, a, r, s_):
		self.q_table[s + a] = r + self.gamma * np.max(self.q_table[s_])

	"""
	Updates Q table with a terminal value

	@param s state
	@param a action done in state "s"
	@param r reward for doing action "a" in state "s"
	"""
	def terminate_trajectory(self, s, a, r):
		self.q_table[s + a] = r

	def get_action_qs(self, s):
		return self.q_table[s]

	def dump(self, filename):
		print(f"Dumping Q table to \"{filename}\"...")
		with open(filename, "wb") as fp:
			pickle.dump(self.q_table, fp)

	def load(self, filename):
		print(f"Loading Q table from \"{filename}\"...")
		with open(filename, "rb") as fp:
			self.q_table = pickle.load(fp)

