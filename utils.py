import config

import pickle
import time
import random
import math
import collections
import pyqtgraph as plt
import numpy as np

CARDINALS = [
		( 0, -1), # up
		( 0,  1), # down
		(-1,  0), # left
		( 1,  0)] # right

class Timer:

	def __init__(self, length, finished = False):
		self.length = length
		self.elapse = length if finished else 0

	def update(self, deltatime):
		self.elapse += deltatime
		if self.is_finished(): self.elapse = self.length

	def get_progress(self):
		return self.elapse / self.length

	def is_finished(self):
		return self.elapse >= self.length

	def reset(self):
		self.elapse = 0

class GraphData:
	
	def __init__(self, x_values, y_values):
		self.x_values = x_values
		self.y_values = y_values

	def dump(self, filename):
		print(f"Dumping GraphData to \"{filename}\"...")
		with open(filename, "wb") as f:
			pickle.dump(self, f)

	def load(filename):
		print(f"Loading GraphData from \"{filename}\"...")
		with open(filename, "rb") as f:
			data = pickle.load(f)
			#if data is not GraphData:
			#	print(f"\"{filename}\" does not contain GraphData.")
			#	return None

			return data

class LiveGraph:

	def __init__(self, title, subgraph_count, parent, sample_efficiency = 1):
		self.subgraph_count = subgraph_count
		self.times = [0 for _ in range(subgraph_count)]

		# counts calls made to @add_point
		self.call_counts =[0 for _ in self.times]
		# listen to every @listen_interval call
		self.listen_interval = 1 / sample_efficiency

		self.points_x = [[] for _ in self.times]
		self.points_y = [[] for _ in self.times]


		#plt.setConfigOption('background', 'w')
		#plt.setConfigOption('foreground', 'k')
		
		self.widget = plt.PlotWidget(parent, title = title)
		self.curves = []
		for i in range(subgraph_count):
			my_pen = plt.mkPen(plt.intColor(i + 1, subgraph_count), width = 3)
			self.curves.append(
					self.widget.getPlotItem().plot(pen = my_pen))



	def add_point(self, index, x, y):
		self.call_counts[index] += 1
		if self.call_counts[index] < self.listen_interval: return
		self.call_counts[index] -= self.listen_interval

		self.points_x[index].append(x)
		self.points_y[index].append(y)

		self.curves[index].setData(
				self.points_x[index], self.points_y[index])

	def add_val(self, index, y):
		self.times[index] += 1

		self.add_point(index, self.times[index], y)

	def dump(self, filename):
		for i in range(self.subgraph_count):
			data = GraphData(self.points_x[i], self.points_y[i])
			# split word at . to insert subgraph index
			words = filename.split(".", 1)
			data.dump("{}_{}.{}".format(words[0], i, words[1]))
		
class ValueMonitor:

	'''
	@param graph_func a function: Self -> Void
	'''
	def __init__(self, average_size = 100, graph_func = None):
		self.average_size = average_size
		self.buffer = collections.deque(maxlen = average_size)
		self.average = 0
		self.sum = 0
		self.value_count = 0
		self.graph_func = graph_func

	def update(self, new_value):
		self.buffer.append(new_value)
		self.value_count += 1
		# update cumulative average
		self.average = ((self.average * (self.value_count - 1)) + new_value) \
			/ self.value_count
		# update sum
		self.sum += new_value

		if self.graph_func is not None:
			self.graph_func(self)

	def set_graph_func(self, graph_func):
		self.graph_func = graph_func

	def get_recent_average(self):
		return 0 if len(self.buffer) == 0 else \
			sum(self.buffer) / len(self.buffer)

	def get_cumulative_average(self):
		return self.average

	def get_sum(self):
		return self.sum

	def get_count(self):
		return self.value_count


def lin_interp(a, b, alpha):
	return a + (b - a) * alpha

def cos_interp(a, b, alpha):
	return a + (b - a) * (0.5 * (-math.cos(math.pi * alpha) + 1))

def quad_interp(a, b, alpha):
	pass

def interp_point(a, b, alpha, func):
	return (lin_interp(a[0], b[0], func(0, 1, alpha)),
			lin_interp(a[1], b[1], func(0, 1, alpha)))

def foreach_2d(w, h, func):
	for x in range(w):
		for y in range(h):
			func(x, y)

def to_screen(x):
	return (int((x[0] + 0.5) * config.CELL_W),
			int((x[1] + 0.5) * config.CELL_H))

def to_world(x):
	return (int(x[0] / config.CELL_W),
			int(x[1] / config.CELL_H))

def randcell():
	return (random.randint(0, config.GRID_W - 1),
			random.randint(0, config.GRID_H - 1))

def sub(p1, p2):
	return (p1[0] - p2[0],
			p1[1] - p2[1])

def dst2(p1, p2):
	return norm2(sub(p2, p1))

def norm2(p):
	return dot(p, p)

def dot(p1, p2):
	return p1[0] * p2[0] + p1[1] * p2[1]
