import config
import agent
import utils
import q_learner

import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget 
from PyQt5.QtGui import QGridLayout
from PyQt5.QtCore import QTimer

import os.path
import pygame as pg
import numpy as np
import random


class World():

	def __init__(self, main_window, use_saved_data = False):
		self.use_saved_data = use_saved_data
		self.vip = agent.VIP((config.GRID_W / 2, config.GRID_H / 2))

		self.guard = agent.Guard(
			pos = (0, 0), 
			vip = self.vip, 
			hostile = None, 
			use_saved_data = use_saved_data)

		self.ghost_guards = []

		self.hostile = agent.Hostile(
			pos = (config.GRID_W - 1, config.GRID_H - 1), 
			vip = self.vip, 
			guard = self.guard, 
			use_saved_data = use_saved_data)

		self.ghost_hostiles = []

		# hostile needed to be created before giving it to the guard
		self.guard.hostile = self.hostile
		
		ghost_count = config.GHOST_COUNT
		config.GHOST_COUNT = 0
		self.set_ghost_count(ghost_count)

		if config.RENDER_ENABLED:
			self.font = pg.font.SysFont("Hack", 12)

		# get rewards graph
		self.rewards_graph = main_window.rewards_graph
		# set graph callbacks for agents
		self.guard.attach_rewards_graph(
				lambda val: self.rewards_graph.add_val(0, val))
		self.hostile.attach_rewards_graph(
				lambda val: self.rewards_graph.add_val(1, val))

	def mouse_vip(self, mouse_pos):
		self.vip.move_to(utils.to_world(mouse_pos))

	def set_ghost_count(self, count):
		count = max(0, count)
		current_count = config.GHOST_COUNT

		if count > current_count:
			# add ghosts
			for i in range(count - current_count):
				self.ghost_guards.append(
					self.guard.create_ghost(utils.randcell()))
				self.ghost_hostiles.append(
					self.hostile.create_ghost(utils.randcell()))

		elif count < current_count:
			# remove ghosts
			for i in range(current_count - count):
				del self.ghost_guards[-1]
				del self.ghost_hostiles[-1]

		config.GHOST_COUNT = count

	def update(self, deltatime):
		hostile_rewards = []
		guard_rewards = []

		for ghost in self.ghost_hostiles:
			ghost.update(deltatime)

		for ghost in self.ghost_guards:
			ghost.update(deltatime)

		self.hostile.update(deltatime)
		self.guard.update(deltatime)

		self.vip.update(deltatime)

		hostile_reward = self.hostile.get_average_reward()
		guard_reward = self.guard.get_average_reward()

		#print(f"rewards: hostile = {hostile_reward} guard = {guard_reward}")

		# end program if episode count is given
		if config.ITERATION_MAX > 0 and \
			self.guard.get_iteration_count() >= config.ITERATION_MAX:
			return True

		return False

	def get_cell_text(self, cell_pos):
		qs = self.guard.get_superpos_qs(cell_pos)
		return "{:.3}\n{:.3}\n{:.3}\n{:.3}\n".format(
			qs[0], qs[1], qs[2], qs[3])

	def render_cell_text(self, screen, cell_pos, text):
		(x, y) = cell_pos
		lines = text.splitlines()
		h = self.font.get_linesize()
		for i, line in enumerate(lines):
			text_surface = self.font.render(line, True, (0, 0, 0))
			(nx, ny) = utils.to_screen((x - 0.5, y - 0.5))
			screen.blit(text_surface, (nx, ny + i * h))

	def render_grid_text(self, screen):
		for x in range(config.GRID_W): 
			for y in range(config.GRID_H):
				self.render_cell_text(screen, (x, y),
					self.get_cell_text((x, y)))
	
	def render_grid(self, screen):
		for x in range(config.GRID_W):
			for y in range(config.GRID_H):
				pos = utils.to_screen((x, y))
				pg.draw.circle(screen, (100, 100, 100), pos, 5)

	def render(self, screen):
		self.render_grid(screen)

		self.vip.render(screen)

		if config.RENDER_GHOSTS_ENABLED:
			for ghost in self.ghost_guards:
				ghost.render(screen)

		if config.RENDER_GHOSTS_ENABLED:
			for ghost in self.ghost_hostiles:
				ghost.render(screen)

		self.guard.render(screen)
		self.hostile.render(screen)

		if config.RENDER_TEXT_ENABLED:
			self.render_grid_text(screen)


	def on_mouse_move(self, mouse_pos):
		if config.VIP_STATE == config.VIPState.MOUSE: 
			self.mouse_vip(mouse_pos)

	def on_number_pressed(self, number):
		self.set_ghost_count(number * config.GHOST_COUNT_INTERVAL)

	def on_key_pressed(self, key):
		if key == pg.K_RETURN:
			# toggle mouse control
			config.VIP_STATE = config.VIPState((config.VIP_STATE + 1) % 3)
			print(f"VIP_STATE is now {config.VIP_STATE.name}.")

		elif key == pg.K_g:
			# toggle ghost display
			config.RENDER_GHOSTS_ENABLED ^= True

		elif key == pg.K_q:
			config.RENDER_TEXT_ENABLED ^= True

	def get_fitness(self):
		return self.guard.reward_monitor \
			.get_cumulative_average()

	def on_close(self):
		#plt.show(self.rewards_graph.p)
		# print stats
		guard_mon = self.guard.reward_monitor
		guard_avg = guard_mon.get_cumulative_average()
		guard_sum = guard_mon.get_sum()

		hostile_mon = self.hostile.reward_monitor
		hostile_avg = hostile_mon.get_cumulative_average()
		hostile_sum = hostile_mon.get_sum()
		
		iterations = self.guard.reward_monitor.get_count()
		
		
		print(f"\n\nStatistics over {iterations} iterations \n\n"
			  f"  Guard: \n"
			  f"      average reward: {guard_avg} \n"
			  f"      total reward:   {guard_sum} \n\n"
			  f"  Hostile: \n"
			  f"      average reward: {hostile_avg} \n"
			  f"      total reward:   {hostile_sum} \n\n\n")

		if self.use_saved_data:
			self.hostile.dump(config.HOSTILE_Q_FILE)
			self.guard.dump(config.GUARD_Q_FILE)

class PygameWindow():

	def __init__(self):

		if config.RENDER_ENABLED:
			pg.init()
			pg.display.set_caption("Bodyguarding")
			self.screen = pg.display.set_mode(
				(config.SCREEN_W, config.SCREEN_H), pg.RESIZABLE)

			self.clock = pg.time.Clock()
			self.clock.tick()

		self.deltatime = 0
		self.world = None

	def run_world(self, world):
		self.world = world

	def on_resize(self, w, h):

		config.SCREEN_W = w
		config.SCREEN_H = h
		config.CELL_W = config.SCREEN_W / config.GRID_W
		config.CELL_H = config.SCREEN_H / config.GRID_H
		print(f"{w} x {h}")


	def update_no_render(self):
		world_running = not self.world.update(10)

		if not world_running:
			self.world.on_close()
			self.world = None

		return True, world_running

	def update(self):
		if self.world is None: return True, False

		if not config.RENDER_ENABLED:
			return self.update_no_render()

		running = True
		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False

			if event.type == pg.VIDEORESIZE:
				# resize window
				screen = pg.display.set_mode(
					(event.w, event.h), pg.RESIZABLE)
				self.on_resize(event.w, event.h)

			if event.type == pg.MOUSEMOTION:
				self.world.on_mouse_move(pg.mouse.get_pos())

			if event.type == pg.KEYUP:
				if event.key == pg.K_ESCAPE:
					running = False

				num = event.key - pg.K_0
				if num >= 0 and num <= 9:
					self.world.on_number_pressed(num)

				self.world.on_key_pressed(event.key)

		# update world
		world_running = not self.world.update(self.deltatime)
		# clear canvas
		self.screen.fill((255, 255, 255))
		# render world
		self.world.render(self.screen)

		pg.display.flip()

		self.deltatime = self.clock.tick(config.TARGET_FPS) / 1000

		if not (running and world_running):
			self.world.on_close()
			self.world = None

		return running, world_running

class WorldTester:
	
	'''
	@param set_param a function Float -> Void
	@param graph_fitness a function Float -> Float -> Void
	'''
	def __init__(self, set_param, graph_fitness, start, 
			step, end, default_param = 0):
		self.set_param = set_param
		self.graph_fitness = graph_fitness
		self.param_val = start
		self.step = step 
		self.end = end
		self.default_param = default_param

		self.world = None

	def next_world(self, main_window):

		if self.world is not None:
			self.graph_fitness(self.param_val - self.step, 
					self.world.get_fitness())

		if self.param_val <= self.end:
			self.set_param(self.param_val)
			# create new world
			self.world = World(main_window, use_saved_data = False)

			self.param_val += self.step
			return self.world
			
		# reset to default param
		self.set_param(self.default_param)
		# there is no next world!
		return None

class WorldTesterChain:

	def __init__(self, chain):
		self.chain = chain

	def next_world(self, main_window):
		if len(self.chain) <= 0: return None

		while len(self.chain) > 0:
			# get next world from top tester
			world = self.chain[0].next_world(main_window)
			if world is not None:
				return world
			else:
				# pop top tester off the chain
				self.chain = self.chain[1:]

		return None

class MainWindow(QMainWindow):

	def __init__(self, parent = None):
		super().__init__()

		self.title = "Graphs and Stuff"
		self.width = 1200
		self.height = 900

		self.setWindowTitle(self.title)
		self.setGeometry(0, 0, self.width, self.height)

		self.layout = QGridLayout()
		self.layout.setContentsMargins(10, 10, 10, 10)
		self.layout_widget = QWidget()
		self.layout_widget.setLayout(self.layout)
		self.setCentralWidget(self.layout_widget)

		self.rewards_graph = utils.LiveGraph(
				title = "Guard vs. Hostile", 
				subgraph_count = 2, 
				sample_efficiency = 0.05,
				parent = self)
		self.suffer_graph = utils.LiveGraph(
				"Performance by Suffering", 1, self)
		self.ghost_graph = utils.LiveGraph(
				"Performance by Ghosting", 1, self)
		self.exploration_graph = utils.LiveGraph(
				"Performance by Ghost Exploration", 1, self)

		self.layout.addWidget(self.rewards_graph.widget, 0, 0)
		self.layout.addWidget(self.exploration_graph.widget, 0, 1)
		self.layout.addWidget(self.suffer_graph.widget, 1, 0)
		self.layout.addWidget(self.ghost_graph.widget, 1, 1)

		# create testing chain
		self.tester = WorldTesterChain([
			WorldTester(config.set_suffering,
					lambda p, r: self.suffer_graph.add_point(0, p, r),
					0, 2, 20, config.SUFFERING),
			WorldTester(config.set_ghost_exploration,
					lambda p, r: self.exploration_graph.add_point(0, p, r),
					0, 0.1, 1, config.GHOST_EXPLORATION),
			WorldTester(config.set_ghost_count, 
					lambda p, r: self.ghost_graph.add_point(0, p, r),
					0, 20, 200, config.GHOST_COUNT)
			])

		# create pygame window
		self.pg_window = PygameWindow()

		#self.pg_window.run_world(self.tester.next_world(self))
		self.pg_window.run_world(World(self))
		
		self.show()

		self.is_closed = False

		# setup pygame update loop
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_pygame)
		self.timer.start(0)


	def update_pygame(self):
		pg_is_running, world_is_running = self.pg_window.update()
		
		
		if not (pg_is_running and world_is_running):
			
			if pg_is_running:
				world = self.tester.next_world(self)
				if world is not None:
					self.pg_window.run_world(world)
					return

			
			self.close()

	def close(self):
		if not self.is_closed: self.on_close()
		self.is_closed = True
		super().close()

	def on_close(self):
		# dump graph data
		self.rewards_graph.dump("reward.gph")
		self.suffer_graph.dump("suffer.gph")
		self.ghost_graph.dump("ghost.gph")
		self.exploration_graph.dump("exploration.gph")

def main():

	# create window 
	app = QApplication(sys.argv)
	window = MainWindow()
	# run the app
	app.exec_()

	window.close()

def test_shit():

	config.GHOST_EXPLORATION = 0.5
	config.ITERATION_MAX = 1000

	config.GHOST_FOLLOW_REWARD = False
	world = World(
		ghost_count = 100,
		use_saved_data = False)
	run_world(world, screen)

	config.GHOST_FOLLOW_REWARD = True
	world = World(
		ghost_count = 100,
		use_saved_data = False)
	run_world(world, screen)


	return 

	for i in range(6):
		print(f"Running with ghost exploration = {config.GHOST_EXPLORATION}...")

		world = World(
			use_saved_data = False,
			ghost_count = 100)

		end = run_world(world, screen)
		if end: break

		config.GHOST_EXPLORATION += 0.2

if __name__ == "__main__":
	main()

