import config
import utils
import q_learner

import math
import os.path
import pygame as pg
import random

class Agent():

	def __init__(self, pos, radius, color):
		self.old_pos = pos
		self.new_pos = pos
		self.pos = pos
		self.radius = radius
		self.color = color

		self.interp_timer = utils.Timer(config.STEP_TIME, True)

	def update(self, deltatime):
		if config.STEP_TIME > 0.2:
			self.interp_timer.update(deltatime)
			self.pos = utils.interp_point(
					self.old_pos, 
					self.new_pos, 
					self.interp_timer.get_progress(),
					utils.cos_interp)
		else:
			# optimization for small step times
			self.pos = self.new_pos

	def get_int_pos(self):
		return (int(self.new_pos[0]), 
				int(self.new_pos[1]))

	def cell_is_allowed(self, cell):
		return True

	def move_to(self, pos):
		(x, y) = pos
		if pos != self.new_pos and \
		   self.cell_is_allowed(pos) and \
		   x >= 0 and x < config.GRID_W and \
		   y >= 0 and y < config.GRID_H:
			   self.old_pos = self.pos
			   self.new_pos = pos
			   self.interp_timer.reset()

	def render(self, screen):
		rad = self.radius * min(config.CELL_W, config.CELL_H)
		pos = utils.to_screen(self.pos)

		pg.draw.circle(screen, self.color, pos, int(rad))

class VIP(Agent):

	def __init__(self, pos):
		super(VIP, self).__init__(
				pos, 0.5, (0, 255, 255))

		self.move_timer = utils.Timer(config.VIP_EPISODE * config.STEP_TIME)

	def update(self, deltatime):
		super(VIP, self).update(deltatime)

		self.move_timer.update(deltatime)
		if config.VIP_STATE == config.VIPState.AUTO and self.move_timer.is_finished():
			self.move_timer.reset()
			(x, y) = self.pos
			self.move_to((x + random.randint(-1, 1),
						  y + random.randint(-1, 1)))

class QAgent(Agent):

	last_s = None
	def __init__(self, pos, radius, color, controller = None, can_suffer = False):
		super(QAgent, self).__init__(
				pos, radius, color)

		self.reward_monitor = utils.ValueMonitor()
		self.controller = controller
		self.can_suffer = can_suffer
		self.move_timer = utils.Timer(config.STEP_TIME)


	def randomize(self):
		self.move_to(utils.randcell())
		#self.pos = (x, y)
		#self.old_pos = (x, y)
		#self.new_pos = (x, y)

	'''
	Attach a graphing function for rewards

	@param graph_func a function: Value -> Void
	'''
	def attach_rewards_graph(self, graph_func):
		self.reward_monitor.set_graph_func(
				lambda mon: graph_func(mon.get_recent_average()))

	def is_terminal_state(self, s):
		return False

	def get_state(self, pos):
		raise NotImplementedError

	def get_my_state(self):
		return self.get_state(self.get_int_pos())

	def get_reward(self, s):
		raise NotImplementedError

	def do_action(self, a):
		raise NotImplementedError

	def get_average_reward(self):
		return self.reward_monitor.get_recent_average()

	def get_iteration_count(self):
		return self.reward_monitor.get_count()

	def update(self, deltatime):
		super(QAgent, self).update(deltatime)

		self.move_timer.update(deltatime)
		if self.controller is not None and self.move_timer.is_finished():
			self.move_timer.reset()

			s = self.get_my_state()

			# get action from controller
			a = self.controller.get_action(s)
			# do that action
			self.do_action(a)
			# get new state and reward
			s_ = self.get_my_state()
			r = self.get_reward(s_)
			self.reward_monitor.update(r)

			# add suffering factor for data
			if self.can_suffer:
				r -= (config.SUFFERING - 26)

			if self.is_terminal_state(s_):
				# terminal state
				#self.randomize()
				# notify controller
				self.controller.terminate_trajectory(s, a, r)
				#return
			else:
				self.controller.update_trajectory(s, a, r, s_)

	def get_superpos_qs(self, cell_pos):
		return self.controller.get_action_qs(self.get_state(cell_pos))

	def dump(self, filename):
		self.controller.dump(filename)

def threat_level(vip_pos, guard_pos, hostile_pos):

	tv = utils.sub(hostile_pos, vip_pos)
	gv = utils.sub(guard_pos, vip_pos)


	if utils.norm2(tv) == 0:
		return 0

	dst_threat =  70 * math.exp(-math.sqrt(utils.dst2(
			vip_pos, hostile_pos)))

	coverage = 0
	if utils.norm2(gv) != 0 and utils.norm2(gv) < utils.norm2(tv):
		coverage = 10 * max(utils.dot(tv, gv), 0) / \
				math.sqrt(utils.norm2(tv) * utils.norm2(gv))

	return dst_threat - coverage

class Guard(QAgent):

	def __init__(self, pos, vip, hostile, use_saved_data = True, 
			controller = None, is_ghost = False):
		if controller is None:
			# state space:  (x, y, vip_x, vip_y, hostile_x, hostile_y)
			# action space: (dx, dy)
			controller = q_learner.QController(
				(config.GRID_W, config.GRID_H, 
				 config.GRID_W, config.GRID_H,
				 config.GRID_W, config.GRID_H), (4,),
				 load_file = config.GUARD_Q_FILE if \
				 use_saved_data and os.path.isfile(config.GUARD_Q_FILE) else None,
				 gamma = 0.2,
				 exploration = 0)

		super(Guard, self).__init__(
				pos, 0.4, 
				(0, 255, 0) if not is_ghost else (200, 255, 200), 
				controller = controller,
				can_suffer = True) 

		self.vip = vip
		self.hostile = hostile

	def create_ghost(self, pos):
		return Guard(pos, self.vip, self.hostile, 
				controller = q_learner.QController(
					linked_controller = self.controller,
					exploration = config.GHOST_EXPLORATION,
					follow_reward = config.GHOST_FOLLOW_REWARD),
					is_ghost = True)

	def get_state(self, pos):
		return pos + \
			   self.vip.get_int_pos() + \
			   self.hostile.get_int_pos()

	def get_reward(self, s):
		vip_dst2 = utils.dst2(
				self.vip.get_int_pos(),
				self.get_int_pos())

		return -1 - threat_level(
				self.vip.get_int_pos(),
				self.get_int_pos(),
				self.hostile.get_int_pos()) - \
				10 * int(vip_dst2 > 4 or vip_dst2 <= 0)

	def do_action(self, a):
		(dx, dy) = utils.CARDINALS[a[0]]
		(x, y) = self.new_pos
		self.move_to((x + dx, y + dy))

class Hostile(QAgent):

	def __init__(self, pos, vip, guard, use_saved_data = True, 
			controller = None, is_ghost = False):
		if controller is None:
			# state space:	(x, y, vip_x, vip_y, guard_x, guard_y)
			# action space: (dx, dy)
			controller = q_learner.QController(
				(config.GRID_W, config.GRID_H, 
				 config.GRID_W, config.GRID_H,
				 config.GRID_W, config.GRID_H), (4,),
				 load_file = config.HOSTILE_Q_FILE if \
				 use_saved_data and os.path.isfile(config.HOSTILE_Q_FILE) else None,
				 gamma = 0.8,
				 exploration = 0.4)

		super(Hostile, self).__init__(
				pos, 0.4, 
				(255, 0, 0) if not is_ghost else (255, 200, 200), 
				controller) 

		self.vip = vip
		self.guard = guard

	def create_ghost(self, pos):
		return Hostile(pos, self.vip, self.guard, 
				controller = q_learner.QController(
					linked_controller = self.controller,
					exploration = config.GHOST_EXPLORATION,
					follow_reward = config.GHOST_FOLLOW_REWARD),
				is_ghost = True)

	def get_state(self, pos):
		return pos + \
			   self.vip.get_int_pos() + \
			   self.guard.get_int_pos()

	def get_reward(self, s):
		return -1 + threat_level(
				self.vip.get_int_pos(),
				self.guard.get_int_pos(),
				self.get_int_pos())

	def cell_is_allowed(self, cell):
		dst2 = utils.dst2(cell, self.vip.get_int_pos())
		return dst2 > config.HOSTILE_CLOSEST_DST2

	def do_action(self, a):
		(dx, dy) = utils.CARDINALS[a[0]]
		(x, y) = self.new_pos
		self.move_to((x + dx, y + dy))



