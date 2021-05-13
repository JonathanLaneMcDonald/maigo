
import numpy as np
from numpy.random import choice
from copy import deepcopy

from board import Board
from datasets import game_state_to_model_inputs

class MCTS:

	class Node:
		def __init__(self, game_state, completed_move, player_to_move, policy_score, parent):
			self.parent = parent
			self.children = {}	#{move:Node(...)}

			self.completed_move = completed_move
			self.player_to_move = player_to_move

			self.value_score = 0
			self.simulations = 0
			self.policy_score = policy_score

			self.game_state = game_state

	def __init__(self, game_state, model=None):
		# set up nodes and model
		self.game_root = MCTS.Node(game_state, None, 1, 0, None)
		self.play_root = self.game_root
		self.model = model

		# working toward facilitating multiprocess communication - can't send objects, so i'll send unique identifiers
		self.lut = {0:self.game_root}
		self.rlut = {self.game_root:0}

		# set up game root
		self.expand(self.game_root)

	def get_node_count(self):
		return len(self.lut)

	def get_final_score(self):
		return self.play_root.game_state.get_simple_terminal_score_and_ownership()

	def display(self):
		return self.play_root.game_state.display(self.play_root.completed_move)

	def get_player_to_move(self):
		return self.play_root.player_to_move

	def commit_to_move(self, move):
		if move in self.play_root.children:
			self.play_root = self.play_root.children[move]
			if self.play_root.children == {}:
				self.expand(self.play_root)
			return True
		else:
			return False

	def get_weighted_random_move_from_top_k(self, k=3):
		moves_at_simulation_count = {}
		for child in self.play_root.children.values():
			if child.simulations not in moves_at_simulation_count:
				moves_at_simulation_count[child.simulations] = []
			moves_at_simulation_count[child.simulations].append(child.completed_move)

		inverse_sorted_by_simulation_count = list(reversed(sorted(moves_at_simulation_count.items())))

		moves_under_consideration = []
		for simulations, moves in inverse_sorted_by_simulation_count:
			if len(moves_under_consideration) < k:
				moves_under_consideration += [(simulations, x) for x in moves]

		moves = [mv for sims, mv in moves_under_consideration]
		weights = np.array([sims for sims, mv in moves_under_consideration])

		return choice(moves, p=weights/sum(weights))

	def calculate_selection_score(self, child, sqrt_total_sims):
		if child.completed_move == 81:
			return float('-inf')

		Vc = 0 if child.simulations == 0 else child.value_score / child.simulations
		Pc = child.policy_score
		c = 1.0

		return Vc + c*Pc*(sqrt_total_sims/(1+child.simulations))

	def select_child(self, node):
		sqrt_total_sims = sum([child.simulations for move, child in node.children.items()])**0.5
		sorted_children = sorted([(self.calculate_selection_score(child, sqrt_total_sims), child) for move, child in node.children.items()], key=lambda x: x[0])
		return sorted_children[-1][-1]

	def simulate(self, searches=1):
		if searches == -1:
			searches = len(self.play_root.children)

		for s in range(searches):
			walker = self.play_root
			while walker.children != {}:
				walker = self.select_child(walker)
			self.expand(walker)

	def state_to_model_inputs(self, node):
		return game_state_to_model_inputs(node.game_state, node.player_to_move)

	def expand(self, node):

		sensible_moves = node.game_state.get_sensible_moves_for_player(node.player_to_move)

		model_inputs = self.state_to_model_inputs(node)
		policy, value = self.model.predict(np.moveaxis(np.array([model_inputs]), 1, -1))

		policy = policy[0]
		value = value[0][0]

		node.children = {}
		for move in sensible_moves:
			new_game = deepcopy(node.game_state)
			new_game.place_stone(move, node.player_to_move)
			node.children[move] = MCTS.Node(new_game, move, -node.player_to_move, policy[move], node)
			self.lut[len(self.lut)] = node.children[move]
			self.rlut[node.children[move]] = len(self.lut)-1

		self.backup(node, value)

	def backup(self, node, value):
		node.value_score += -node.player_to_move * value
		node.simulations += 1
		if node != self.game_root and node.parent != self.play_root:
			self.backup(node.parent, value)












