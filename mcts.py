
import numpy as np
from numpy.random import choice, random
from copy import deepcopy
from datasets import game_state_to_model_inputs

class MCTS:

	class Node:
		def __init__(self, game_state, completed_move, player_to_move, parent):
			self.parent = parent

			self.completed_move = completed_move
			self.player_to_move = player_to_move

			self.child_policy = np.zeros(82, dtype=np.float)
			self.child_value = np.zeros(82, dtype=np.float)
			self.child_subtree_sims = np.zeros(82, dtype=np.intc)
			self.child_legality = np.zeros(82, dtype=np.intc)
			self.child_nodes = [None]*82

			self.game_state = game_state

	def __init__(self, game_state, model=None):
		# set up nodes and model
		self.game_root = MCTS.Node(game_state, None, 1, None)
		self.play_root = self.game_root
		self.model = model

		# working toward facilitating multiprocess communication - can't send objects, so i'll send unique identifiers
		self.lut = {0:self.game_root}
		self.rlut = {self.game_root:0}

		# set up game root
		self.evaluate_node(self.game_root)

	def get_node_count(self):
		return len(self.lut)

	def get_final_score(self):
		return self.play_root.game_state.get_simple_terminal_score_and_ownership()

	def display(self):
		return self.play_root.game_state.display(self.play_root.completed_move)

	def get_player_to_move(self):
		return self.play_root.player_to_move

	def commit_to_move(self, move):
		if self.play_root.child_legality[move]:
			if self.play_root.child_nodes[move] == None:
				self.expand_and_evaluate(self.play_root, move)
			self.play_root = self.play_root.child_nodes[move]
			return True
		else:
			return False

	def get_weighted_random_move_from_top_k(self, k=3):
		moves_at_simulation_count = {}
		for move in [x for x in range(82) if self.play_root.child_legality[x]]:
			if self.play_root.child_subtree_sims[move] not in moves_at_simulation_count:
				moves_at_simulation_count[self.play_root.child_subtree_sims[move]] = []
			moves_at_simulation_count[self.play_root.child_subtree_sims[move]].append(move)

		inverse_sorted_by_simulation_count = list(reversed(sorted(moves_at_simulation_count.items())))

		moves_under_consideration = []
		for simulations, moves in inverse_sorted_by_simulation_count:
			if len(moves_under_consideration) < k:
				moves_under_consideration += [(simulations, x) for x in moves]

		moves = [mv for sims, mv in moves_under_consideration]
		weights = np.array([sims for sims, mv in moves_under_consideration])+1

		return choice(moves, p=weights/sum(weights))

	def get_search_value(self, move, value, policy, sims, sqrt_total_sims):
		if move == 81:
			return float('-inf')

		Vc = 0 if sims == 0 else value / sims
		Pc = policy
		c = 1.0

		return Vc + c*Pc*(sqrt_total_sims/(1+sims))

	def select_child(self, node):
		sqrt_total_sims = sum(node.child_subtree_sims)**0.5
		sorted_children = sorted([(self.get_search_value(
			x,
			node.child_value[x],
			node.child_policy[x],
			node.child_subtree_sims[x],
			sqrt_total_sims
		), x) for x in range(82) if node.child_legality[x]], key=lambda x: x[0])
		return sorted_children[-1][-1]

	def simulate(self, searches=-1):
		if searches == -1:
			searches = sum(self.play_root.child_legality)

		for s in range(searches):
			walker = self.play_root
			next_move = self.select_child(walker)

			walking = True
			while walking:
				if walker.child_nodes[next_move] == None:
					walking = False
				else:
					walker = walker.child_nodes[next_move]
					next_move = self.select_child(walker)

			self.expand_and_evaluate(walker, next_move)

	def state_to_model_inputs(self, node):
		return game_state_to_model_inputs(node.game_state, node.player_to_move)

	def handy_inference_wrapper(self, node):
		model_inputs = self.state_to_model_inputs(node)
		policy, value = self.model.predict(np.moveaxis(np.array([model_inputs]), 1, -1))

		return policy[0], value[0][0]

	def dandy_inference_wrapper(self, node):
		return random(82), random()

	def evaluate_node(self, node):

		policy, value = self.handy_inference_wrapper(node)

		for move in node.game_state.get_sensible_moves_for_player(node.player_to_move):
			node.child_legality[move] = 1
			node.child_policy[move] = policy[move]

		self.backup(node, value)

	def backup(self, node, value):
		# "value" is the likelihood that black wins. account for this when selecting nodes in search
		if node != self.game_root and node.parent != self.play_root:
			node.parent.child_value[node.completed_move] += value
			node.parent.child_subtree_sims[node.completed_move] += 1
			self.backup(node.parent, value)

	def expand_and_evaluate(self, node, next_move):

		new_game = deepcopy(node.game_state)
		if not new_game.place_stone(next_move, node.player_to_move):
			raise Exception("error performing move")
		node.child_nodes[next_move] = MCTS.Node(new_game, next_move, -node.player_to_move, node)
		self.evaluate_node(node.child_nodes[next_move])

		self.lut[len(self.lut)] = node.child_nodes[next_move]
		self.rlut[node.child_nodes[next_move]] = len(self.lut)-1






