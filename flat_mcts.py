
import numpy as np
from numpy.random import choice, random
from copy import deepcopy
from datasets import game_state_to_model_inputs

class FlatMCTS:
	"""
	i'll try and sketch out some steps for setup and steady-state operation of the mcts...








	"""

	class Node:
		def __init__(self, game_state, completed_move, player_to_move, parent):
			self.parent = parent

			self.move_completed_to_get_here = completed_move
			self.player_to_move = player_to_move

			self.value_of_my_subtree = 0
			self.simulations_in_my_subtree = 0

			# mv: Node
			self.child_nodes = {}

			# mv: policy
			self.child_policy = {}

			self.game_state = game_state

	def __init__(self):

		self.game_root = None
		self.play_root = None

		self.initialize_root()

	def initialize_root(self):




	class Node:
		def __init__(self, game_state, completed_move, player_to_move, parent):
			self.parent = parent

			self.completed_move = completed_move
			self.player_to_move = player_to_move

			self.child_policy = np.zeros(82, dtype=np.float)
			self.child_value = np.zeros(82, dtype=np.float)
			self.child_subtree_sims = np.zeros(82, dtype=np.intc)
			self.child_outstanding_sims = np.zeros(82, dtype=np.intc)
			self.child_legality = np.zeros(82, dtype=np.intc)
			self.child_nodes = [None]*82

			self.game_state = game_state

	def __init__(self, game_state, process_id, tasks_from_mcts_to_model, results_from_model_to_mcts):
		# set up nodes and model
		self.game_root = MCTS.Node(game_state, None, 1, None)
		self.play_root = self.game_root

		# these are more diagnostic
		self.deepest_leaf = 0
		self.episode_cell_visits = 0

		self.process_id = process_id
		self.random_number = int(random()*100_000)
		self.tasks_to_model = tasks_from_mcts_to_model
		self.results_from_model = results_from_model_to_mcts

		self.node_lut = {0:self.game_root}
		self.node_rlut = {self.game_root:0}

		# set up game root
		self.forward_state_for_inference(self.game_root)

	def get_node_cell_visits(self):
		return self.play_root.game_state.get_cell_visits()

	def get_episode_cell_visits(self):
		return self.episode_cell_visits

	def get_recursion_depth(self):
		return self.deepest_leaf

	def get_outstanding_sims(self):
		return sum(self.play_root.child_outstanding_sims)

	def get_value_at_play_root(self):
		if self.play_root == self.game_root:
			return 'zero'
		else:
			return self.play_root.parent.child_value[self.play_root.completed_move]

	def get_node_count(self):
		return len(self.node_lut)

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

		while sum(self.play_root.child_outstanding_sims):
			self.flush_results_queue()

		print("Outstanding Sims:", self.get_outstanding_sims())

		moves_at_simulation_count = {}
		for move in [x for x in range(82) if self.play_root.child_legality[x]]:
			total_sims = self.play_root.child_subtree_sims[move] + self.play_root.child_outstanding_sims[move]
			if total_sims not in moves_at_simulation_count:
				moves_at_simulation_count[total_sims] = []
			moves_at_simulation_count[total_sims].append(move)

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
		c = 1.5

		return Vc + c*Pc*(sqrt_total_sims/(1+sims))

	def select_child(self, node):
		sqrt_total_sims = sum(node.child_subtree_sims + node.child_outstanding_sims)**0.5
		sorted_children = sorted([(self.get_search_value(
			x,
			node.child_value[x],
			node.child_policy[x],
			node.child_subtree_sims[x] + node.child_outstanding_sims[x],
			sqrt_total_sims
		), x) for x in range(82) if node.child_legality[x]], key=lambda x: x[0])
		return sorted_children[-1][-1]

	def flush_results_queue(self):
		while not self.results_from_model.empty():
			self.pull_and_apply_prediction()

	def simulate(self, min_searches=9, searches=-1, max_recursion_depth=30):
		if searches == -1:
			searches = max(min_searches, sum(self.play_root.child_legality))

		self.deepest_leaf = 0
		self.episode_cell_visits = 0

		for s in range(searches):

			self.flush_results_queue()

			walker = self.play_root
			next_move = self.select_child(walker)

			recurse_depth = 0

			walking = True
			while walking:

				walker.child_outstanding_sims[next_move] += 1

				if walker.child_nodes[next_move] == None:
					self.expand_and_evaluate(walker, next_move)
					walking = False
				else:
					walker = walker.child_nodes[next_move]
					next_move = self.select_child(walker)

					recurse_depth += 1
					if recurse_depth >= max_recursion_depth:
						self.dummy_backup(walker, self.play_root)
						walking = False

			if self.deepest_leaf < recurse_depth:
				self.deepest_leaf = recurse_depth

	def state_to_model_inputs(self, node):
		return game_state_to_model_inputs(node.game_state, node.player_to_move)

	def forward_state_for_inference(self, node):

		# this is hacky. i'm getting errors because i haven't established legality, so this is a work-around
		# but i need to figure out the right way to organize all of this once i actually get it working
		for move in node.game_state.get_sensible_moves_for_player(node.player_to_move):
			node.child_legality[move] = 1

		self.tasks_to_model.put((self.random_number, self.process_id, self.node_rlut[node], self.node_rlut[self.play_root], self.state_to_model_inputs(node)))

	def pull_and_apply_prediction(self):

		if self.results_from_model.empty():
			raise Exception("how did i get here if the queue is empty?")

		random_id, node_id, parent_id, policy, value = self.results_from_model.get()

		if self.random_number == random_id:
			#print("received result for node", node_id, "parent", parent_id)
			node = self.node_lut[node_id]
			originating_ancestor = self.node_lut[parent_id]
			for move in node.game_state.get_sensible_moves_for_player(node.player_to_move):
				node.child_legality[move] = 1
				node.child_policy[move] = policy[move]

			self.backup(node, originating_ancestor, value)
		else:
			print("found a stale result (probably from an old game)", self.random_number, "!=", random_id)

	def dummy_backup(self, node, originating_ancestor):
		'''the max recursion depth has been reached, so we're giving up on recursing farther and '''
		assert node != self.play_root
		if node.parent.child_subtree_sims[node.completed_move]:
			self.backup(node, originating_ancestor, node.parent.child_value[node.completed_move]/node.parent.child_subtree_sims[node.completed_move])
		else:
			#raise Exception("surprisingly, we got to the recursion limit on an unsimulated node")
			print("Reached recursion limit on unsimulated node -- Assigning value of zero")
			self.backup(node, originating_ancestor, 0)

	def backup(self, node, originating_ancestor, value):
		# "value" is the likelihood that black wins. account for this when selecting nodes in search
		if node != originating_ancestor:
			node.parent.child_value[node.completed_move] += value
			node.parent.child_subtree_sims[node.completed_move] += 1
			node.parent.child_outstanding_sims[node.completed_move] -= 1
			self.backup(node.parent, originating_ancestor, value)

	def expand_and_evaluate(self, node, next_move):

		new_game = deepcopy(node.game_state)
		if not new_game.place_stone(next_move, node.player_to_move):
			raise Exception("error performing move")

		self.episode_cell_visits += new_game.get_cell_visits()
		node.child_nodes[next_move] = MCTS.Node(new_game, next_move, -node.player_to_move, node)

		self.node_lut[len(self.node_lut)] = node.child_nodes[next_move]
		self.node_rlut[node.child_nodes[next_move]] = len(self.node_lut)-1

		self.forward_state_for_inference(node.child_nodes[next_move])




