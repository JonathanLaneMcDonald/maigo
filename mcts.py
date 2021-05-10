
import numpy as np
from copy import deepcopy

class MCTS:

	class Child:
		def __init__(self, child):
			self.child = child

	class Node:
		def __init__(self, game_state, completed_move, player_to_move, policy_score, parent):
			self.parent = parent
			self.children = []	#[Child(...)]

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

		# set up game root
		self.expand(self.game_root)

	def get_top_k_moves(self, k=3):
		pass

	def calculate_selection_score(self, child, sqrt_total_sims):
		Vc = 0 if child.simulations == 0 else child.value_score / child.simulations
		Pc = child.policy_score
		c = 1.0

		return Vc + c*Pc*(sqrt_total_sims/(1+child.simulations))

	def select_child(self, node):
		sqrt_total_sims = sum([child.simulations for child in node.children])**0.5
		return sorted([(self.calculate_selection_score(child, sqrt_total_sims), child) for child in node.children])[-1][1].node

	def simulate(self):
		walker = self.play_root
		while walker.children != []:
			walker = self.select_child(walker)
		self.expand(walker)
		self.backup(walker)

	def state_to_input_features(self, node):
		edge_size = node.game_state.side
		player_to_move = node.player_to_move
		features = np.zero((1, edge_size, edge_size, 2), dtype=np.intc)

		# TODO: figure out where this stuff belongs, because it doesn't blong here
		for i in range(node.game_state.area):
			y, x = i//edge_size, i%edge_size
			if node.game_state.is_black(i):
				if player_to_move == 1:
					features[0][y][x][0] = 1
				else:
					features[0][y][x][1] = 1
			elif node.game_state.is_white(i):
				if player_to_move == -1:
					features[0][y][x][0] = 1
				else:
					features[0][y][x][1] = 1

		return features

	def expand(self, node):
		# def __init__(self, game_state, completed_move, player_to_move, policy_score, parent):

		sensible_moves = node.game_state.get_sensible_moves_for_player(node.player_to_move)

		state = self.state_to_input_features(node)
		policy, value = self.model.predict(state)

		node.value_score = value
		node.simulations = 1

		node.children = [None]
		for move in sensible_moves:
			new_game = deepcopy(node.game_state)
			new_game.place_stone()



'''
mcts:

	expand(node):
		back up to most recent game state
		replay moves down to this node
		populate children with legal children

	simulate(node):
		produce a value for this particular node
		produce policy values for each child

	backup(node, value):
		for each node until current play root
			walker_node.value += value
			walker_node.simulations += 1

'''











