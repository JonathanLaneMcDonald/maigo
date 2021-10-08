
import numpy as np
from numpy.random import choice

from game import TeachableGame, GameStatus


class TraditionalMCTS:

	class Node:
		def __init__(self, game: TeachableGame, completed_move: int, player_to_move: int, parent):
			self.parent = parent

			self.completed_move = completed_move
			self.player_to_move = player_to_move

			self.subtree_victories = 0
			self.subtree_simulations = 0
			self.subtree_truncations = 0 # max_depth exceeded or end_game reached - added to policy to discourage visits

			self.legality = np.zeros(game.get_action_space(), dtype=int)
			self.children = [None]*game.get_action_space()

			# the copy we're given should be a deep-copied object
			self.game = game

			self.expand_and_evaluate()

		def value_of(self, child):
			policy = self.subtree_victories/max(1, self.subtree_simulations)
			if isinstance(self.children[child], TraditionalMCTS.Node):
				policy = self.children[child].subtree_victories/max(1, self.children[child].subtree_simulations)

			sqrt_parent_simulations = self.subtree_simulations**0.5
			child_simulations = 1 + (0 if self.children[child] is None else self.children[child].subtree_simulations + self.children[child].subtree_truncations)

			return policy*(sqrt_parent_simulations/child_simulations)

		def expand_and_evaluate(self):
			self.legality = self.game.get_move_legality()
			self.policy = self.legality
			self.value = self.game.complete_as_rollout(self.player_to_move)
			self.subtree_simulations = 1

	def __init__(self, game_constructor):
		# need handles for the game and play root nodes
		self.game_root = None
		self.play_root = None

		# deepest backprop signal depth
		self.depth = 0
		self.max_lookahead = 20

		# what game are we playing?
		self.game_constructor = game_constructor

		# set up game root
		self.initialize_root(game_constructor())

	def get_tree_status(self):
		return self.play_root.game.get_status()

	def display_play_root(self):
		self.play_root.game.display()

	def initialize_root(self, game: TeachableGame):
		self.game_root = TraditionalMCTS.Node(game, -1, 1, None)
		self.play_root = self.game_root

	def backup(self, node, value, search_aborted=False):
		depth = 0
		walker = node
		while walker.parent is not None:
			depth += 1
			walker = walker.parent
			if search_aborted:
				walker.subtree_truncations += 1
			else:
				walker.subtree_victories += value
				walker.subtree_simulations += 1
		self.depth = max(self.depth, depth)

	def simulate(self, iterations=None):
		"""
		we'll pick use the UCT equation to recurse down to a leaf, then we'll expand it and backprop the value
		"""
		if iterations is None:
			iterations = self.game_constructor.get_action_space()

		for _ in range(iterations):
			node = self.play_root

			recursing = True
			lookahead_depth = 0
			while recursing:
				scores = [(node.value_of(move), move) for move in range(len(node.children))]

				move = sorted(scores)[-1][1]

				lookahead_depth += 1
				if lookahead_depth == self.max_lookahead:
					self.backup(node, value=0, search_aborted=True)
					recursing = False
				elif node.game.get_status() != GameStatus.in_progress:
					self.backup(node, value=node.game.get_winner())
					recursing = False
				elif node.children[move] is not None:
					node = node.children[move]
				else:
					new_game = self.game_constructor(node.game)
					new_game.do_move(move, node.player_to_move)
					node.children[move] = TraditionalMCTS.Node(new_game, move, -node.player_to_move, node)
					self.backup(node.children[move], node.value_of(move))
					recursing = False

	def play_weighted_random_move(self, top_k=None, show_weights=False):
		if top_k is None:
			top_k = self.game_constructor.get_action_space()

		simulations = [
			(
				0 if self.play_root.children[move] is None else self.play_root.children[move].subtree_simulations,
				move
			) for move in range(len(self.play_root.children))
		]

		if show_weights:
			print("weights:", simulations, end=' ')

		simulations = [(sims, move) for sims, move in sorted(simulations)[-top_k:]]

		weights = np.array([x[0] for x in simulations], dtype=float)
		weights /= max(1, sum(weights))

		moves = [x[1] for x in simulations]

		selected_move = choice(moves, p=weights)

		if show_weights:
			print("move:", selected_move)

		self.play_root = self.play_root.children[selected_move]

		return selected_move





