
import numpy as np
from numpy.random import choice

from game import TeachableGame


class MCTS:

	class Node:
		def __init__(self, game: TeachableGame, completed_move: int, player_to_move: int, parent):
			self.parent = parent

			self.completed_move = completed_move
			self.player_to_move = player_to_move

			self.subtree_value = 0
			self.subtree_simulations = 0

			self.policy = np.zeros(game.get_action_space(), dtype=float)
			self.legality = np.zeros(game.get_action_space(), dtype=int)
			self.children = [None]*game.get_action_space()

			# the copy we're given should be a deep-copied object
			self.game = game

		def value_of(self, child):
			value = self.subtree_value/max(1, self.subtree_simulations)
			if isinstance(self.children[child], MCTS.Node):
				value = self.children[child].subtree_value/max(1, self.children[child].subtree_simulations)

			policy = self.policy[child]

			sqrt_parent_simulations = self.subtree_simulations**0.5
			child_simulations = 1 + (0 if self.children[child] is None else self.children[child].subtree_simulations)

			return value + policy*(sqrt_parent_simulations/child_simulations)

		def expand_and_evaluate(self, model):
			features = np.array([self.game.get_state_as_features(self.player_to_move)], dtype=np.ubyte)
			policy, value = model.predict(features)
			self.legality = self.game.get_move_legality()
			self.policy = policy[0] * self.legality
			self.subtree_value = value[0]
			self.subtree_simulations = 1

	def __init__(self, game_constructor, model):
		# need handles for the game and play root nodes
		self.game_root = None
		self.play_root = None

		# what game are we playing?
		self.game_constructor = game_constructor

		# set up game root
		self.initialize_root(game_constructor(), model)

	def initialize_root(self, game: TeachableGame, model):
		self.game_root = MCTS.Node(game, -1, 1, None)
		self.play_root = self.game_root
		self.expand_and_evaluate(self.play_root, model)

	def backup(self, node, value):
		walker = node.parent
		while walker is not None:
			walker.subtree_value += value
			walker.subtree_simulations += 1
			walker = node.parent

	def expand_and_evaluate(self, node: Node, model):
		node.expand_and_evaluate(model)
		self.backup(node, node.subtree_value)

	def simulate(self, model):
		"""
		we'll pick use the UCT equation to recurse down to a leaf, then we'll expand it and backprop the value
		"""
		node = self.play_root

		recursing = True
		while recursing:
			scores = {
				value:move for value, move in [
					(node.value_of(move), move) for move in range(len(node.children))
				]
			}

			move = sorted(scores)[-1][1]

			if node.children[move] is not None:
				node = node.children[move]
			else:
				new_game = self.game_constructor(node.game)
				new_game.do_move(move, node.player_to_move)
				node.children[move] = MCTS.Node(new_game, move, 1 if node.player_to_move == 2 else 2, node)
				node.children[move].expand_and_evaluate(model)
				recursing = False










