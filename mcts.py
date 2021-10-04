
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

	def expand_and_evaluate(self, node: Node, model):
		node.expand_and_evaluate(model)

	def initialize_root(self, game: TeachableGame, model):
		self.game_root = MCTS.Node(game, -1, 1, None)
		self.play_root = self.game_root
		self.expand_and_evaluate(self.play_root, model)






