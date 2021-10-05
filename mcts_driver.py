
"""
i'll use this script to try and get a model, a tree search, and a game working together
"""

import time
from connect_four import ConnectFour
from mcts import MCTS
from model import build_agz_model
from game import GameStatus

blocks = 5
filters = 64
game_constructor = ConnectFour

model = build_agz_model(
	blocks=blocks,
	filters=filters,
	input_shape=game_constructor.get_feature_dimensions(),
	policy_options=game_constructor.get_action_space()
)

start_time = time.time()
for g in range(1000):
	tree = MCTS(game_constructor, model)
	while tree.get_tree_status() == GameStatus.in_progress:
		tree.simulate(None, iterations=10000)
		_ = tree.play_weighted_random_move()
		tree.display_play_root()
	print((g+1)/(time.time()-start_time))






