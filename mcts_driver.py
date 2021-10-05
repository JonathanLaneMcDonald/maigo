
"""
i'll use this script to try and get a model, a tree search, and a game working together
"""

import time
from connect_four import ConnectFour
from mcts import MCTS
from model import build_agz_model

blocks = 5
filters = 64
game_constructor = ConnectFour

model = build_agz_model(
	blocks=blocks,
	filters=filters,
	input_shape=game_constructor.get_feature_dimensions(),
	policy_options=game_constructor.get_action_space()
)

tree = MCTS(game_constructor, model)

print("moves: ")
for i in range(100):
	start_time = time.time()
	tree.simulate(None, iterations=1000)
	print(i, tree.play_weighted_random_move(), int(1000/(time.time() - start_time)), tree.depth)







