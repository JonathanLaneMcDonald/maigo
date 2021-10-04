
"""
i'll use this script to try and get a model, a tree search, and a game working together
"""

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

print("moves: ", end='')
for i in range(100):
	tree.simulate(model, iterations=100)
	print(tree.play_weighted_random_move(), end=' ')







