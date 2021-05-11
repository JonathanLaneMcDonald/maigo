
from datasets import *
from board import Board
from model import build_agz_model
from mcts import MCTS

game = Board()
model = build_agz_model(5, 64, (9, 9, 8))

filename = 'simple agz training data'
frames = [x for x in open(filename, 'r').read().split('\n') if len(x)]
print(len(frames), 'training frames loaded')

batch_size = 128
updates_per_epoch = 500
samples = batch_size * updates_per_epoch

for e in range(1):
	model_inputs, policy_targets, value_targets = create_training_dataset_from_frames(frames, samples)
	model.fit(model_inputs, [policy_targets, value_targets], batch_size=batch_size, epochs=1, verbose=1)

games = 10
for g in range(games):
	moves = []
	mcts = MCTS(game, model=model)
	game_in_progress = True
	while game_in_progress:
		print('Player to move:', 'black' if mcts.get_player_to_move() == 1 else 'white')
		mcts.simulate(81)
		move = mcts.get_weighted_random_move_from_top_k(max(1, 9-len(moves)))
		moves.append(move)
		mcts.commit_to_move(move)
		print('moves to date:', moves)
		print(mcts.display())
		print()

		if len(moves) > 81*2 or (len(moves) > 2 and moves[-2:] == [81, 81]):
			game_in_progress = False

	print(''.join([_smallnum_to_char_(x) for x in moves]))

