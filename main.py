
import time
from keras.models import load_model, save_model
from datasets import _smallnum_to_char_, create_game_record, stringify_game_record, game_info_to_stringified_training_data, create_training_dataset_from_frames
from board import Board
from model import build_agz_model
from mcts import MCTS
import os

blocks = 4
channels = 32

model = None
if os.path.exists('current go model.h5'):
	model = load_model('current go model.h5')
if model == None:
	model = build_agz_model(blocks, channels, (9, 9, 8))

if os.path.exists('9x9 real games') == False:
	open('9x9 real games','w').write('')

real_games = [x for x in open('9x9 real games','r').read().split('\n') if len(x)]
real_games_writer = open('9x9 real games','a')

if os.path.exists('9x9 training frames') == False:
	open('9x9 training frames','w').write('')

training_frames = [x for x in open('9x9 training frames','r').read().split('\n') if len(x)]
training_frames_writer = open('9x9 training frames','a')

start_time = time.time()
games = 1000000
for g in range(games):
	moves = []
	game = Board()
	mcts = MCTS(game, model=model)
	game_in_progress = True
	while game_in_progress:
		print('Player to move:', 'black' if mcts.get_player_to_move() == 1 else 'white', 'Nodes:', mcts.get_node_count())
		mcts.simulate(27)
		move = mcts.get_weighted_random_move_from_top_k(3 if len(moves) < 27 else 1)
		moves.append(move)
		mcts.commit_to_move(move)
		print('moves to date:', len(moves), moves)
		print(mcts.display())
		print()

		if len(moves) > 81*2 or (len(moves) > 2 and moves[-2:] == [81, 81]):
			game_in_progress = False

	if len(moves) < 81*2:
		print((g+1), 'games played @', (time.time()-start_time)/(g+1), 'seconds per game')

		ownership, final_score = mcts.get_final_score()
		outcome = final_score/abs(final_score)

		game_record = create_game_record(9, 6.5, ''.join([_smallnum_to_char_(x) for x in moves]),
			''.join(['b' if x == 1 else 'w' for x in ownership]), final_score, outcome)
		stringified = stringify_game_record(game_record)

		real_games.append(stringified)
		real_games_writer.write(stringified+'\n')

		for frame in game_info_to_stringified_training_data(game_record):
			training_frames.append(frame)
			training_frames_writer.write(frame+'\n')

		batch_size = 512
		if len(real_games) % 10 == 0:
			model_inputs, policy_targets, value_targets = create_training_dataset_from_frames(training_frames, batch_size)
			model.fit(model_inputs, [policy_targets, value_targets], batch_size=batch_size, epochs=1, verbose=1)
			save_model(model, "b" + str(blocks) + "c" + str(channels) + "@" + str(len(real_games)) + ".h5", save_format="h5")
			save_model(model, "current go model.h5", save_format="h5")


