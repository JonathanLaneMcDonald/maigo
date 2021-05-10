
import numpy as np
from numpy.random import random

from board import Board
from model import build_agz_model

from keras.models import save_model

def _smallnum_to_char_(m):
	return chr(40+m)

def _char_to_smallnum_(m):
	return ord(m)-40

def parse_game_record(game_record):
	return {
		'timestamp': int(game_record.split()[0]),
		'boardsize': int(game_record.split()[1]),
		'komi': float(game_record.split()[2]),
		'moves': game_record.split()[3],
		'ownership': game_record.split()[4],
		'score': float(game_record.split()[5]),
		'outcome': float(game_record.split()[6])
	}

def game_state_to_model_inputs(feature_history, game, board_size, player):

	rules = np.array([player, game.komi/10], dtype=np.float)

	# i'm just going to hard-code this for now while i'm trying to decide what i'm doing
	feature_history[7] = feature_history[5]
	feature_history[6] = feature_history[4]
	feature_history[5] = feature_history[3]
	feature_history[4] = feature_history[2]
	feature_history[3] = feature_history[1]
	feature_history[2] = feature_history[0]

	# my stones/opponent stones
	new_black_plane = np.zeros((board_size, board_size), dtype=np.intc)
	new_white_plane = np.zeros((board_size, board_size), dtype=np.intc)

	for i in range(game.area):
		y, x = i // board_size, i % board_size
		if game.is_black(i):
			new_black_plane[y][x] = 1
		elif game.is_white(i):
			new_white_plane[y][x] = 1

	feature_history[0] = new_black_plane
	feature_history[1] = new_white_plane

	return rules

def game_record_to_model_outputs(game_as_dict, moves, target, board_size):
	policy = np.zeros((board_size, board_size), dtype=np.intc)
	ownership = np.zeros((board_size, board_size), dtype=np.intc)
	final_score = np.zeros(1, dtype=np.intc)
	value = np.zeros(1, dtype=np.float)

	move_value = _char_to_smallnum_(moves[target])
	if move_value < board_size**2:
		y, x = move_value // board_size, move_value % board_size
		policy[y][x] = 1

	for i in range(len(game_as_dict['ownership'])):
		y, x = i // board_size, i % board_size
		if game_as_dict['ownership'][i] == 'b':
				ownership[y][x] = 1
		elif game_as_dict['ownership'][i] == 'w':
				ownership[y][x] = -1

	final_score[0] = max(0, min(2*game_as_dict['boardsize']**2, game_as_dict['score']+game_as_dict['boardsize']**2))

	value[0] = game_as_dict['outcome']

	return policy, ownership, final_score, value

def stringify_inputs_and_targets(meta_data, features, rules, policy, ownership, score, value):
	feature_state = []
	for r in range(meta_data['boardsize']):
		for c in range(meta_data['boardsize']):
			f1 = 0
			if features[0][r][c]:				f1 = 1
			elif features[1][r][c]:				f1 = 2

			f2 = 0
			if features[2][r][c]:				f2 = 1
			elif features[3][r][c]:				f2 = 2

			f3 = 0
			if features[4][r][c]:				f3 = 1
			elif features[5][r][c]:				f3 = 2

			f4 = 0
			if features[6][r][c]:				f4 = 1
			elif features[7][r][c]:				f4 = 2

			feature_state.append(_smallnum_to_char_(f4*27 + f3*9 + f2*3 + f1))

	prediction_state = []
	one_hot_policy = meta_data['boardsize']
	for r in range(meta_data['boardsize']):
		for c in range(meta_data['boardsize']):
			prediction_state.append(_smallnum_to_char_(ownership[r][c] + 1))

			if policy[r][c]:
				one_hot_policy = r*meta_data['boardsize'] + c

	return ''.join(feature_state) + ' ' + str(rules[0]) + ' ' + str(10*rules[1]) + ' ' + str(one_hot_policy) + ' ' + ''.join(prediction_state) + ' ' + str(score[0]) + ' ' + str(value[0])

def string_frame_to_training_frame(frame, board_size):
	features = np.zeros((8, board_size, board_size), dtype=np.intc)
	rules = np.zeros(2, dtype=np.float)
	policy = np.zeros(1, dtype=np.intc)
	ownership = np.zeros((board_size, board_size), dtype=np.intc)
	score = np.zeros(1, dtype=np.intc)
	value = np.zeros(1, dtype=np.float)

	packed_features = frame.split()[0]
	for i in range(board_size**2):
		y, x = i // board_size, i % board_size

		position = _char_to_smallnum_(packed_features[i])

		f1 = position % 3
		f2 = (position // 3) % 3
		f3 = (position // 9) % 3
		f4 = (position // 27) % 3

		if f1 == 1:		features[0][y][x] = 1
		elif f1 == 2:	features[1][y][x] = 1

		if f2 == 1:		features[2][y][x] = 1
		elif f2 == 2:	features[3][y][x] = 1

		if f3 == 1:		features[4][y][x] = 1
		elif f3 == 2:	features[5][y][x] = 1

		if f4 == 1:		features[6][y][x] = 1
		elif f4 == 2:	features[7][y][x] = 1

	packed_player = float(frame.split()[1])
	rules[0] = packed_player

	packed_komi = float(frame.split()[2])
	rules[1] = packed_komi

	packed_policy = int(frame.split()[3])
	policy[0] = packed_policy

	packed_territory = frame.split()[4]
	for i in range(board_size**2):
		y, x = i // board_size, i % board_size
		position = _char_to_smallnum_(packed_territory[i])
		ownership[y][x] = position-1

	packed_score = int(frame.split()[5])
	score[0] = packed_score

	packed_value = float(frame.split()[6])
	value[0] = packed_value

	return features, rules, policy, ownership, score, value

def create_training_dataset_from_frames(frames, samples, board_size):
	features = np.zeros((samples, 8, board_size, board_size), dtype=np.intc)
	rules = np.zeros((samples, 2), dtype=np.float)
	policy = np.zeros((samples, 1), dtype=np.intc)
	ownership = np.zeros((samples, board_size, board_size), dtype=np.intc)
	score = np.zeros((samples, 1), dtype=np.intc)
	value = np.zeros((samples, 1), dtype=np.intc)

	for s in range(samples):
		features[s], rules[s], policy[s], ownership[s], score[s], value[s] = string_frame_to_training_frame(frames[int(random()*len(frames))], board_size)

		if s % 1024 == 0:
			print(s, end=' ')
	print()

	# convert features from channels_first to channels_last
	features = np.moveaxis(features, 1, -1)

	return features, rules, policy, ownership, score, value

def game_record_to_training_strings_with_symmetries(gr, history=4):

	stringified_training_data = []

	feature_history = np.zeros((2*history, gr['boardsize'], gr['boardsize']), dtype=np.intc)

	player = 1
	game = Board(gr['boardsize'], gr['komi'])
	for target in range(len(gr['moves'])):

		# 1) generate model inputs and targets from the current state of the game
		original_rules = game_state_to_model_inputs(feature_history, game, gr['boardsize'], player)
		original_policy, original_ownership, original_score, original_outcome = game_record_to_model_outputs(gr, gr['moves'], target, gr['boardsize'])

		# 2) enumerate all the symmetries
		for k in range(4):
			rotated_features = np.rot90(feature_history, k=k, axes=(-2,-1))
			rotated_policy = np.rot90(original_policy, k=k, axes=(-2,-1))
			rotated_ownership = np.rot90(original_ownership, k=k, axes=(-2,-1))
			stringified_training_data.append(stringify_inputs_and_targets(gr, rotated_features, original_rules, rotated_policy, rotated_ownership, original_score, original_outcome))

			flipped_features = np.flip(rotated_features, axis=-1)
			flipped_policy = np.flip(rotated_policy, axis=-1)
			flipped_ownership = np.flip(rotated_ownership, axis=-1)
			stringified_training_data.append(stringify_inputs_and_targets(gr, flipped_features, original_rules, flipped_policy, flipped_ownership, original_score, original_outcome))

		# 3) advance the game state
		error_happen = not game.place_stone(_char_to_smallnum_(gr['moves'][target]), player)
		if error_happen:
			raise Exception("oh shit, a happening!")
		player *= -1

	return stringified_training_data

if __name__ == "__main__":

	'''
	filename = 'modernized 9x9 games'
	games = [x for x in open(filename, 'r').read().split('\n') if len(x)]
	print(len(games), 'real games loaded')

	with open('test output','w') as fwrite:
		for g in range(len(games)):
			for frame in game_record_to_training_strings_with_symmetries(parse_game_record(games[g])):
				fwrite.write(frame+'\n')
			print(g)
	'''

	filename = 'test output'
	frames = [x for x in open(filename, 'r').read().split('\n') if len(x)]
	print(len(frames), 'training frames loaded')

	model = build_agz_model(4, 32, input_shape=(9, 9, 8))

	for e in range(0,128):
		features, rules, policy, ownership, score, value = create_training_dataset_from_frames(frames, 16384, 9)
		model.fit([features, rules], [policy, ownership, score, value], batch_size=128, epochs=1, verbose=1)
		save_model(model, "crappy go model" + str(e) + ".h5", save_format="h5")





