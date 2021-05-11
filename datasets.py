
import numpy as np
import time
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

def create_game_record(board_size, komi, moves, ownership, score, outcome):
	return {
		'timestamp': int(time.time()),
		'boardsize': board_size,
		'komi': komi,
		'moves': moves,
		'ownership': ownership,
		'score': score,
		'outcome': outcome
	}

def game_state_to_model_inputs(game, player_to_move):

	model_inputs = np.zeros((8, 9, 9), dtype=np.intc)

	# stone placement
	black_stones_plane = np.zeros((9, 9), dtype=np.intc)
	white_stones_plane = np.zeros((9, 9), dtype=np.intc)

	for i in range(game.area):
		y, x = i // 9, i % 9
		if game.is_black(i):
			black_stones_plane[y][x] = 1
		elif game.is_white(i):
			white_stones_plane[y][x] = 1

	model_inputs[0] = black_stones_plane
	model_inputs[1] = white_stones_plane

	# legal moves
	black_legal_moves = np.zeros((9, 9), dtype=np.intc)
	white_legal_moves = np.zeros((9, 9), dtype=np.intc)

	for i in range(game.area):
		y, x = i // 9, i % 9
		if game.is_legal_for_black(i):
			black_legal_moves[y][x] = 1
		if game.is_legal_for_white(i):
			white_legal_moves[y][x] = 1

	model_inputs[2] = black_legal_moves
	model_inputs[3] = white_legal_moves

	# liberties
	liberties = np.zeros((3, 9, 9), dtype=np.intc)

	for i in range(game.area):
		y, x = i // 9, i % 9
		if 0 < game.get_liberties_for_position(i) <= 3:
			liberties[game.get_liberties_for_position(i)-1][y][x] = 1

	model_inputs[4] = liberties[0]
	model_inputs[5] = liberties[1]
	model_inputs[6] = liberties[2]

	# who's turn is it?
	if player_to_move == 1:
		model_inputs[7] = np.zeros((9, 9), dtype=np.intc)
	else:
		model_inputs[7] = np.ones((9, 9), dtype=np.intc)

	return model_inputs

def game_info_to_model_outputs(info, target):
	policy = np.array([_char_to_smallnum_(info['moves'][target])], dtype=np.intc)
	value = np.array([info['outcome']], dtype=np.intc)
	return policy, value

def stringify_inputs_and_targets(model_inputs, policy_target, value_target, player_to_move):
	stringified = []
	for r in range(9):
		for c in range(9):
			# encoding stone locations {0,1,2}
			stones = 0
			if model_inputs[0][r][c] == 1:		stones = 1
			if model_inputs[1][r][c] == 1:		stones = 2

			# encoding legal moves {0,1,2,3}
			legality = (model_inputs[2][r][c]<<1) + model_inputs[3][r][c]

			# encoding liberties {0,1,2,3}
			liberties = 0
			if model_inputs[4][r][c] == 1:		liberties = 1
			if model_inputs[5][r][c] == 1:		liberties = 2
			if model_inputs[6][r][c] == 1:		liberties = 3

			# we already know who's turn it is, so we can leave channel 7 out to make this more compressible

			stringified.append(_smallnum_to_char_((stones*16) + (legality*4) + liberties))

	return ''.join(stringified) + ' ' + str(player_to_move) + ' ' + str(policy_target[0]) + ' ' + str(value_target[0])

def stringified_to_training_frame(frame):
	model_inputs = np.zeros((8, 9, 9), dtype=np.intc)

	packed_model_inputs = frame.split()[0]
	player_to_move = int(frame.split()[1])
	for i in range(81):
		y, x = i // 9, i % 9

		value = _char_to_smallnum_(packed_model_inputs[i])

		stones = value // 16
		if stones == 1:		model_inputs[0][y][x] = 1
		if stones == 2:		model_inputs[1][y][x] = 1

		legality = (value % 16) // 4
		if legality & 2:	model_inputs[2][y][x] = 1
		if legality & 1:	model_inputs[3][y][x] = 1

		liberties = value % 4
		if liberties:		model_inputs[3+liberties][y][x] = 1

		if player_to_move == 1:		model_inputs[7] = np.zeros((9, 9), dtype=np.intc)
		else:						model_inputs[7] = np.ones((9, 9), dtype=np.intc)

	policy_target = np.array([int(frame.split()[2])], dtype=np.intc)
	value_target = np.array([int(frame.split()[3])], dtype=np.intc)

	return model_inputs, policy_target, value_target

def create_training_dataset_from_frames(frames, samples):
	model_inputs = np.zeros((samples, 8, 9, 9), dtype=np.intc)
	policy_targets = np.zeros((samples, 1), dtype=np.intc)
	value_targets = np.zeros((samples, 1), dtype=np.intc)

	for s in range(samples):
		model_inputs[s], policy_targets[s], value_targets[s] = stringified_to_training_frame(frames[int(random()*len(frames))])

	# convert features from channels_first to channels_last
	model_inputs = np.moveaxis(model_inputs, 1, -1)

	return model_inputs, policy_targets, value_targets

def game_info_to_stringified_training_data(info):

	stringified_training_data = []

	player_to_move = 1
	game = Board(9, 6.5)
	for target in range(len(info['moves'])):
		model_inputs = game_state_to_model_inputs(game, player_to_move)
		policy_target, value_target = game_info_to_model_outputs(info, target)

		stringified_training_data.append(stringify_inputs_and_targets(model_inputs, policy_target, value_target, player_to_move))

		error_happen = not game.place_stone(_char_to_smallnum_(info['moves'][target]), player_to_move)
		if error_happen:
			raise Exception("oh shit, a happening!")
		player_to_move *= -1

	return stringified_training_data

if __name__ == "__main__":

	'''
	filename = 'modernized 9x9 games'
	games = [x for x in open(filename, 'r').read().split('\n') if len(x)]
	print(len(games), 'real games loaded')

	with open('simple agz training data','w') as fwrite:
		for g in range(len(games)):
			for frame in game_info_to_stringified_training_data(parse_game_record(games[g])):
				fwrite.write(frame+'\n')
			print(g)
	'''


	filename = 'simple agz training data'
	frames = [x for x in open(filename, 'r').read().split('\n') if len(x)]
	print(len(frames), 'training frames loaded')

	blocks = 6
	channels = 96

	model = build_agz_model(blocks, channels, input_shape=(9, 9, 8))

	batch_size = 512
	updates_per_epoch = 500
	samples = batch_size*updates_per_epoch
	total_updates = 12500
	epochs = total_updates // updates_per_epoch

	for e in range(1, epochs+1):
		model_inputs, policy_targets, value_targets = create_training_dataset_from_frames(frames, samples)
		model.fit(model_inputs, [policy_targets, value_targets], batch_size=batch_size, epochs=1, verbose=1)
		save_model(model, "crappy go model b" + str(blocks) + "c" + str(channels) + " " + str(e*updates_per_epoch) + ".h5", save_format="h5")





