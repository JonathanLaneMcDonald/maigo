
import numpy as np
from numpy.random import random

from board import Board
from model import build_agz_model

from keras.models import save_model

def _9x9_to_char_(m):
	return chr(40+m)

def _char_to_9x9_(m):
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

def recreate_random_game_state(game_records, board_size):

	tries = 0
	target = 0
	moves = [_9x9_to_char_(board_size**2)]
	game_as_dict = {}
	while _char_to_9x9_(moves[target]) == board_size**2:
		game_as_dict = parse_game_record(game_records[int(random() * len(game_records))])
		moves = game_as_dict['moves']
		target = int(random() * len(moves))

		tries += 1
		if tries >= 100:
			raise Exception("there seems to be something seriously wrong with your game dataset")

	game = Board(board_size)
	player = 1
	warmup = moves[:target]
	for m in warmup:
		move = _char_to_9x9_(m)
		error_happen = not game.place_stone(move, player)
		if error_happen:
			raise Exception("oh shit, a happening!")
		player *= -1

	return game_as_dict, game, moves, target, player

def game_state_to_model_inputs(game, board_size, player):

	features = np.zeros((4, board_size, board_size), dtype=np.intc)
	rules = np.array([-player * game.komi/15], dtype=np.float)

	# my stones/opponent stones
	for i in range(game.area):
		y, x = i // board_size, i % board_size
		if game.is_black(i):
			if player == 1:
				features[0][y][x] = 1
			else:
				features[1][y][x] = 1
		elif game.is_white(i):
			if player == -1:
				features[0][y][x] = 1
			else:
				features[1][y][x] = 1

	# legal moves for myself and my opponent
	for i in range(game.area):
		y, x = i // board_size, i % board_size
		if game.is_legal_for_black(i):
			if player == 1:
				features[2][y][x] = 1
			else:
				features[3][y][x] = 1
		if game.is_legal_for_white(i):
			if player == -1:
				features[2][y][x] = 1
			else:
				features[3][y][x] = 1

	return features, rules

def game_record_to_model_outputs(game_as_dict, moves, target, player, board_size):
	policy = np.zeros((board_size, board_size), dtype=np.intc)
	ownership = np.zeros((board_size, board_size), dtype=np.intc)
	value = np.zeros(1, dtype=np.float)

	move_value = _char_to_9x9_(moves[target])
	if move_value < board_size**2:
		y, x = move_value // board_size, move_value % board_size
		policy[y][x] = 1

	if 'ownership' not in game_as_dict or len(game_as_dict['ownership']) != board_size**2:
		raise Exception("your game record's messed up, yo!")

	for i in range(len(game_as_dict['ownership'])):
		y, x = i // board_size, i % board_size
		if game_as_dict['ownership'] == 'b':
			if player == 1:
				ownership[y][x] = 1
			else:
				ownership[y][x] = -1
		elif game_as_dict['ownership'] == 'w':
			if player == -1:
				ownership[y][x] = 1
			else:
				ownership[y][x] = -1

	if 'outcome' not in game_as_dict:
		raise Exception("your game record's messed up, yo! there's no outcame!?")

	if player == 1:
		value[0] = game_as_dict['outcome']
	elif player == -1:
		value[0] = -game_as_dict['outcome']

	return policy, ownership, value

def create_training_dataset(game_records, samples, board_size):
	features = np.zeros((samples, 4, board_size, board_size), dtype=np.intc)
	rules = np.zeros((samples, 1), dtype=np.float)
	policy = np.zeros((samples, board_size * board_size), dtype=np.intc)
	ownership = np.zeros((samples, board_size, board_size), dtype=np.intc)
	value = np.zeros((samples, 1), dtype=np.intc)

	for s in range(samples):
		# select and recreate a random game state
		game_as_dict, game, moves, target, player = recreate_random_game_state(game_records, board_size)
		instance_features, instance_rules = game_state_to_model_inputs(game, board_size, player)
		instance_policy, instance_ownership, instance_value = game_record_to_model_outputs(game_as_dict, moves, target, player, board_size)

		# for data augmentation, select a random rotation
		k = int(random()*4)
		instance_features = np.rot90(instance_features, k=k, axes=(-2,-1))
		instance_policy = np.rot90(instance_policy, k=k, axes=(-2,-1))
		instance_ownership = np.rot90(instance_ownership, k=k, axes=(-2,-1))

		# for data augmentation, choose whether to flip the board
		f = int(random()*2)
		if f:
			instance_features = np.flip(instance_features, axis=-1)
			instance_policy = np.flip(instance_policy, axis=-1)
			instance_ownership = np.flip(instance_ownership, axis=-1)

		features[s] = instance_features
		rules[s] = instance_rules
		policy[s] = np.reshape(instance_policy, board_size**2)
		ownership[s] = instance_ownership
		value[s] = instance_value

		if s % 100 == 0:
			print(s, end=' ')
	print()

	# convert features from channels_first to channels_last
	features = np.moveaxis(features, 1, -1)

	return features, rules, policy, ownership, value

def stringify_inputs_and_targets(meta_data, player, features, rules, policy, ownership, value):
	state = []
	for r in range(meta_data['boardsize']):
		for c in range(meta_data['boardsize']):
			f1 = features[0][r][c]
			f2 = features[1][r][c]
			f3 = features[2][r][c]
			f4 = features[3][r][c]
			p1 = policy[r][c]
			o1 = 1 if ownership[r][c] == 1 else 0
			state.append(_9x9_to_char_((f1<<5) + (f2<<4) + (f3<<3) + (f4<<2) + (p1<<1) + o1))
	return ''.join(state) + ' ' + str(player) + ' ' + str(15*rules[0]) + ' ' + str(value[0]) + ' ' + str(meta_data['score'])

def string_frame_to_training_frame(frame, board_size):
	features = np.zeros((4, board_size, board_size), dtype=np.intc)
	rules = np.array([float(frame.split()[1])/15], dtype=np.float)
	policy = np.zeros((board_size * board_size), dtype=np.intc)
	ownership = np.zeros((board_size, board_size), dtype=np.intc)
	value = np.array([float(frame.split()[3])], dtype=np.float)

	packed_boards = frame.split()[0]
	for i in range(board_size**2):
		y, x = i // board_size, i % board_size
		features[0][y][x] = 1 if _char_to_9x9_(packed_boards[i]) & 1<<5 else 0
		features[1][y][x] = 1 if _char_to_9x9_(packed_boards[i]) & 1<<4 else 0
		features[2][y][x] = 1 if _char_to_9x9_(packed_boards[i]) & 1<<3 else 0
		features[3][y][x] = 1 if _char_to_9x9_(packed_boards[i]) & 1<<2 else 0
		policy[y*board_size + x] = 1 if _char_to_9x9_(packed_boards[i]) & 1<<1 else 0
		ownership[y][x] = 1 if _char_to_9x9_(packed_boards[i]) & 1 else -1

	return features, rules, policy, ownership, value

def create_training_dataset_from_frames(frames, samples, board_size):
	features = np.zeros((samples, 4, board_size, board_size), dtype=np.intc)
	rules = np.zeros((samples, 1), dtype=np.float)
	policy = np.zeros((samples, board_size * board_size), dtype=np.intc)
	ownership = np.zeros((samples, board_size, board_size), dtype=np.intc)
	value = np.zeros((samples, 1), dtype=np.intc)

	for s in range(samples):
		features[s], rules[s], policy[s], ownership[s], value[s] = string_frame_to_training_frame(frames[int(random()*len(frames))], board_size)

		if s % 1024 == 0:
			print(s, end=' ')
	print()

	# convert features from channels_first to channels_last
	features = np.moveaxis(features, 1, -1)
	policy = np.reshape(policy, (samples, board_size**2))

	return features, rules, policy, ownership, value

def game_record_to_training_strings_with_symmetries(gr):

	stringified_training_data = []

	player = 1
	game = Board(gr['boardsize'], gr['komi'])
	for target in range(len(gr['moves'])):

		# 1) generate model inputs and targets from the current state of the game
		instance_features, instance_rules = game_state_to_model_inputs(game, gr['boardsize'], player)
		instance_policy, instance_ownership, instance_value = game_record_to_model_outputs(gr, gr['moves'], target, player, gr['boardsize'])

		# 2) enumerate all the symmetries
		for _ in range(4):
			instance_features = np.rot90(instance_features, axes=(-2,-1))
			instance_policy = np.rot90(instance_policy, axes=(-2,-1))
			instance_ownership = np.rot90(instance_ownership, axes=(-2,-1))
			stringified_training_data.append(stringify_inputs_and_targets(gr, player, instance_features, instance_rules, instance_policy, instance_ownership, instance_value))

			flipped_features = np.flip(instance_features, axis=-1)
			flipped_policy = np.flip(instance_policy, axis=-1)
			flipped_ownership = np.flip(instance_ownership, axis=-1)
			stringified_training_data.append(stringify_inputs_and_targets(gr, player, flipped_features, instance_rules, flipped_policy, flipped_ownership, instance_value))

		# 3) advance the game state
		error_happen = not game.place_stone(_char_to_9x9_(gr['moves'][target]), player)
		if error_happen:
			raise Exception("oh shit, a happening!")
		player *= -1

	return stringified_training_data

if __name__ == "__main__":
	filename = 'modern 9x9 training frames'
	frames = [x for x in open(filename, 'r').read().split('\n') if len(x)]
	print(len(frames), 'training frames loaded')

	model = build_agz_model(6, 96, input_shape=(9, 9, 4))

	for e in range(0,128):
		features, rules, policy, ownership, value = create_training_dataset_from_frames(frames, 16384, 9)
		model.fit([features, rules], [policy, ownership, value], batch_size=128, epochs=1, verbose=1)
		save_model(model, "crappy go model" + str(e) + ".h5", save_format="h5")






