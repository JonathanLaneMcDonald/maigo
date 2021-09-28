
import time
import os
from os import path

import numpy as np
from numpy.random import random

from board import Board
from model import build_agz_model, build_rollout_policy


def encode_state(state, channels):
	encoded = ''
	for r in range(19):
		for c in range(19):
			numeric_value = 48
			for ch in range(channels):
				numeric_value += (state[r][c][ch] << ch)
			encoded += chr(numeric_value)
	return encoded


def decode_state(state, channels):
	assert len(state) == 19**2

	decoded = np.zeros((19, 19, channels), dtype=np.ubyte)
	for i in range(len(state)):
		for ch in range(channels):
			decoded[i//19][i%19][ch] = 1 if (ord(state[i])-48) & (1 << ch) else 0

	return decoded


def encode_kgs_dataset():
	if path.exists("encoded_kgs_dataset"):
		print("the kgs dataset already exists -- skipping")
		return

	def str_to_int(move):
		if move == "pass":
			return 19**2
		return 19*(ord(move[0])-ord('a')) + (ord(move[1])-ord('a'))

	games = [x.split(',') for x in open("go_kgs_6d+_games",'r').read().split('\n') if len(x) and 100 <= len(x.split(','))]
	total_games = len(games)
	total_moves = sum([len(x) for x in games])

	games_written = 0
	moves_written = 0
	tests_passed = 0
	total_tests = 0
	with open("encoded_kgs_dataset","w") as dataset:
		for g in games:
			board = Board(side=19)
			player = 1
			for mv in g:
				stone_positions, legal_moves, liberties = board.get_features()

				encoded_stone_positions = encode_state(np.moveaxis(stone_positions, 0, -1), 2)
				encoded_legal_moves = encode_state(np.moveaxis(legal_moves, 0, -1), 2)
				encoded_liberties = encode_state(np.moveaxis(liberties, 0, -1), 6)

				action = str_to_int(mv)

				encoded_state_action = encoded_stone_positions + '\t' + encoded_legal_moves + '\t' + encoded_liberties + '\t' + str(player) + '\t' + str(action)

				dataset.write(encoded_state_action + '\n')

				if random() < 0.01:
					if (np.moveaxis(stone_positions, 0, -1) == decode_state(encoded_stone_positions, 2)).all() and \
						(np.moveaxis(legal_moves, 0, -1) == decode_state(encoded_legal_moves, 2)).all() and \
						(np.moveaxis(liberties, 0, -1) == decode_state(encoded_liberties, 6)).all():
							tests_passed += 1
					total_tests += 1

				board.place_stone(action, player)
				player *= -1

			games_written += 1
			moves_written += len(g)
			print("games written:",games_written,'/',total_games, "moves written:",moves_written,'/',total_moves, "tests:",total_tests,"passing:",tests_passed,"all passing:",total_tests==tests_passed)


def prepare_dataset(samples):
	if not path.exists("encoded_kgs_dataset"):
		print("the kgs dataset has not been encoded -- this will take a while")
		encode_kgs_dataset()

	dataset = [line.split() for line in open("encoded_kgs_dataset","r").read().split('\n') if len(line.split()) == 5]
	print(len(dataset),"items in dataset")

	features = np.zeros((samples, 19, 19, 11), dtype=np.ubyte)
	policies = np.zeros((samples, 1), dtype=int)
	values = np.zeros((samples, 1), dtype=int)

	for s in range(samples):
		index = int(random()*len(dataset))

		# 2 channels
		stone_positions = decode_state(dataset[index][0], 2)

		# 2 channels
		legal_moves = decode_state(dataset[index][1], 2)

		# 6 channels
		liberties = decode_state(dataset[index][2], 6)

		# 1 channel
		player = np.ones((19, 19, 1), dtype=np.ubyte) if int(dataset[index][3]) == 1 else np.zeros((19, 19, 1), dtype=np.ubyte)

		action = dataset[index][4]

		features[s] = np.concatenate((stone_positions, legal_moves, liberties, player), axis=-1)

		policies[s] = int(action)

		if s % 10000 == 0:
			print(s, samples)

	return features, policies, values


def convert_dataset_to_memory_mappable(slice_size):
	if not path.exists("encoded_kgs_dataset"):
		print("the kgs dataset has not been encoded -- this will take a while")
		encode_kgs_dataset()

	"""
	i'll start simple, but i want to try to create a simple, general way to refer to features so i can easily create datasets
	0,1 = black stones, white stones
	2,3 = legal for black, legal for white
	4,5,6 = liberties for black
	7,8,9 = liberties for white
	10 = player to move (black=1, white=0)	
	"""

	BLACK = 1
	WHITE = -1

	def extract_feature_for_player_from_step(game_frames, feature_index, number_of_channels, feature_selector, step):
		if len(game_frames) < abs(step):
			# print(f"cannot read {step} as frame stack only has depth {len(game_frames)} -- setting step to -1")
			step = -1
		features = np.moveaxis(decode_state(game_frames[step][feature_index], number_of_channels), -1, 0)
		return np.reshape(features[feature_selector], (19, 19, 1))

	def get_stones_for_player_from_step(game_frames, feature_owner, step=-1):
		return extract_feature_for_player_from_step(
			game_frames=game_frames,
			feature_index=0,
			number_of_channels=2,
			feature_selector=0 if feature_owner == 1 else 1,
			step=step
		)

	def get_legal_moves_for_player_from_step(game_frames, feature_owner, step=-1):
		return extract_feature_for_player_from_step(
			game_frames=game_frames,
			feature_index=1,
			number_of_channels=2,
			feature_selector=0 if feature_owner == 1 else 1,
			step=step
		)

	def get_liberties_for_player_from_step(game_frames, feature_owner, liberty_count, step=-1):
		return extract_feature_for_player_from_step(
			game_frames=game_frames,
			feature_index=2,
			number_of_channels=6,
			feature_selector=(liberty_count-1) + (0 if feature_owner == 1 else 3),
			step=step
		)

	def get_player(game_frames):
		return int(game_frames[-1][3])

	def get_player_plane(game_frames):
		if get_player(game_frames) == 1:
			return np.ones((19, 19, 1), dtype=np.ubyte)
		elif get_player(game_frames) == -1:
			return np.zeros((19, 19, 1), dtype=np.ubyte)
		else:
			raise Exception("looks like the player thing is messed up in your data, dude")

	def get_move(game_frames):
		return np.array([int(game_frames[-1][4])], dtype=int)

	features = []
	rollout_features = []
	policies = []

	slice_number = 0

	game_frames = []
	game_counter = 0
	frame_counter = 0
	with open("encoded_kgs_dataset","r") as dataset:
		for line in dataset:
			# try not to shit yourself if you reach eof
			if len(line.split()) == 5:
				frame = line.split()

				if set(list(frame[0])) == {"0"}:
					frame_counter += len(game_frames)
					print("games encoded:", game_counter, "frames encoded:", frame_counter)
					game_counter += 1
					game_frames = []

				game_frames.append(frame)

				state = []
				state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=BLACK, step=-1))
				state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=WHITE, step=-1))

				state.append(get_legal_moves_for_player_from_step(game_frames=game_frames, feature_owner=BLACK))
				state.append(get_legal_moves_for_player_from_step(game_frames=game_frames, feature_owner=WHITE))

				state.append(get_player_plane(game_frames=game_frames))

				features.append(np.concatenate(state, axis=-1))

				state = []
				if get_player(game_frames=game_frames) == 1:
					state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=BLACK, step=-1))
					state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=WHITE, step=-1))
				else:
					state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=WHITE, step=-1))
					state.append(get_stones_for_player_from_step(game_frames=game_frames, feature_owner=BLACK, step=-1))

				rollout_features.append(np.concatenate(state, axis=-1))

				policies.append(get_move(game_frames=game_frames))

				if len(features) == slice_size:
					assert len(features) == len(policies)

					slice_number += 1
					np.save(f"mappable features -- slice {slice_number} -- {slice_number * slice_size}", np.array(features, dtype=np.ubyte))
					np.save(f"mappable rollout features -- slice {slice_number} -- {slice_number * slice_size}", np.array(rollout_features, dtype=np.ubyte))
					np.save(f"mappable policies -- slice {slice_number} -- {slice_number * slice_size}", np.array(policies, dtype=int))

					features = []
					rollout_features = []
					policies = []


def prepare_dataset_from_mappable(samples, channels):

	feature_files = sorted([x for x in os.listdir('./') if x.find('mappable features') != -1])
	rollout_files = sorted([x for x in os.listdir('./') if x.find('mappable rollout features') != -1])
	policy_files = sorted([x for x in os.listdir('./') if x.find('mappable policies') != -1])

	mappable_features = [np.load(x, mmap_mode="r") for x in feature_files]
	mappable_rollouts = [np.load(x, mmap_mode="r") for x in rollout_files]
	mappable_policies = [np.load(x, mmap_mode="r") for x in policy_files]

	memory_mapped_features = sum([x.shape[0] for x in mappable_features])
	memory_mapped_rollouts = sum([x.shape[0] for x in mappable_rollouts])
	memory_mapped_policies = sum([x.shape[0] for x in mappable_policies])

	print("memory mapped features:", ' '.join([str(x.shape) for x in mappable_features]), memory_mapped_features)
	print("memory mapped rollouts:", ' '.join([str(x.shape) for x in mappable_rollouts]), memory_mapped_rollouts)
	print("memory mapped policies:", ' '.join([str(x.shape) for x in mappable_policies]), memory_mapped_policies)

	def get_index(mappable, index):
		for i in mappable:
			if index < i.shape[0]:
				return i[index]
			index -= i.shape[0]
		raise Exception("we shouldn't be getting to here")

	features = np.zeros((samples, 19, 19, channels), dtype=np.ubyte)
	rollouts = np.zeros((samples, 19, 19, 3), dtype=np.ubyte)
	policies = np.zeros((samples, 1), dtype=int)
	values = np.zeros((samples, 1), dtype=int)

	for s in range(samples):
		index = int(random()*memory_mapped_features)
		features[s] = get_index(mappable_features, index)
		rollouts[s] = get_index(mappable_rollouts, index)
		policies[s] = get_index(mappable_policies, index)

	return features, rollouts, policies, values


if __name__ == "__main__":

	"""
	can i learn a sudoku solver through this kind of policy/value mcts based reinforcement learning?
		you'll basically always solve a sudoku puzzle and i guess the big challenge is how many auto-regresses you have to do to get there...
		like... you can't not solve a sudoku puzzle... this is really interesting all the sudden...
		maybe you formulate the problem like this...
			3D binary feature space like last time,
			a tower of 3D convolutions,
			project into like 3Dx2 space or something,
			flatten,
			project onto a 729 dimensional output (r,c,value)
		maybe start with almost-solved puzzles and blank out illegal moves (moves that are already filled in) to speed things up
		gonna need some kind of replay buffer, too... this could get really interesting...
	"""

	slice_size = 2**20
	convert_dataset_to_memory_mappable(slice_size)
	exit()

	input_channels = 11
	samples = 2**20

	tree_batch_size = 128
	tree_policy_blocks = 10
	tree_policy_channels = 128

	rollout_batch_size = 128
	rollout_policy_blocks = 4
	rollout_policy_channels = 32

	model = build_agz_model(tree_policy_blocks, tree_policy_channels, input_shape=(19, 19, input_channels))
	rollout = build_rollout_policy(rollout_policy_blocks, rollout_policy_channels, input_shape=(19, 19, 3))

	for e in range(10):
		start_time = time.time()
		features, rollouts, policies, values = prepare_dataset_from_mappable(samples, input_channels)
		total_time = time.time() - start_time

		print("samples generated per second:", samples / total_time)

		rollout.fit(rollouts, policies, verbose=1, batch_size=rollout_batch_size, epochs=1, validation_split=0.125)

		model.fit(features, [policies, values], verbose=1, batch_size=tree_batch_size, epochs=1, validation_split=0.125)
		model.save(f"ag policy epochs={e+1}", save_format="h5")



