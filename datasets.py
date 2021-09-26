
from os import path
import numpy as np
from numpy.random import random

from board import Board
from model import build_agz_model

def build_kgs_dataset(samples):
	def str_to_int(move):
		if move == "pass":
			return -1
		return 19*(ord(move[0])-ord('a')) + (ord(move[1])-ord('a'))

	games = [x.split(',') for x in open("go_kgs_6d+_games",'r').read().split('\n') if len(x) and 100 <= len(x.split(',')) < 200]
	print(len(games),"games loaded")
	print(sum([len(x) for x in games]),"moves in dataset")

	features = np.zeros((samples, 19, 19, 5), dtype=np.ubyte)
	policies = np.zeros((samples, 1), dtype=np.int)
	values = np.zeros((samples, 1), dtype=np.int)

	s = 0
	while s < samples:
		g = int(random()*len(games))
		m = int(random()*len(games[g]))

		if games[g][m] != "pass":
			board = Board(side=19)
			player = 1
			for mv in games[g][:m]:
				board.place_stone(str_to_int(mv), player)
				player *= -1

			features[s] = np.moveaxis(board.get_features(player), 0, -1)
			policies[s][0] = str_to_int(games[g][m])
			# values are ignored for now (they're all zeroes by default, so we're fine)

			s += 1
			if s % 100 == 0:
				print(samples, s)

	print(features.shape)

	return features, policies, values

def encode_state(state):
	encoded = ''
	for r in range(19):
		for c in range(19):
			numeric_value = 48 + (state[r][c][0]<<0) + (state[r][c][1]<<1) + (state[r][c][2]<<2) + (state[r][c][3]<<3) + (state[r][c][4]<<4)
			encoded += chr(numeric_value)
			# print(state[r][c], numeric_value)
	# print(encoded)
	return encoded

def decode_state(state):
	assert len(state) == 19**2

	decoded = np.zeros((19, 19, 5), dtype=np.ubyte)
	for i in range(len(state)):
		for ch in range(5):
			decoded[i//19][i%19][ch] = 1 if (ord(state[i])-48) & (1<<ch) else 0

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
	with open("encoded_kgs_dataset","w") as dataset:
		for g in games:
			board = Board(side=19)
			player = 1
			for mv in g:
				state = np.moveaxis(board.get_features(player), 0, -1)
				action = str_to_int(mv)
				dataset.write(encode_state(state) + '\t' + str(action) + '\n')

				if random() < 0.01:
					encoded = encode_state(state)
					decoded = decode_state(encoded)
					if np.array_equal(state, decoded):
						tests_passed += 1
					else:
						raise Exception("there's a problem with the codecs")

				board.place_stone(action, player)
				player *= -1

			games_written += 1
			moves_written += len(g)
			print("games written:",games_written,'/',total_games, "moves written:",moves_written,'/',total_moves, "tests passed:",tests_passed)

def prepare_dataset(samples):
	if not path.exists("encoded_kgs_dataset"):
		print("the kgs dataset has not been encoded -- this will take a while")
		encode_kgs_dataset()

	dataset = [line.split() for line in open("encoded_kgs_dataset","r").read().split('\n') if len(line.split()) == 2]

	features = np.zeros((samples, 19, 19, 5), dtype=np.ubyte)
	policies = np.zeros((samples, 1), dtype=int)
	values = np.zeros((samples, 1), dtype=int)

	for s in range(samples):
		index = int(random()*len(dataset))

		state = dataset[index][0]
		action = dataset[index][1]

		features[s] = decode_state(state)
		policies[s] = int(action)

	return features, policies, values

if __name__ == "__main__":

	blocks = 4
	channels = 32

	model = build_agz_model(blocks, channels, input_shape=(19, 19, 5))

	for e in range(1000):
		features, policies, values = prepare_dataset(2**20)
		model.fit(features, [policies, values], verbose=1, batch_size=128, epochs=1, validation_split=0.125)
		model.save(f"ag policy epochs={e+1}", save_format="h5")


