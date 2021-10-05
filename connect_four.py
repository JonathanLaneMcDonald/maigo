
import time

from copy import copy
import numpy as np
from game import TeachableGame

class ConnectFour(TeachableGame):
	"""
	let's start with a trivial game with a tiny action space
	"""

	COLUMNS = 7
	ROWS = 6

	class Status:
		in_progress = 0
		player_1_wins = 1
		player_2_wins = 2
		nobody_wins = 3
		killed = 4

	@staticmethod
	def get_feature_dimensions():
		"""
		we've got 3 dimensions:
			columns,
			rows,
			feature planes {player 1 pieces, player 2 pieces, player to move}
		"""
		return ConnectFour.COLUMNS, ConnectFour.ROWS, 3

	@staticmethod
	def get_action_space():
		"""
		we can place a stone in any column (in principle)
		"""
		return ConnectFour.COLUMNS

	def get_state_as_features(self, player_to_move):
		"""
		convert state to input features so we can do inference
		"""
		features = np.zeros((ConnectFour.COLUMNS, ConnectFour.ROWS, 3), dtype=np.ubyte)
		for c in range(ConnectFour.COLUMNS):
			for r in range(ConnectFour.ROWS):
				if self.grid[c][r] == 1:
					features[c][r][0] = 1
				if self.grid[c][r] == 2:
					features[c][r][1] = 1
				if player_to_move == 1:
					features[c][r][2] = 1
		return features

	def get_move_legality(self):
		"""
		tell the caller which moves are legal
		"""
		return np.array([1 if self.fullness[col] < ConnectFour.ROWS else 0 for col in range(ConnectFour.COLUMNS)], dtype=int)

	def __init__(self, other=None):
		super().__init__()

		if other is None:
			self.grid = np.zeros((self.COLUMNS, self.ROWS), dtype=int)
			self.fullness = np.zeros(self.COLUMNS, dtype=int)
			self.status = ConnectFour.Status.in_progress
		else:
			self.grid = copy(other.grid)
			self.fullness = copy(other.fullness)
			self.status = copy(other.status)

	def do_move(self, column, player):
		if self.fullness[column] < self.ROWS:
			self.grid[column][self.fullness[column]] = player
			self.check_state(column, self.fullness[column], player)

			self.fullness[column] += 1
			if not len(self.get_legal_moves()):
				self.status = ConnectFour.Status.nobody_wins

			return True
		return False

	def get_legal_moves(self):
		return [col for col in range(ConnectFour.COLUMNS) if self.fullness[col] < ConnectFour.ROWS]

	def get_travel(self, start_col, start_row, col_delta, row_delta, player, max_steps):
		player_pieces = 0
		for i in range(1, max_steps+1):
			if \
				0 <= start_col+(i*col_delta) < self.COLUMNS and \
				0 <= start_row+(i*row_delta) < self.ROWS and \
				self.grid[start_col+(i*col_delta)][start_row+(i*row_delta)] == player:
				player_pieces += 1
			else:
				return player_pieces
		return player_pieces

	def check_state(self, move_col, move_row, player):
		"""check vertical, horizontal, and both diagonals -- walk up to 3 in each of the 8 directions"""
		if \
			1 + self.get_travel(move_col, move_row, 0,+1, player, 3) + self.get_travel(move_col, move_row, 0,-1, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1, 0, player, 3) + self.get_travel(move_col, move_row,+1, 0, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1,-1, player, 3) + self.get_travel(move_col, move_row,+1,+1, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1,+1, player, 3) + self.get_travel(move_col, move_row,+1,-1, player, 3) >= 4:
			if player == 1:
				self.status = ConnectFour.Status.player_1_wins
			elif player == -1:
				self.status = ConnectFour.Status.player_2_wins

	def display(self):
		print()
		print("cf.display()")
		print(np.flip(np.transpose(self.grid), axis=0))


from numpy.random import choice, random

def play_a_game():
	cf = ConnectFour()

	player = 1
	move_count = 0
	while cf.status == ConnectFour.Status.in_progress:
		legal_moves = cf.get_legal_moves()
		if len(legal_moves):
			normalized_legal_moves = np.array([1 for x in legal_moves]) / len(legal_moves)
			move = choice(legal_moves, p=normalized_legal_moves)
			success = cf.do_move(move, player)
			if not success:
				raise Exception("a happening!")
			else:
				move_count += 1
			#print("legal moves:", legal_moves, "weights:", normalized_legal_moves, "selected move:", move, "player:", player)
			#cf.display()
			player = -player
		else:
			raise Exception("another happening!")

	return move_count, cf.status


class Orchestrator:
	"""
	Context:
		the bottleneck right now is batching for inference. the numbers look like this for a 5x64 (the model size matters like +/- 10%)
			batch size		features=np.zeros()		features=ConnectFour.as_features()
			1				44						45
			16				716						691
			128				5353					4334
			256				8903					6300
			1024			17000					9800
		so the process of sending a batch to the cpu/gpu for inference has some intrinsic overhead and we need large batch sizes to make it economical

	So the Orchestrator is here to help me automate some of this stuff...

	It'll manage the tree search, then, when a game ends, it'll clean up the tree search, get the moves, submit them to the replay buffer, then start a new game
	"""
	...


class ModelCoach:
	"""
	this is where everything gets tied together

	breakfast of champions:
		load or initialize a model
		load or initialize a replay buffer

		batch_size = 128 (or something)
		checkpoint_frequency = 100 (or whatever)

		initialize an array of [batch_size] Orchestrators

		next_training_target = replay_buffer.size()//batch_size + 1
		while True (basically):
			features = np.zeros((batch_size, 8, 8, 3), dtype=np.ubyte)
			for o in range(len(orchestrators)):
				features[o] = orchestrators[o].get_next_inference_job()

			inferences = model.predict(features)
			for o in range(len(orchestrators)):
				orchestrators[o].do_something_with_this_result()

			if next_training_target <= replay_buffer.size():
				training_features, training_policies, training_values = replay_buffer.get_training_batch()
				model.fit(training_features, [training_policies, training_values], verbose=1, batch_size=batch_size, epochs=1)
				if next_training_target % checkpoint_frequency == 0:
					model.save(whatever the checkpoint number is)
					TournamentFacilitator.have_a_tournament(model_prefix="cf model", games=10_000, report_name=f"cf model tournament at {next_training_target}")
				model.save("current model")
				next_training_target = replay_buffer.size()//batch_size + 1

	"""
	...

def tf_test():

	from tensorflow.keras.models import Model
	from tensorflow.keras.layers import Input, Convolution2D, Convolution1D, BatchNormalization, Flatten, Dense, Reshape
	from tensorflow.keras.optimizers import Adam

	blocks = 5
	filters = 64
	input_shape = (9, 9, 3)

	inputs = Input(shape=input_shape)

	x = inputs
	for block in range(blocks):
		x = Convolution2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
		x = BatchNormalization()(x)

	x = Reshape((9, 9*filters))(x)

	x = Convolution1D(filters=9, kernel_size=1, padding='same', activation='relu')(x)

	x = Flatten()(x)

	outputs = Dense(9, activation='softmax')(x)

	model = Model(inputs, outputs)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
	model.summary()

	multiplier = 128

	cf = ConnectFour()
	start_time = time.time()
	for i in range(2**10):
		features = np.zeros((multiplier, 9, 9, 3), dtype=np.ubyte)
		for j in range(multiplier):
			features[j] = cf.get_state_as_features(1)
		prediction = model.predict(features)

		if i and i % 100 == 0:
			print(i, (multiplier*100)/(time.time()-start_time))
			start_time = time.time()


if __name__ == "__main__":

	#tf_test()
	#exit()

	start_time = time.time()
	move_counts = []
	status = {k:0 for k in range(5)}
	for i in range(1_000_000+1):
		mv, st = play_a_game()
		move_counts.append(mv)
		status[st] += 1
		if i and i % 10_000 == 0:
			print(i, status, i/(time.time()-start_time), sum(move_counts)/(time.time()-start_time))
	print(np.average(move_counts), np.std(move_counts))
	print(status)



