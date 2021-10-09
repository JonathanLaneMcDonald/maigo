
import time
import xxhash

from copy import copy
import numpy as np
from game import TeachableGame, GameStatus

class ConnectFour(TeachableGame):
	"""
	let's start with a trivial game with a tiny action space
	"""

	COLUMNS = 7
	ROWS = 6

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

	@staticmethod
	def get_name():
		return f"ConnectFour_{ConnectFour.COLUMNS}x{ConnectFour.ROWS}"

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

	def get_move_legality(self, player):
		"""
		tell the caller which moves are legal
		"""
		return np.array([1 if self.fullness[col] < ConnectFour.ROWS else 0 for col in range(ConnectFour.COLUMNS)], dtype=int)

	def get_legal_moves(self, player):
		return [col for col in range(ConnectFour.COLUMNS) if self.fullness[col] < ConnectFour.ROWS]

	def get_status(self):
		return self.status

	def get_winner(self):
		if self.status == GameStatus.player_1_wins:
			return 1
		elif self.status == GameStatus.player_2_wins:
			return -1
		else:
			return 0

	def copy(self):
		return ConnectFour(self)

	def complete_as_rollout(self, player_to_move):
		game = self.copy()
		while game.status == GameStatus.in_progress:
			legal_moves = game.get_legal_moves(player_to_move)
			if len(legal_moves):
				normalized_legal_moves = np.array([1 for x in legal_moves]) / len(legal_moves)
				move = choice(legal_moves, p=normalized_legal_moves)
				success = game.do_move(move, player_to_move)
				if not success:
					raise Exception("failed to play a move!")
				player_to_move = -player_to_move
			else:
				raise Exception("there's no legal moves!")

		if game.status == GameStatus.player_1_wins:
			return 1
		elif game.status == GameStatus.player_2_wins:
			return -1
		else:
			return 0

	def zobrist_hash(self):
		game_representation = ' '.join([str(x) for x in np.reshape(self.grid, (self.COLUMNS*self.ROWS))])
		return xxhash.xxh64(game_representation.encode("utf-8")).hexdigest()

	def zobrist_hash_for_child(self, move, player):
		new_game = self.copy()
		new_game.do_move(move, player)
		return new_game.zobrist_hash()

	def __init__(self, other=None):
		super().__init__()

		if other is None:
			self.grid = np.zeros((self.COLUMNS, self.ROWS), dtype=int)
			self.fullness = np.zeros(self.COLUMNS, dtype=int)
			self.status = GameStatus.in_progress
		else:
			self.grid = copy(other.grid)
			self.fullness = copy(other.fullness)
			self.status = copy(other.status)

	def do_move(self, column, player):
		if self.fullness[column] < self.ROWS:
			self.grid[column][self.fullness[column]] = player
			self.check_state(column, self.fullness[column], player)

			self.fullness[column] += 1
			if not len(self.get_legal_moves(player)):
				self.status = GameStatus.nobody_wins

			return True
		return False

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
				self.status = GameStatus.player_1_wins
			elif player == -1:
				self.status = GameStatus.player_2_wins

	def display(self):
		print()
		print("cf.display()")
		print(np.flip(np.transpose(self.grid), axis=0))


from numpy.random import choice, random

def tree_test(simulations, random_proportional=True):

	class Node:
		def __init__(self, game: TeachableGame):
			self.game = game.copy()
			self.visits = 0
			self.victories = {x: 0 for x in {-1, 0, 1}}
			self.children = {p: {mv: game.zobrist_hash_for_child(mv, p) for mv in game.get_legal_moves(p)} for p in {-1, 1}}

	def register_state(state_library, game):
		child = game.zobrist_hash()
		if child not in state_library:
			state_library[child] = Node(game)
		return child

	def recurse_to_leaf(state_library, current_root, player_to_move, broadcast_recipients, current_depth, depth_limit):
		# the current node needs to know about the result of this evaluation
		broadcast_recipients.add(current_root)

		current_node = state_library[current_root]

		legal_moves = current_node.children[player_to_move].keys()

		values = {}
		for x in legal_moves:
			if current_node.children[player_to_move][x] in state_library:
				child = state_library[current_node.children[player_to_move][x]]
				values[x] = child.victories[player_to_move]/(1+child.visits) + (current_node.visits**0.5)/(1+child.visits)
			else:
				values[x] = current_node.visits**0.5

		move = sorted([(wr, mv) for mv, wr in values.items()])[-1][1]

		next_root_hash = current_node.children[player_to_move][move]

		if next_root_hash in state_library and state_library[next_root_hash].game.status == GameStatus.in_progress and current_depth < depth_limit:
			return recurse_to_leaf(state_library, next_root_hash, -player_to_move, broadcast_recipients, current_depth+1, depth_limit)
		else:
			broadcast_recipients.add(next_root_hash)
			return current_root, player_to_move, move, broadcast_recipients

	game = ConnectFour()
	state_library = {}
	register_state(state_library, game)
	current_root = game.zobrist_hash()

	player = 1
	move_stack = []
	while state_library[current_root].game.status == GameStatus.in_progress:
		# do some simulations to figure out what move to play
		for s in range(simulations[player]):
			recurse_root, player_to_move, move, broadcast_recipients = recurse_to_leaf(state_library, current_root, player, set(), 0, 10)

			new_game = state_library[recurse_root].game.copy()
			new_game.do_move(move, player_to_move)
			leaf = register_state(state_library, new_game)

			broadcast_recipients.add(leaf)

			rollout_result = state_library[leaf].game.complete_as_rollout(-player_to_move)

			for br in broadcast_recipients:
				state_library[br].visits += 1
				state_library[br].victories[rollout_result] += 1

		# select move ;P
		visits = {mv: 0 if h not in state_library else state_library[h].visits for mv, h in state_library[current_root].children[player].items()}
		move = sorted([(wr, mv) for mv, wr in visits.items()])[-1][1]

		weights = []
		if random_proportional:
			moves = [k for k, v in sorted(visits.items())]
			weights = np.array([v for k, v in sorted(visits.items())], dtype=float)
			weights /= max(weights)
			weights **= 2
			weights /= sum(weights)
			move = choice(moves, p=weights)

		move_stack.append(move)
		current_root = state_library[current_root].children[player][move]
		'''
		print('*'*80)
		print(len(state_library),"states in library")
		print(move_stack)
		if random_proportional:
			print(list(weights))
		print(visits, sum(visits.values()), "visits to this node")
		print(move)
		state_library[current_root].game.display()
		'''
		player = -player

	return move_stack, state_library[current_root].game.status


def play_games_with_models(games, model):

	trained_model_victories = 0
	random_model_victories = 0
	for g in range(games+1):

		player_map = {}
		if random() < 0.50:
			player_map = {1: "random", -1: "model"}
		else:
			player_map = {1: "model", -1: "random"}

		player = 1
		game = ConnectFour()
		while game.status == GameStatus.in_progress:
			legal_moves = game.get_legal_moves(player)

			# start with random selection (this will be right half the time :D)
			move = choice(legal_moves)
			if player_map[player] == "model":
				features = np.array([game.get_state_as_features(player)], dtype=np.ubyte)
				policy, value = model.predict(features)
				move_legality = game.get_move_legality(player)
				weighted_legal_moves = np.array([x*y for x,y in zip(policy[0], move_legality)], dtype=float)
				weighted_legal_moves /= max(weighted_legal_moves)
				weighted_legal_moves **= 2
				weighted_legal_moves /= sum(weighted_legal_moves)
				moves = list(range(len(weighted_legal_moves)))
				move = choice(moves, p=weighted_legal_moves)

			game.do_move(move, player)
			player = -player

		if game.status in {GameStatus.player_1_wins, GameStatus.player_2_wins}:
			winner = 1 if game.status == GameStatus.player_1_wins else -1
			if player_map[winner] == "model":
				trained_model_victories += 1
			elif player_map[winner] == "random":
				random_model_victories += 1
			else:
				print("something confusing is happening because nobody won?")

		if g % 100 == 0:
			print("model wins:", trained_model_victories, "random policy wins:", random_model_victories)

	return trained_model_victories / (trained_model_victories + random_model_victories)


def train_on_games():
	"""
	look in local directory for games,
	convert each game to state-action pairs,
	build a simple model,
	try to train it ;)

	later:
		set up a function that can play two models against each other just using raw policies and play like a bunch of games
		hopefully, the model that's trained will be the clear winner!
	"""

	import os
	from tqdm import tqdm
	from numpy.random import permutation

	from model import build_tree_policy

	known_games = []
	for f in [x for x in sorted(os.listdir('./')) if len(x) == 10 and x.isdigit()]:
		known_games += [x for x in open(f,'r').read().split('\n') if x.find('::') != -1]

	features = []
	policy = []
	value = []
	print("replaying games to create dataset")
	for line in tqdm(range(len(known_games))):
		moves = [int(x) for x in known_games[line].split('::')[0].split()]
		outcome = int(known_games[line].split('::')[1])

		player = 1
		game = ConnectFour()
		for mv in moves:
			features.append(game.get_state_as_features(player))
			policy.append(mv)
			value.append(outcome)

			game.do_move(mv, player)
			player = -player

	features = np.array(features, dtype=np.ubyte)
	policy = np.array(policy, dtype=int)
	value = np.array(value, dtype=int)

	print("features:",features.shape)
	print("policies:",policy.shape)
	print("values:",value.shape)

	sbox = permutation(list(range(features.shape[0])))

	p_features = np.zeros(features.shape, dtype=np.ubyte)
	p_policy = np.zeros(policy.shape, dtype=int)
	p_value = np.zeros(value.shape, dtype=int)

	print("permuting dataset")
	for s in tqdm(range(len(sbox))):
		p_features[s] = features[sbox[s]]
		p_policy[s] = policy[sbox[s]]
		p_value[s] = value[sbox[s]]

	model = build_tree_policy(
		blocks=5,
		filters=64,
		input_shape=ConnectFour.get_feature_dimensions(),
		policy_options=ConnectFour.get_action_space(),
		value_options=5
	)

	history = []
	for e in range(10):
		history.append(play_games_with_models(games=1000, model=model))
		print(history)
		model.fit(p_features, [p_policy, p_value], verbose=1, batch_size=128, epochs=1, validation_split=0.10)

if __name__ == "__main__":

	"""
	from model import build_tree_policy

	play_games_with_models(
		1000,
		build_tree_policy(
			blocks=4,
			filters=32,
			input_shape=ConnectFour.get_feature_dimensions(),
			policy_options=ConnectFour.get_action_space(),
			value_options=5
		)
	)
	exit()
	"""

	train_on_games()
	exit()

	import time

	start_time = time.time()
	save = open(str(int(start_time)), "w")
	for _ in range(100_000):
		simulations = 100
		moves, end_game_status = tree_test({1: simulations, -1: simulations})
		print(' '.join([str(x) for x in moves]) + '::' + str(end_game_status))
		save.write(' '.join([str(x) for x in moves]) + '::' + str(end_game_status) + '\n')
	exit()

