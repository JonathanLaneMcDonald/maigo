
import numpy as np

class ConnectFour:
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

	def __init__(self):

		self.grid = np.zeros((self.COLUMNS, self.ROWS), dtype=int)
		self.fullness = np.zeros(self.COLUMNS, dtype=int)

		self.status = ConnectFour.Status.in_progress

	def do_move(self, column, player):
		if self.fullness[column] < self.ROWS:
			self.grid[column][self.fullness[column]] = player
			self.check_state(column, self.fullness[column], player)
			self.fullness[column] += 1
			return True
		return False

	def get_legal_moves(self):
		return np.array([col for col in range(self.COLUMNS) if self.fullness[col] < self.ROWS])

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
		# vertical		(0,1 -> 0,-1)
		# horizontal	(-1,0 -> +1,0)
		# diagonal 		(-1,-1 -> +1,+1)
		# diagonal 		(-1,+1 -> +1,-1)

		if \
			1 + self.get_travel(move_col, move_row, 0,+1, player, 3) + self.get_travel(move_col, move_row, 0,-1, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1, 0, player, 3) + self.get_travel(move_col, move_row,+1, 0, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1,-1, player, 3) + self.get_travel(move_col, move_row,+1,+1, player, 3) >= 4 or \
			1 + self.get_travel(move_col, move_row,-1,+1, player, 3) + self.get_travel(move_col, move_row,+1,-1, player, 3) >= 4:
			if player == 1:
				self.status = ConnectFour.Status.player_1_wins
			elif player == 2:
				self.status = ConnectFour.Status.player_2_wins
		elif sum(self.get_legal_moves()) == 0:
			self.status = ConnectFour.Status.nobody_wins

	def display(self):
		print()
		print("cf.display()")
		print(np.flip(np.transpose(self.grid), axis=0))


from numpy.random import choice

if __name__ == "__main__":
	cf = ConnectFour()

	change = {1:2, 2:1}

	player = 1
	while cf.status == ConnectFour.Status.in_progress:
		legal_moves = cf.get_legal_moves()
		if len(legal_moves):
			normalized_legal_moves = np.array([1 for x in legal_moves]) / len(legal_moves)
			move = choice(legal_moves, p=normalized_legal_moves)
			success = cf.do_move(move, player)
			if not success:
				print("a thing happen!")
			print("legal moves:", legal_moves, "weights:", normalized_legal_moves, "selected move:", move, "player:", player)
			cf.display()
			player = change[player]






