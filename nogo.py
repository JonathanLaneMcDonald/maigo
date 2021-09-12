
import numpy as np

class NoGo:
	"""
	I'm reviving the rlgo project. There's a useful paper, too, where they used NoGo as a model for computationally inexpensive zero-sum, perfect information games
	They demonstrated a better approach than AlphaZero because they won like 81:19 after 20,000 vs 120,000 games of self-play
	NoGo is a simple game with a low enough skill cap to see results, but a high enough skill cap to see progress and not be trivial
	I thought about it a little and it should also be much cheaper to implement than Go because no stone ever moves ;)

	NoGo rules:
		same as Go, but:
			captures are illegal
			territory doesn't matter
			first player to run out of legal moves loses

	So I should be able to copy over the Board class and implement a much cheaper function to track legal moves,
	so i can cut down the class i just copied over

	"""

	class Ko:
		def __init__(self, position, restricted_player):
			self.ko_position = position
			self.restricted_player = restricted_player

	class FloodFillResult:
		def __init__(self, group_id):
			self.group_id = group_id
			self.group_stones = set()
			self.group_liberties = set()
			self.adjacent_groups = set()

	def __init__(self, side=9, komi=6.5):

		self.side = side
		self.area = side**2
		self.komi = komi
		self.ko = None

		self.board = np.zeros(self.area, dtype=np.intc)+self.area
		self.ownership = np.zeros(self.area, dtype=np.intc)
		self.liberties = np.zeros(self.area, dtype=np.intc)

		self.legal_for_black = np.ones(self.area, dtype=np.intc)
		self.legal_for_white = np.ones(self.area, dtype=np.intc)

		self.cell_visits = np.zeros(self.area, dtype=np.intc)

		self.setup_neighbors()

	def get_cell_visits(self):
		return self.cell_visits

	def get_neighbors_at_position(self, board_index):
		return self.neighbors[board_index]

	def setup_neighbors(self):
		self.neighbors = []
		for r in range(self.side):
			for c in range(self.side):
				neighbor_list = []
				if 0 <= r - 1:			neighbor_list.append(self.side * (r - 1) + c)
				if r + 1 < self.side:	neighbor_list.append(self.side * (r + 1) + c)
				if 0 <= c - 1:			neighbor_list.append(self.side * r + (c - 1))
				if c + 1 < self.side:	neighbor_list.append(self.side * r + (c + 1))
				self.neighbors.append(neighbor_list)

	def place_stone(self, move, player):

		self.cell_visits = np.zeros(self.area, dtype=np.intc)

		if move == -1 or move == self.area or move == None:
			self.unregister_ko()
			return True
		elif move < 0 or self.area < move:
			return False

		if (player == 1 and self.legal_for_black[move] == 1) or (player == -1 and self.legal_for_white[move] == 1):
			liberties_needing_review = set()

			# we already know this is legal because we're tracking legal moves, so go ahead and transfer ownership
			self.board[move] = move
			self.ownership[move] = player
			liberties_needing_review.update(self.floodfill(move).group_liberties)

			number_of_stones_removed = set()
			neighboring_stones_to_investigate = {x for x in self.get_neighbors_at_position(move) if self.board[x] < self.area and self.ownership[self.board[x]] == -player}
			neighboring_groups_to_investigate = {self.board[x] for x in neighboring_stones_to_investigate}
			for pos in neighboring_groups_to_investigate:
				ffr = self.floodfill(pos)
				if len(ffr.group_liberties) == 0:
					number_of_stones_removed.update(ffr.group_stones)

					for removed_stone in ffr.group_stones:
						self.board[removed_stone] = self.area

					liberties_needing_review.update(ffr.group_stones)

					for group_id in ffr.adjacent_groups:
						liberties_needing_review.update(self.floodfill(group_id).group_liberties)
				else:
					liberties_needing_review.update(ffr.group_liberties)

			self.legal_for_black[move] = 0
			self.legal_for_white[move] = 0

			for liberty in liberties_needing_review:
				self.determine_legality_at_position(liberty)

			self.unregister_ko()
			if len(number_of_stones_removed) == 1 and len(self.floodfill(move).group_stones) == 1:
				self.register_ko(list(number_of_stones_removed)[0], -player)

			return True
		else:
			return False

	def register_ko(self, ko_position, restricted_player):
		self.ko = Board.Ko(ko_position, restricted_player)
		self.determine_legality_at_position(ko_position)

	def unregister_ko(self):
		if self.ko != None:
			old_ko_position = self.ko.ko_position
			self.ko = None
			self.determine_legality_at_position(old_ko_position)

	def violating_ko(self, position, player):
		return self.ko != None and self.ko.ko_position == position and self.ko.restricted_player == player

	def position_is_legal_for_player(self, position, player):
		friends_to_join = [x for x in self.get_neighbors_at_position(position) if self.board[x] < self.area and self.ownership[self.board[x]] == player and len(self.floodfill(x).group_liberties) >= 2]
		baddies_to_kill = [x for x in self.get_neighbors_at_position(position) if self.board[x] < self.area and self.ownership[self.board[x]] == -player and len(self.floodfill(x).group_liberties) == 1]
		return not self.violating_ko(position, player) and (len(friends_to_join) + len(baddies_to_kill)) != 0

	def determine_legality_at_position(self, position):

		# TODO: write a test to see if this function, as currently written, incorrectly marks a position as legal just because there are spaces nearby
		# TODO: for example, [position] might already have a stone, which would make it illegal, but i don't seem to be paying attention to that
		if len([x for x in self.get_neighbors_at_position(position) if self.board[x] == self.area or self.ownership[self.board[x]] == 0]):
			self.legal_for_black[position] = 1
			self.legal_for_white[position] = 1
		else:
			if self.position_is_legal_for_player(position, 1):
				self.legal_for_black[position] = 1
			else:
				self.legal_for_black[position] = 0

			if self.position_is_legal_for_player(position, -1):
				self.legal_for_white[position] = 1
			else:
				self.legal_for_white[position] = 0

	def floodfill(self, move):

		ffr = Board.FloodFillResult(move)

		to_visit = [move]
		visited = [False]*self.area
		while len(to_visit):
			position = to_visit.pop()

			if not visited[position]:
				if self.area <= self.board[position] or self.ownership[self.board[position]] == 0:
					ffr.group_liberties.add(position)
				elif self.ownership[self.board[position]] == self.ownership[move]:
					self.board[position] = move
					ffr.group_stones.add(position)
					to_visit += [x for x in self.get_neighbors_at_position(position) if not visited[x]]
				else:
					ffr.adjacent_groups.add(self.board[position])

				self.cell_visits[position] += 1

			visited[position] = True

		if self.board[move] < self.area and self.ownership[self.board[move]]:
			self.liberties[self.board[move]] = len(ffr.group_liberties)

		return ffr

	def get_sensible_moves_for_player(self, player):
		if player == 1:
			return {x for x in range(self.area) if self.legal_for_black[x] == 1 and (self.legal_for_white[x] == 1 or self.move_is_suicide_for_player(x, -1))}.union({self.area})
		if player == -1:
			return {x for x in range(self.area) if self.legal_for_white[x] == 1 and (self.legal_for_black[x] == 1 or self.move_is_suicide_for_player(x, 1))}.union({self.area})

	def move_is_suicide_for_player(self, position, player):

		# TODO: test if this fails when there is an enemy group that can be captured
		my_neighboring_stones = {x for x in self.get_neighbors_at_position(position) if self.board[x] < self.area and self.ownership[self.board[x]] == player}
		my_neighboring_groups = {self.board[x] for x in my_neighboring_stones if self.liberties[self.board[x]] > 1}
		return len(my_neighboring_stones) and not len(my_neighboring_groups)

	def get_simple_terminal_score_and_ownership(self):
		''' getting the simple terminal score means you've satisfied the following conditions:
			1) you've sampled "sensible" moves, meaning you'll do every move except filling your own eyes, and
			2) you've played until there have been two consecutive passes, meaning...
			The two players have exhausted all moves, all groups are connected, and there are no shared liberties, so
			It should be possible to count the score by checking a very simple condition, as follows'''
		simple_ownership = [1 if (self.board[x] < self.area and self.ownership[self.board[x]] == 1) or self.legal_for_black[x] == 1 else -1 for x in range(self.area)]
		simple_score = sum(simple_ownership) - self.komi
		return simple_ownership, simple_score

	def game_has_ended(self):
		return self.get_sensible_moves_for_player(1) == set() or self.get_sensible_moves_for_player(-1) == set()

	def is_black(self, position):
		return 0 <= position < self.area and self.board[position] < self.area and self.ownership[self.board[position]] == 1

	def is_white(self, position):
		return 0 <= position < self.area and self.board[position] < self.area and self.ownership[self.board[position]] == -1

	def is_legal_for_black(self, position):
		return self.legal_for_black[position] == 1

	def is_legal_for_white(self, position):
		return self.legal_for_white[position] == 1

	def get_liberties_for_position(self, position):
		if self.board[position] < self.area and self.ownership[self.board[position]]:
			return self.liberties[self.board[position]]
		else:
			return 0

	def display(self, move=None):
		string = ''
		for i in range(self.area):
			prefix, suffix = ' ', ' '
			if i == move:
				prefix, suffix = '(', ')'

			this_position = None
			if self.board[i] < self.area and self.ownership[self.board[i]]:
				this_position = self.ownership[self.board[i]]

			if this_position == 1:		string += prefix + 'X' + suffix
			elif this_position == -1:	string += prefix + 'O' + suffix
			else:						string += prefix + '.' + suffix

			if (i+1) % self.side == 0:
				string += '\n'

		return '*'*self.side*3+'\n'+string+'\n'
