

class GameStatus:
	in_progress = 0
	player_1_wins = 1
	player_2_wins = 2
	nobody_wins = 3
	killed = 4


class TeachableGame:

	@staticmethod
	def get_feature_dimensions():
		raise Exception("not implemented")

	@staticmethod
	def get_action_space():
		raise Exception("not implemented")

	@staticmethod
	def get_name():
		raise Exception("not implemented")

	def get_state_as_features(self, player_to_move):
		raise Exception("not implemented")

	def get_move_legality(self, player):
		raise Exception("not implemented")

	def get_legal_moves(self, player):
		raise Exception("not implemented")

	def get_status(self):
		raise Exception("not implemented")

	def get_winner(self):
		raise Exception("not implemented")

	def copy(self):
		raise Exception("not implemented")

	def complete_as_rollout(self, player_to_move):
		raise Exception("not implemented")

	def zobrist_hash(self):
		raise Exception("not implemented")

	def zobrist_hash_for_child(self, move, player):
		raise Exception("not implemented")