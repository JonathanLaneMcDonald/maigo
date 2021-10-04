

class TeachableGame:

	@staticmethod
	def get_feature_dimensions():
		raise Exception("not implemented")

	@staticmethod
	def get_action_space():
		raise Exception("not implemented")

	def get_state_as_features(self, player_to_move):
		raise Exception("not implemented")

	def get_move_legality(self):
		raise Exception("not implemented")
