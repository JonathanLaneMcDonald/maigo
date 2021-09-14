
from connect_four import ConnectFour

class McTuss:
	"""
	i'll try and sketch out some steps for setup and steady-state operation of the mcts...

	tree is created:
		use model to predict policy/value for root node

	are we in steady-state now?

	steady-state:
		do some simulations:
			for each simulation:
				recurse to leaf:
					establish graph walker by pointing at current play root
					while graph walker is not leaf:
						do p-uct to select the next node:
							todo
								when a node is established, we:
									1) ask the game for a representation of the current game state
									2) do an inference to predict policies for all the children and the value of the current node
									3) we cross-reference the policy with legal moves to define:
										i) children of this node {mv: Node}, and
										ii) policy of child {mv: float (normalized to 1 over legal moves)}, and
							todo
								to calculate p-uct score, we need value, policy, N(c'), and N(c)
								value and N(c') are stored inside the child node
								policy and N(c) are stored in the parent node
								i can tell if the current node has been expanded by seeing if its simulation count is greater than zero
								if the current node has been expanded:
									score = {}
									for each child in self.children.keys():
										value = parent.value if self.children[key] is None else self.children[key].value
										policy = parent.policies[key] (because this exists for every legal move)
										N(c') = 0 if self.children[key] is None else self.children[key].simulations
										N(c) = parent.simulations
										score[key] = value + policy * sqrt(N(c') / (1+N(c)))
								else:
									the current node is the one we should expand
						maybe push each node in the branch to a stack so we can easily replay the results later

				expand leaf / do inference:
					copy the game state over from the parent and perform the move that got us to this point
					ask the game for a representation of the game board and for a set of legal moves
					we'll do inference and create the dictionaries we need and that'll be that

				propagate signal back up the branch:
					maybe have it so the model always predicts 1 if player 1 should win or 0 if player 2 should win
					that way, you can basically say if player == 1, then value += value, simulations += 1
					else, if player == 2, then value += (1-value), simulations += 1
					or something like that

		after simulations are complete, select the move to do:
			take a weighted random sample across all legal moves with the total number of simulations normalized to 1.0

		if a game goes longer than expected, we can expedite it by:
			sampling the raw policy of the network to drive the game to conclusion and get a winner
			we still track the game in mcts, but expand only the nodes we need to in order to follow the branch

		connect four should be a great test game for this! it's simple, yet non-trivial :D

		i wonder if i should go for NoGo after connect four or if I should go straight for Go?

	"""

	class Node:
		def __init__(self, parent, player_to_move, completed_move, game_state: ConnectFour, model):
			self.parent = parent

			self.move_completed_to_get_here = completed_move
			self.player_to_move = player_to_move

			self.value_of_my_subtree = 0
			self.simulations_in_my_subtree = 0

			self.game_state = game_state

			# mv: Node
			self.child_nodes = {}
			self.initialize_children()

			# mv: policy
			self.child_policy = {}
			self.initialize_policy_and_value(model)

		def initialize_children(self):
			self.child_nodes = {mv: None for mv in self.game_state.get_legal_moves()}

		def initialize_policy_and_value(self, model):
			model_inputs = self.game_state.as_model_features(self.player_to_move)
			policy, value = model.predict(model_inputs)
			sum_of_weights = sum([policy[p] for p in range(len(policy)) if p in self.child_nodes.keys()])
			self.child_policy = {p: policy[p]/sum_of_weights for p in range(len(policy)) if p in self.child_nodes.keys()}
			self.value_of_my_subtree = value
			self.simulations_in_my_subtree = 1

		def update_value(self, additional_value_in_subtree):
			self.value_of_my_subtree += additional_value_in_subtree
			self.simulations_in_my_subtree += 1

	def __init__(self, model):

		self.game_root = self.create_node(None, 1, None, ConnectFour(), model)
		self.play_root = self.game_root

	def create_node(self, parent, player_to_move, completed_move, game_state, model):
		return McTuss.Node(parent, player_to_move, completed_move, game_state, model)

	def simulate(self, n_simulations):
		for _ in range(n_simulations):
			leaf = self.recurse_to_and_expand_leaf(self.play_root)

	def recurse_to_and_expand_leaf(self, node):
		walker = node

		keep_walking = True
		while keep_walking:
			



