
from model_manager import ModelManager

def do_the_thing(to_model, from_model):
	# probably pretty dumb to just construct the thing and never do anything like own or join the process...
	# just trying to get the thing working for now ;D
	model_manager = ModelManager((4, 32), to_model, from_model)

if __name__ == "__main__":
	from multiprocessing import SimpleQueue, Process

	import time
	from datasets import _smallnum_to_char_, create_game_record
	from board import Board
	from mcts import MCTS

	queue_to_model = SimpleQueue()
	queue_from_model = SimpleQueue()

	process = Process(target=do_the_thing, args=(queue_to_model, [queue_from_model]))
	process.start()

	start_time = time.time()
	games = 1
	diagnostics_total_cell_visits = {}
	for g in range(games):
		moves = []
		visits = []
		game = Board()
		mcts = MCTS(game, 0, queue_to_model, queue_from_model)
		game_in_progress = True
		while game_in_progress:
			mcts.simulate(searches=1)

			player_to_move = 'Player to move: black' if mcts.get_player_to_move() == 1 else 'Player to move: white'
			nodes_in_graph = ', Nodes:' + str(mcts.get_node_count())
			observable_sims= ', Observable Sims:' + str(mcts.get_outstanding_sims())
			recurse_depth  = ', Recurse Depth:' + str(mcts.get_recursion_depth())
			k = 81
			top_k = ', top_k:' + str(k)
			value_at_play_root = ', Current Value:' + str(mcts.get_value_at_play_root())

			print(player_to_move + nodes_in_graph + recurse_depth + observable_sims + top_k + value_at_play_root)
			move = mcts.get_weighted_random_move_from_top_k(k)
			moves.append(move)
			if not mcts.commit_to_move(move):
				raise Exception("error executing move")
			print('moves to date:', len(moves), moves)

			visits.append(sum(mcts.get_episode_cell_visits()))
			print('visits/move:', len(visits), visits)

			bored = mcts.get_node_cell_visits()
			for r in range(9):
				for c in range(9):
					if bored[r*9+c]:
						print(bored[r*9+c],end='\t')
					else:
						print('.',end='\t')
				print()

			print(mcts.display())
			print()

			if len(moves) > 81*2 or (len(moves) > 2 and moves[-2:] == [81, 81]):
				game_in_progress = False

		if len(moves) < 81*2:
			print((g+1), 'games played @', (time.time()-start_time)/(g+1), 'seconds per game')

			ownership, final_score = mcts.get_final_score()
			outcome = final_score/abs(final_score)

			game_record = create_game_record(9, 6.5, ''.join([_smallnum_to_char_(x) for x in moves]),
				''.join(['b' if x == 1 else 'w' for x in ownership]), final_score, outcome)

			#example_queue_name_that_goes_to_replay_buffer.put(stringify_game_record(game_record), game_info_to_stringified_training_data(game_record))

			for i in range(len(visits)):
				if i not in diagnostics_total_cell_visits:
					diagnostics_total_cell_visits[i] = 0
				diagnostics_total_cell_visits[i] += visits[i]
			print('\n'.join([str(move_number) + ' ' + str(visit_count) for move_number, visit_count in diagnostics_total_cell_visits.items()]))
