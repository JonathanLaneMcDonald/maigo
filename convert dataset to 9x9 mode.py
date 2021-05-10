
import time

from board import Board

def parse_game_record(game_record):
	return {
		'timestamp': int(game_record.split()[0]),
		'boardsize': int(game_record.split()[1]),
		'komi': float(game_record.split()[2]),
		'moves': game_record.split()[3],
		'ownership': game_record.split()[4],
		'score': float(game_record.split()[5]),
		'outcome': float(game_record.split()[6])
	}

def create_game_record(board_size, komi, moves, ownership, score, outcome):
	return {
		'timestamp': int(time.time()),
		'boardsize': board_size,
		'komi': komi,
		'moves': moves,
		'ownership': ownership,
		'score': score,
		'outcome': outcome
	}

def stringify_game_record(gr):
	return str(gr['timestamp']) + ' ' + str(gr['boardsize']) + ' ' + str(gr['komi']) + ' ' + gr['moves'] + ' ' + gr['ownership'] + ' ' + str(gr['score']) + ' ' + str(gr['outcome'])

def parse_old_game_record(game_record):
	return {
		'boardsize': int(game_record.split()[0]),
		'komi': float(game_record.split()[1]),
		'moves': game_record.split()[2],
	}

def _9x9_to_char_(m):
	return chr(40+m)

def special_purpose_sgf_to_linear_for_9x9(mv):
	if mv == 'pass':
		return 9*9
	if len(mv) != 2:
		raise Exception("something wrong with this game record!")

	x = ord(mv[0]) - 97
	y = ord(mv[1]) - 97

	return y*9 + x

def convert_old_game_record_to_new_game_record(old_game_record):
	# here, we're going to load old game records, play out the whole game, and then get the end state
	old_game_dict = parse_old_game_record(old_game_record)

	new_moves = []
	player = 1
	game = Board(old_game_dict['boardsize'], old_game_dict['komi'])
	for mv in old_game_dict['moves'].split(','):
		new_moves.append(special_purpose_sgf_to_linear_for_9x9(mv))
		if not game.place_stone(new_moves[-1], player):
			raise Exception("your game is busted")
		player *= -1

	if not game.game_has_ended():
		print(game.display())
		print("i keep telling you, your game is busted!")

	ownership, final_score = game.get_simple_terminal_score_and_ownership()
	outcome = final_score / abs(final_score)

	return create_game_record(
		old_game_dict['boardsize'],
		old_game_dict['komi'],
		''.join([_9x9_to_char_(x) for x in new_moves]),
		''.join(['b' if x == 1 else 'w' for x in ownership]),
		final_score,
		outcome
	)

from datasets import game_record_to_training_strings_with_symmetries

if __name__ == "__main__":
	lines = 0
	dataset = open('modern 9x9 training frames','w')
	with open('modernized 9x9 games','w') as fwrite:
		with open('self-play games','r') as fread:
			for line in fread:
				old_record = line[:-1]
				new_record = convert_old_game_record_to_new_game_record(old_record)
				fwrite.write(stringify_game_record(new_record) + '\n')

				for frame in game_record_to_training_strings_with_symmetries(new_record):
					dataset.write(frame + '\n')

				lines += 1
				if lines % 100 == 0:
					print(lines)


