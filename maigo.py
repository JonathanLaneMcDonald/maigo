#coding: utf-8

from tkinter import *
from sklearn import neural_network
from numpy.random import random as npr
from hashlib import sha256
from copy import copy, deepcopy
import numpy as np
import math
import time
import sys
import os

import gzip
import base64

if __name__ == '__main__':
	from keras.models import Sequential, model_from_json, Model
	from keras.layers import Dense, Dropout, Activation, Flatten, Input
	from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
	from keras.optimizers import Adam, Adagrad, Adadelta
	from keras.utils import to_categorical
	from keras.layers.merge import add
	from keras.layers.normalization import BatchNormalization
	
from math import log as ln

debug = open('debugging','w')

DEBUGGING = False

P1_ID = 1
P2_ID = -1
ONE_LIBERTY  = (1<<10)
LIBERTY_MASK = (1<<10)-1
infinity = (1<<64)

EDIT_BLACK = (1<<0)
EDIT_WHITE = (1<<1)

NSEW = [(1,0),(-1,0),(0,1),(0,-1)]

global global_search_calls
global_search_calls = 0

def load_model(fname):
	if fname == None:
		return None

	model = model_from_json(open(fname,'r').read())
	model.load_weights(fname+' weights')
	return model

def load_both_models(filename_base):
	policy = load_model(filename_base+' policy')
	territory = load_model(filename_base+' territory')
	return policy, territory

def save_model(filename_base, model):
	open(filename_base,'w').write(model.to_json())
	model.save_weights(filename_base + ' weights')

def save_both_models(filename_base, policy, territory):
	save_model(filename_base+' policy', policy)
	save_model(filename_base+' territory', territory)

def copy_model(model):
	if model == None:
		return None

	new_model = model_from_json(model.to_json())
	new_model.set_weights(model.get_weights())
	return new_model

def bsearch(array, pattern, lo, hi):
	if hi < lo:
		return -1
	mid = (lo+hi)//2
	if pattern == array[mid]:
		return mid
	if pattern < array[mid]:
		return bsearch(array, pattern, lo, mid-1)
	if pattern > array[mid]:
		return bsearch(array, pattern, mid+1, hi)

class GoGroup():
	def __init__(self, player, stones, vacant_liberties, all_liberties):
		self.player = player
		self.stones = stones
		self.vacant_liberties = vacant_liberties
		self.all_liberties = all_liberties

class Board():
	#this function is going to interleave a bunch of game boards so the display will appear to be two-dimensional
	def assemble_displays(self):
		print ('Last Liberties')
		for p in self.last_liberties:
			print (p)

		displays = []
		displays.append(self.show_board().split('\n'))
		displays.append(self.show_last_liberties().split('\n'))
		displays.append(self.show_legal_moves().split('\n'))

		print ('Board               Last Liberties                 Legal Moves')
		for row in range(self.N):
			line = ''
			for topic in displays:
				line += topic[row] + '     '
			print (line)

	def show_board(self):
		string = ''
		for y in range(self.N):
			for x in range(self.N):
				if self.board[y][x] == 1:		#then we're black
					string += 'X '
				elif self.board[y][x] == -1:	#then we're white
					string += 'O '
				else:
					string += '. '
			string += '\n'
		return string

	def show_legal_moves(self):
		string = ''
		for y in range(self.N):
			for x in range(self.N):
				if self.P1_legal_moves[y][x] and self.P2_legal_moves[y][x]:
					string += 'c '
				elif self.P1_legal_moves[y][x]:
					string += 'a '
				elif self.P2_legal_moves[y][x]:
					string += 'b '
				else:
					string += '. '
			string += '\n'
		return string

	def show_last_liberties(self):
		string = ''
		for y in range(self.N):
			for x in range(self.N):
				if self.P1_last_liberties[y][x] and self.P2_last_liberties[y][x]:
					string += 'c '
				elif self.P1_last_liberties[y][x]:
					string += 'a '
				elif self.P2_last_liberties[y][x]:
					string += 'b '
				else:
					string += '. '
			string += '\n'
		return string

	def in_bounds(self, move):
		row = move[0]
		col = move[1]
		if row < 0 or self.N <= row:		return False
		if col < 0 or self.N <= col:		return False
		return True

	def is_vacant(self, move):
		row = move[0]
		col = move[1]
		if self.groups[row][col]:
			return False
		return True

	def this_move_is_legal(self, move, player):
		# it's always legal to not do anything
		if move == None:
			return True
		
		if not self.in_bounds(move):
			return False
		
		row = move[0]
		col = move[1]
		
		if player == P1_ID:
			if self.P1_legal_moves[row][col]:
				return True
		elif player == P2_ID:
			if self.P2_legal_moves[row][col]:
				return True
		return False

	def __init__(self,N):

		self.edge = N
		self.size = N**2

		self.N = N

		self.board = np.zeros(self.size, dtype=int).reshape(self.edge, self.edge)
		self.groups = np.array([None]*self.size).reshape(self.edge, self.edge)

		# set up liberties and legal moves for both players.  the += 1 on the numpy array sets all the positions to 1 so all moves start out legal.
		self.P1_last_liberties = np.zeros(self.size, dtype=int).reshape(self.edge, self.edge)
		self.P1_legal_moves = np.zeros(self.size, dtype=int).reshape(self.edge, self.edge)
		self.P1_legal_moves += 1

		self.P2_last_liberties = np.zeros(self.size, dtype=int).reshape(self.edge, self.edge)
		self.P2_legal_moves = np.zeros(self.size, dtype=int).reshape(self.edge, self.edge)
		self.P2_legal_moves += 1

		self.last_liberties = []
		self.ko = None
		
	def discover_liberties(self, group):
		# do the full N,S,E,W thing on each stone to see which liberties are not occupied by other group members
		# checking this list instead of doing a full check every time could speed up refresh times by up to half
		liberties = []
		for s in group.stones:
			
			# from each stone position, we'll look up, down, left, and right for liberties
			row = s[0]
			col = s[1]
			
			# look up
			if 0 <= row-1:
				if self.groups[row-1][col] != group:
					liberties.append((row-1, col))
			
			# look down
			if row+1 < self.edge:
				if self.groups[row+1][col] != group:
					liberties.append((row+1, col))
			
			# look left
			if 0 <= col-1:
				if self.groups[row][col-1] != group:
					liberties.append((row, col-1))
			
			# look right
			if col+1 < self.edge:
				if self.groups[row][col+1] != group:
					liberties.append((row, col+1))
			
		group.all_liberties = list(set(liberties))

	def refresh_liberties(self, group):
		liberties = []
		for l in group.all_liberties:
			if self.groups[l[0]][l[1]] == None:
				liberties.append(l)

		group.vacant_liberties = list(set(liberties))

		if len(group.vacant_liberties) == 1:
			self.last_liberties.append((group.player, group.stones[0], group.vacant_liberties[0]))

	def update_liberties(self, group, new_move):
		group.vacant_liberties.sort()
		liberty_index = bsearch(group.vacant_liberties, new_move, 0, len(group.vacant_liberties)-1)
		if liberty_index != -1:
			group.vacant_liberties = group.vacant_liberties[:liberty_index]+group.vacant_liberties[liberty_index+1:]

		if len(group.vacant_liberties) == 1:
			self.last_liberties.append((group.player, group.stones[0], group.vacant_liberties[0]))

	def merge_groups(self, group, existing_groups):
		"""
		Try to merge groups in an efficient way.
		1) sort groups by the number of stones
		2) new_group = sorted_groups.pop()
		3) for g in groups:
			add the stones and mark the group in self.groups
		4) for g in groups:
			updating liberties now - so check each liberty to make sure there's not a group member there
		5) return the pointer to new_group - the vacant liberties will be refreshed later
		"""
		existing_groups.append(group)
		
		sortable_groups = []
		for eg in existing_groups:
			sortable_groups.append((len(eg.stones),eg))
		sortable_groups.sort(key=lambda x: int(x[0]))
		
		new_group = sortable_groups.pop()[1]
		
		for sg in sortable_groups:
			new_group.stones += sg[1].stones
			new_group.all_liberties += sg[1].all_liberties
		new_group.stones = list(set(new_group.stones))
		new_group.all_liberties = list(set(new_group.all_liberties))

		for stone in new_group.stones:
			self.groups[stone[0]][stone[1]] = new_group
		
		liberties = []
		for liberty in new_group.all_liberties:
			if self.groups[liberty[0]][liberty[1]] != new_group:
				liberties.append(liberty)
		
		new_group.all_liberties = liberties
		
		return new_group
		
	def remove_group(self, group):
		# actually remove the stones from the zobrist_grid
		neighboring_groups = []
		for s in group.stones:
			self.groups[s[0]][s[1]] = None
			self.P1_legal_moves[s[0]][s[1]] = 1
			self.P2_legal_moves[s[0]][s[1]] = 1
			self.board[s[0]][s[1]] = 0

			row = s[0]
			col = s[1]
			
			# look up
			if 0 <= row-1:
				if self.groups[row-1][col]:
					neighboring_groups.append(self.groups[row-1][col])
			
			# look down
			if row+1 < self.edge:
				if self.groups[row+1][col]:
					neighboring_groups.append(self.groups[row+1][col])
			
			# look left
			if 0 <= col-1:
				if self.groups[row][col-1]:
					neighboring_groups.append(self.groups[row][col-1])
			
			# look right
			if col+1 < self.edge:
				if self.groups[row][col+1]:
					neighboring_groups.append(self.groups[row][col+1])

		neighboring_groups = list(set(neighboring_groups))

		return neighboring_groups

	def move_is_legal(self, move):
		"""
		perform the legality tests to see if a move is legal.

		qualifications for a legal move
			1) the move can be a pass
			2) the move must be on the board
			3) the move must be into a vacant spot
			4) the move must not violate the ko rule
			5) the move must satisfy one of the following
				5.1) the move position has at least one vacancy near it
				5.2) the move will capture an enemy group
				5.3) the move will join a friendly group without consuming its last liberty
		"""

		# in practice, we'll try to go in order of how expensive it is to determine legality, 
		# because the faster we can exit this function, the better

		if move == None:
			return 1, 1

		row = move[0]
		col = move[1]
		
		# the move has to be on the board or it's not legal
		if row < 0 or self.edge <= row or col < 0 or self.edge <= col:
			return 0, 0
		
		# if the position is occupied, it's not a legal move
		if self.groups[row][col] != None:
			return 0, 0
		
		# check to see if there's a vacancy near where my stone is going, because that makes it legal

		# look up
		if 0 <= row-1:
			if self.groups[row-1][col] == None:
				return 1, 1
		
		# look down
		if row+1 < self.edge:
			if self.groups[row+1][col] == None:
				return 1, 1
		
		# look left
		if 0 <= col-1:
			if self.groups[row][col-1] == None:
				return 1, 1
		
		# look right
		if col+1 < self.edge:
			if self.groups[row][col+1] == None:
				return 1, 1

		# at this point, the only options are the ko and to check if we're joining one of our own groups
		# these are both relatively rare conditions, but they have to be evaluated separately for each player
		# so we're stuck with the full workload if we get this far
		
		P1_legality = 0
		P2_legality = 0

		# start the ko check by taking away legality and then giving it back
		if self.ko:
			if self.ko != (move, P1_ID):
				P1_legality = 1
			if self.ko != (move, P2_ID):
				P2_legality = 1
		else:
			P1_legality = 1
			P2_legality = 1

		if P1_legality:		P1_legality = self.satisfies_rule_5(move, P1_ID)
		if P2_legality:		P2_legality = self.satisfies_rule_5(move, P2_ID)

		return P1_legality, P2_legality

	def satisfies_rule_5(self, move, player):
		for direction in NSEW:
			new_move = (move[0]+direction[0],move[1]+direction[1])
			if self.in_bounds(new_move):
				group = self.groups[new_move[0]][new_move[1]]
				if group:
					if player == group.player:
						if 1 < len(group.vacant_liberties):
							return 1
					else:
						if len(group.vacant_liberties) <= 1:
							return 1
		return 0

	def still_a_qualified_last_liberty(self, input_tuple):
		player       = input_tuple[0]
		group_head   = input_tuple[1]
		last_liberty = input_tuple[2]
		
		group = self.groups[group_head[0]][group_head[1]]
		if group:
			if player == group.player and [last_liberty] == group.vacant_liberties:
				return True
		return False

	def do_move(self, move, player):
		return self.place_stone(move, player)

	def group_report(self, group):
		print (group)
		print (group.player)
		print (group.stones)
		print (group.vacant_liberties)
		print (group.all_liberties)
	
	def place_stone(self, move, player):
		"""
		****************************************************************
		try adding another element to the tuple to say which liberties to check when refreshing liberties.  it doesn't make sense to check all four directions for all stones.
		you can cut the work by about half if you only "refresh" liberties that do not land on another member of the group.
		"""

		if move:
			if player == P1_ID:
				if self.P1_legal_moves[move[0]][move[1]] == 0:
					return -1
			elif player == P2_ID:
				if self.P2_legal_moves[move[0]][move[1]] == 0:
					return -1
		else:
			return 0

		if self.ko:
			m = self.ko[0]
			self.ko = None
			# if we forget to clear this ko before asking about these cells, it's just going to fail again
			self.P1_legal_moves[m[0]][m[1]], self.P2_legal_moves[m[0]][m[1]] = self.move_is_legal(m)

		#print ('I am',player,'at',move)
		# this nesting doubles as 1) the final qualification for a legal move, and 2) discovery of friends and foes for later
		friendly_groups = []
		adversarial_groups = []
		for direction in NSEW:
			new_move = (move[0]+direction[0],move[1]+direction[1])
			if self.in_bounds(new_move):

				group = self.groups[new_move[0]][new_move[1]]
				if group != None:
					if group.player == player:
						friendly_groups.append(group)
						#print ('I see a friend',group)
					else:
						adversarial_groups.append(group)
						#print ('I see an enemy',group)

		friendly_groups = list(set(friendly_groups))
		adversarial_groups = list(set(adversarial_groups))

		new_group = GoGroup(player, [move], [], [])
		self.discover_liberties(new_group)

		if len(friendly_groups):
			new_group = self.merge_groups(new_group, friendly_groups)
		else:
			self.groups[move[0]][move[1]] = new_group
		
		self.P1_legal_moves[move[0]][move[1]] = 0
		self.P2_legal_moves[move[0]][move[1]] = 0
		self.board[move[0]][move[1]] = player

		# reduce liberties of hostile groups and if they run out of liberties, remove them and update their neighbors
		effected_groups = []
		for ag in adversarial_groups:
			self.update_liberties(ag, move)
			
			if len(ag.vacant_liberties) == 0:
				effected_groups += self.remove_group(ag)
				if len(ag.stones) == 1 and len(new_group.stones) == 1:
					self.ko = (ag.stones[0], -1 * player)
		
		# next, update liberties for groups that are affiliated with the current player
		for eg in effected_groups:
			self.refresh_liberties(eg)

		self.refresh_liberties(new_group)

		# now the liberties for all groups are up to date and we can check the liberties of groups effected by the current move to update our legal moves list
		liberties = copy(new_group.vacant_liberties)
		for ag in adversarial_groups:
			if len(ag.vacant_liberties) == 1:
				liberties += ag.vacant_liberties
		for eg in effected_groups:
			liberties += eg.vacant_liberties
		liberties = list(set(liberties))

		for lb in liberties:
			self.P1_legal_moves[lb[0]][lb[1]], self.P2_legal_moves[lb[0]][lb[1]] = self.move_is_legal(lb)

		if len(self.last_liberties):
			# don't repeat work if it can be avoided
			self.last_liberties = list(set(self.last_liberties))

			# for all eyes in the list, see if it's still an eye
			updated_liberties = []
			for i in self.last_liberties:
				value = 0
				if self.still_a_qualified_last_liberty(i):
					updated_liberties.append(i)
					value = 1

				p = i[0]
				m = i[2]
				if p == P1_ID:
					self.P1_last_liberties[m[0]][m[1]] = value
				else:
					self.P2_last_liberties[m[0]][m[1]] = value

			self.last_liberties = updated_liberties

		if DEBUGGING:
			print ('The newly formed group has the following attributes:')
			self.group_report(new_group)
			#print ('Groups:\n',self.groups)

		return 0

class BoardUtils(Board):
	def __init__(self):
		pass
	
	def tromp_taylor_scoring(self, board):
		black = 0
		white = 0
		
		territory = np.zeros(board.size, dtype=np.intc).reshape(board.edge, board.edge)
		for r in range(board.edge):
			for c in range(board.edge):
				if board.board[r][c] == 1:
					black += 1
					territory[r][c] = 1
				elif board.board[r][c] == -1:
					white += 1
					territory[r][c] = -1
				else:
					# if this cell is unoccupied, we defer to the Tromp-Taylor scoring rules
					# specifically, if an empty cell is connected to only one player via a horizontal or vertical pathway, then it belongs to that player
					reaches_black = False
					reaches_white = False

					for delta_r, delta_c in NSEW:
						local_r = r
						local_c = c

						keep_walking = True
						while keep_walking:
							local_r += delta_r
							local_c += delta_c
							if local_r < 0 or board.edge <= local_r or local_c < 0 or board.edge <= local_c:
								keep_walking = False
							else:
								if board.board[local_r][local_c] == 1:
									reaches_black = True
									keep_walking = False
								elif board.board[local_r][local_c] == -1:
									reaches_white = True
									keep_walking = False
					
					if reaches_black and not reaches_white:
						territory[r][c] = 1
						black += 1
					elif reaches_white and not reaches_black:
						territory[r][c] = -1
						white += 1
		
		return territory, black - white

	def game_winner(self, board):
		territory, score = self.tromp_taylor_scoring(board)
		if score != 0:
			return score // abs(score)
		return 0
	
	def moves_to_sgf(self, moves):
		sgf = []
		for m in moves:
			if m == None:
				sgf.append('pass')
			else:
				sgf.append(chr(m[0]+97)+chr(m[1]+97))
		return ','.join(sgf)			

	def sgf_to_moves(self, sgf):
		moves = []
		for m in sgf.split(','):
			if m == 'pass':
				moves.append(None)
			elif len(m) == 2:
				moves.append((ord(m[0])-97,ord(m[1])-97))
			else:
				return moves
		return moves
	
	def moves_to_features(self, source_board, moves, player):
		
		input_features = []
		for m in moves:
			board = deepcopy(source_board)
			board.do_move(m, player)
			
			features = np.zeros(board.edge*board.edge*2).reshape(board.edge, board.edge, 2)
			
			for r in range(board.edge):
				for c in range(board.edge):
					if player * board.board[r][c] == P1_ID:
						features[r][c][0] = 1
					elif player * board.board[r][c] == P2_ID:
						features[r][c][1] = 1

			if m != None:
				input_features.append(features)

		input_features = np.array(input_features).reshape(len(input_features), board.edge, board.edge, 2)
		
		return input_features
		
	def position_to_state(self, board, player):
		edge = board.edge

		features = np.zeros(edge*edge*2, dtype=np.intc).reshape(edge, edge, 2)
		
		for r in range(edge):
			for c in range(edge):
				if player * board.board[r][c] == P1_ID:
					features[r][c][0] = 1
				elif player * board.board[r][c] == P2_ID:
					features[r][c][1] = 1

		return features

	def position_to_multistate(self, board, player, max_liberty_channels):
		edge = board.edge

		features = np.zeros(edge*edge*(2+2*max_liberty_channels), dtype=np.intc).reshape(edge, edge, (2+2*max_liberty_channels))
		visited = np.zeros(edge*edge, dtype=np.intc).reshape(edge, edge)
		
		for r in range(edge):
			for c in range(edge):
				if player * board.board[r][c] == P1_ID:
					features[r][c][0] = 1
				elif player * board.board[r][c] == P2_ID:
					features[r][c][1] = 1

				# put a few channels into liberties now - i think this is going to be 2+2*i for "me" and 2+2*i+1 for "him" where the first two channels are for stones
				# so there will be alternating channels for liberties from 1 to "5+" for 5 liberty related channels per player for a total of 12 channels including stones and liberties
				if visited[r][c] == 0:
					if board.groups[r][c] != None:
						local_player = board.groups[r][c].player
						stones = board.groups[r][c].stones
						spaces = board.groups[r][c].vacant_liberties
						#liberty_channel = min(len(spaces), max_liberty_channels)
						liberty_channel = len(spaces)

						if liberty_channel <= max_liberty_channels:
							feature_channel = 2*liberty_channel
							if local_player == P2_ID:
								feature_channel += 1
							
							for R,C in spaces:
								features[R][C][feature_channel] = 1
						
						for R,C in stones:
							visited[R][C] = 1

		return features

	def sgf_to_dataset(self, sgf, edge, max_liberties):
		"""
		Replays an sgf record, recording state-action pairs, tabulates the outcome of the game, and then returns an array of tuples (s,a,z) from the perspective of the player placing the stone
		(s,a) is used to train the policy network
		(s,z) is used to train the value network
		"""
		input_features = []
		policy_labels = []
		value_labels = []

		moves = self.sgf_to_moves(sgf)
		board = Board(edge)
		player = 1
		
		for m in moves:
			if m != None:
				if max_liberties:
					input_features.append(self.position_to_multistate(board, player, max_liberties))
				else:
					input_features.append(self.position_to_state(board, player))

				action = m[0]*edge + m[1]
				policy_labels.append(to_categorical(action, edge*edge))

				result = player
				value_labels.append(result)

			board.do_move(m,player)
			player *= -1
		
		territory, score = self.tromp_taylor_scoring(board)
		for i in range(len(value_labels)):
			value_labels[i] *= score

		return input_features, policy_labels, value_labels
		
	def sgf_array_to_dataset(self, sgf_array, edge, max_liberties):
		"""
		Automate sgf_to_dataset() over an array of sgf records - those records can be imported from a file and randomized or something, and then I can train on like 10,000 games or something.
		"""
		input_features = []
		policy_labels = []
		value_labels = []
		
		for sgf in sgf_array:
			s, a, z = self.sgf_to_dataset(sgf, edge, max_liberties)
			
			input_features += s
			policy_labels += a
			value_labels += z

		channels = 2 + 2*max_liberties
		
		input_features = np.array(input_features, dtype=np.intc).reshape(len(input_features), edge, edge, channels)
		policy_labels = np.array(policy_labels, dtype=np.intc).reshape(len(policy_labels), edge*edge)
		value_labels = np.array(value_labels).reshape(len(value_labels), 1)

		return input_features, policy_labels, value_labels

	def compress_features(self, features):

		return base64.b64encode(gzip.zlib.compress(features))

	def expand_features(self, compressed_features):
		
		return gzip.zlib.decompress(base64.b64decode(compressed_features))

	def dataset_to_string_array(self, state, action, outcome, edge, total_channels):
		
		assert len(state) == len(action) and len(action) == len(outcome)
		
		state = state.reshape(len(state),edge*edge,total_channels)

		string_array = []
		for i in range(len(state)):
			# first, get the channels out of the feature array
			features = ''
			for j in state[i]:
				for k in j:
					features += str(k)

			#print ('this is what the features look like:',features)
			string = self.compress_features(features)

			#assert features == self.expand_features(self.compress_features(features))

			string += ' '
			for j in range(len(action[i])):
				if int(action[i][j]):
					string += str(int(j))
			
			string += ' '+str(int(outcome[i]))

			string_array.append(string)
		
		return string_array

	def string_array_to_dataset(self, string_array, edge, total_channels):
		
		state_array = np.zeros(len(string_array)*edge*edge*total_channels,dtype=bool).reshape(len(string_array), edge, edge, total_channels)
		action_array = np.zeros(len(string_array)*edge*edge,dtype=bool).reshape(len(string_array), edge*edge)
		outcome_array = np.zeros(len(string_array),dtype=int).reshape(len(string_array))

		counter = 0
		start_time = time.time()
		for string in string_array:
			if len(string.split()) == 3:
				state_string = self.expand_features(string.split()[0])

				#print ('state string',state_string)

				#state = np.zeros((edge*edge)*total_channels).reshape(edge*edge, total_channels)
				#state = np.fromstring(' '.join(state_string), sep=' ')
				state = np.array(list(state_string)).reshape(edge, edge, total_channels)
				state -= 48

				#print ('state',state)

				#state_array[counter] = state.reshape(edge, edge, total_channels)
				state_array[counter] = state
				
				action_string = string.split()[1]
				action_array[counter][int(action_string)] = 1

				outcome_string = string.split()[2]
				outcome_array[counter] = int(outcome_string)
				
				counter += 1

			if counter and counter % 100000 == 0:
				print (counter, counter/(time.time()-start_time))

		return state_array, action_array, outcome_array

	def sgf_to_multichannel_dataset(self, sgf, edge, channels):
		"""
		Replays an sgf record, recording state-action pairs, tabulates the outcome of the game, and then returns an array of tuples (s,a,z) from the perspective of the player placing the stone
		(s,a) is used to train the policy network
		(s,z) is used to train the value network

		The multichannel part refers to the number of moves we want to show from each player's history in the spirit of AG0
		to be clear, each player gets channels/2 channels.  channels is the total number of channels
		"""
		input_features = []
		policy_labels = []
		value_labels = []

		moves = self.sgf_to_moves(sgf)
		board = Board(edge)
		player = 1
		
		board_history = []

		for i in range(len(moves)):
			m = moves[i]

			board_history.append(copy(board.board))

			# this channel format is to mimic the deepmind paper
			features = np.zeros(edge*edge*(channels+1), dtype=np.intc).reshape(edge, edge, channels+1)
			
			for chan in range((channels>>1)):
				# if we don't have enough history, then leave it and it'll show up as zeros
				if chan < len(board_history):
					for r in range(edge):
						for c in range(edge):
							if 		board_history[-(1+chan)][r][c] == P1_ID:
								features[r][c][2*chan + 0] = 1
							elif 	board_history[-(1+chan)][r][c] == P2_ID:
								features[r][c][2*chan + 1] = 1

			if player == P1_ID:
				pass # because we were just going to fill it with zeros, but it's already zeros
			elif player == P2_ID:
				for r in range(edge):
					for c in range(edge):
						features[r][c][channels] = 1 # note that 'channels' == the 'channels+1'th channel

			if m != None:
				input_features.append(features)

				action = m[0]*edge + m[1]
				policy_labels.append(to_categorical(action, edge*edge))

				result = player
				value_labels.append(result)

			board.do_move(m,player)
			player *= -1
		
		if self.game_winner(board) == -1:
			for i in range(len(value_labels)):
				value_labels[i] *= -1

		return input_features, policy_labels, value_labels
		
	def sgf_array_to_multichannel_dataset(self, sgf_array, edge, channels):
		"""
		Automate sgf_to_dataset() over an array of sgf records - those records can be imported from a file and randomized or something, and then I can train on like 10,000 games or something.
		"""
		input_features = []
		policy_labels = []
		value_labels = []
		
		for sgf in sgf_array:
			s, a, z = self.sgf_to_multichannel_dataset(sgf, edge, channels)
			
			input_features += s
			policy_labels += a
			value_labels += z
		
		input_features = np.array(input_features).reshape(len(input_features), edge, edge, channels+1)
		policy_labels = np.array(policy_labels).reshape(len(policy_labels), edge*edge)
		value_labels = np.array(value_labels).reshape(len(value_labels), 1)

		return input_features, policy_labels, value_labels

	def multichannel_dataset_to_string_array(self, state, action, outcome, edge, channels):
		
		assert len(state) == len(action) and len(action) == len(outcome)
		
		state = state.reshape(len(state),edge*edge*channels)

		string_array = []
		for i in range(len(state)):

			features = self.compress_features(state[i].tobytes()).decode('utf-8')
			#s = np.frombuffer(self.expand_features(features.encode('utf-8')), dtype=np.int32)
			
			#if s.all() == state[i].all():
			#	print ('true')

			features += ' '
			for j in range(len(action[i])):
				if int(action[i][j]):
					features += str(int(j))
			
			features += ' '+str(int(outcome[i]))
		
			string_array.append(features)
		
		return string_array

	def string_array_to_multichannel_dataset(self, string_array, edge, channels):
		
		state_array = np.zeros(len(string_array)*edge*edge*channels, dtype=bool).reshape(len(string_array), edge, edge, channels)
		action_array = np.zeros(len(string_array)*edge*edge).reshape(len(string_array), edge*edge)
		outcome_array = np.zeros(len(string_array)).reshape(len(string_array))

		counter = 0
		start_time = time.time()
		for string in string_array:
			if len(string.split()) == 3:
				state_string = string.split()[0]
				
				state = np.frombuffer(self.expand_features(state_string.encode('utf-8')), dtype=np.int32)
				
				state_array[counter]=state.reshape(edge, edge, channels)
				
				action_string = string.split()[1]
				action = np.zeros(edge*edge)
				action[int(action_string)] = 1
				action_array[counter]=action
				
				outcome_string = string.split()[2]
				outcome_array[counter]=int(outcome_string)

			counter += 1
			if counter and counter % 100000 == 0:
				print (counter, counter/(time.time()-start_time))

		return state_array, action_array, outcome_array

	def replay(self, sgf, edge, show_replay):
		moves = self.sgf_to_moves(sgf)
		
		board = Board(edge)
		player = 1
		for m in moves:
			error = board.do_move(m, player)
			if error:
				return None, None
			player *= -1

			if show_replay:
				board.assemble_displays()

		territory, score = self.tromp_taylor_scoring(board)

		if show_replay:
			board.assemble_displays()
			print ('Score:',score,'\nTerritory:\n',territory)

		return territory, score
		





'''
games_with_errors = 0
counter = 0
with open('go_kgs_6d+_symmetries','r') as openfileobject:
	for line in openfileobject:
		a,b,c = BoardUtils().replay(line[:-1], 19, False)
		if c:
			pass
		else:
			print ('error replaying game',line[:-1])
			games_with_errors += 1

		counter += 1
		if counter % 100 == 0:
			print (counter, games_with_errors)
'''
#*************************************
#	miscellaneous functions
#*************************************

h = '0123456789abcdef'
def tohex(r,g,b):
	color = '#'
	color += h[r//16]+h[r%16]
	color += h[g//16]+h[g%16]
	color += h[b//16]+h[b%16]
	return color

# ************************************
#           strategies                
# ************************************

# what would sensible random guy do
def wwsrgd(board,player):

	local_board = np.copy(board.board)
	
	size = board.board.size
	columns = board.board.shape[1] #we only need to know the number of columns to figure out what row we're on

	preferences = []
	for i in range(size):
		row = i//columns
		col = i%columns
		score = 1
		preferences.append((score, (row, col)))
	preferences.sort()
	preferences.reverse()
	
	my_legal_moves = board.P2_legal_moves
	his_legal_moves = board.P1_legal_moves
	his_liberties = board.P1_liberty_map
	he_has_a_ko = board.P1_has_a_ko
	if player == P1_ID:
		my_legal_moves = board.P1_legal_moves
		his_legal_moves = board.P2_legal_moves
		his_liberties = board.P2_liberty_map
		he_has_a_ko = board.P2_has_a_ko

	considerable_sum = 0
	under_consideration = []
	for p in preferences:
		w = p[0]
		r = p[1][0]
		c = p[1][1]
		if my_legal_moves[r][c] and (his_legal_moves[r][c] or his_liberties[r][c] or he_has_a_ko[r][c]):
			considerable_sum += w
			under_consideration.append(p)

	if len(under_consideration) == 0:
		return None
	
	selection = int(npr()*len(under_consideration))
	return under_consideration[selection][1]

def sgf_to_coords(s):
	if s == 'pass':
		return None
	if len(s) != 2:
		return None

	x = ord(s[0])-97
	y = ord(s[1])-97

	return (y,x)

N = 9
THINKTIME = 10
player_1 = 'human'
player_2 = 'human'
player_1_version = 1
player_2_version = 1
handicap = 0
use_gui = True

#not safe - if you use the wrong parameters, you're on your own ;P
for i in range(1,len(sys.argv)):
	if sys.argv[i] == '-s':
		N = int(sys.argv[i+1])
		print ('Board size set to',N)
	if sys.argv[i] == '-t':
		THINKTIME = int(sys.argv[i+1])
		print ('MCTS Think Time set to',THINKTIME)
	if sys.argv[i] == '-hc':
		handicap = int(sys.argv[i+1])-1
		handicap = max(handicap,0)
		print ('Handicap set to',handicap)
	if sys.argv[i] == '-p1':
		player = sys.argv[i+1]
		if player == 'human' or player == 'random' or player == 'srandom' or player == 'themachine':
			player_1 = player
			print ('Player 1 set to',player_1)
		else:
			print ('Unrecognized player')
	if sys.argv[i] == '-p2':
		player = sys.argv[i+1]
		if player == 'human' or player == 'random' or player == 'srandom' or player == 'themachine':
			player_2 = player
			print ('Player 2 set to',player_2)
		else:
			print ('Unrecognized player')
	if sys.argv[i] == '-p1v':
		player_1_version = int(sys.argv[i+1])
		print ('Player 1 Version:',player_1_version)
	if sys.argv[i] == '-p2v':
		player_2_version = int(sys.argv[i+1])
		print ('Player 2 Version:',player_2_version)
	if sys.argv[i] == '-nogui':
		use_gui = False
		print ('Not using GUI')

def randomize(array):
	for i in range(len(array)):
		if i % 5000000 == 0:
			print ('Permuted',i,'items.')
		
		j = int(npr()*len(array))
		while j == i:
			j = int(npr()*len(array))
		
		temp = array[i]
		array[i] = array[j]
		array[j] = temp

	print ('Permutation Complete')

	return array

# this function will actually return a training-ready dataset
def string_to_state_action(string, edge_length):
	board_size = edge_length**2
	
	boardstr = string.split('\t')[0]
	player   = int(string.split('\t')[1])
	movestr  = string.split('\t')[2]
	
	board = np.zeros(board_size*2).reshape(board_size,2)
	for c in range(len(boardstr)):
		stone = int(boardstr[c])
		if player == 1:
			# the current player is always channel 0 and the opponent is always channel 1
			if stone == 1:				board[c][0] = 1
			elif stone == 2:			board[c][1] = 1
		elif player == 2:
			# so if we're doing this from player 2's perspective, reverse the channels
			if stone == 1:				board[c][1] = 1
			elif stone == 2:			board[c][0] = 1
	board = board.reshape(edge_length, edge_length, 2)

	moves = np.zeros(board_size)
	moves[int(movestr)] = 1

	return board, moves

# this function will actually return a training-ready dataset
def complex_string_to_state_action(string, edge_length, use_win_rates):
	board_size = edge_length**2
	
	boardstr = string.split('\t')[0]
	player   = int(string.split('\t')[1])
	movestr  = string.split('\t')[2]
	selection_rate = string.split('\t')[3].split(',')
	win_rate = string.split('\t')[4].split(',')
	
	if len(boardstr) != board_size or len(win_rate) != board_size or len(selection_rate) != board_size:
		return [], []

	board = np.zeros(board_size*2).reshape(board_size,2)
	for c in range(len(boardstr)):
		stone = int(boardstr[c])
		if player == 1:
			# the current player is always channel 0 and the opponent is always channel 1
			if stone == 1:				board[c][0] = 1
			elif stone == 2:			board[c][1] = 1
		elif player == 2:
			# so if we're doing this from player 2's perspective, reverse the channels
			if stone == 1:				board[c][1] = 1
			elif stone == 2:			board[c][0] = 1
	board = board.reshape(edge_length, edge_length, 2)

	moves = np.zeros(board_size,dtype=np.float32).reshape(board_size)
	for i in range(board_size):
		if use_win_rates:
			moves[i] = float(win_rate[i])
		else:
			moves[i] = float(selection_rate[i])

	return board, moves

def state_action_to_string(board, player, move):
	# board to 1-dimmensional array - empty = 0, P1 = 1, P2 = 2 - these will also correspond to channels in the convnet
	# move is (y,x) or (r,c)
	columns = board.shape[1]
	local_board = board.reshape(board.size)
	string = ''
	for i in local_board:
		if i == 0:
			string += '0'
		elif i == 1:
			string += '1'
		else:
			string += '2'	
	
	player_alias = 1
	if player == -1:
		player_alias = 2

	row = move[0]
	col = move[1]
	pos = row*columns + col
	
	string += '\t'+str(player_alias)+'\t'+str(pos)
	return string

# **********************************************************************
# *****************************train models*****************************
# **********************************************************************

def build_model(edge_length):
	board_size = edge_length**2
	
	model = Sequential()

	model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(edge_length, edge_length, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(board_size, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def build_normal_model(edge_length, loss_function):
	board_size = edge_length**2
	
	model = Sequential()

	model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(edge_length, edge_length, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(board_size, activation='softmax'))

	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

	return model

def build_little_model(edge_length, loss_function):
	board_size = edge_length**2
	
	model = Sequential()

	model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(edge_length, edge_length, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(board_size, activation='softmax'))

	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

	return model

def train_a_new_model(training_data_file, trained_model_file, samples_to_load, edge_length, training_epochs, use_little_model, use_complex_model, use_win_rates, loss_function):
	board_size = edge_length**2
	
	training_features = []
	training_labels = []

	training_data = randomize(open(training_data_file,'r').read().split('\n')[:-2])
	for i in training_data[:samples_to_load]:
		features, labels = string_to_state_action(i, edge_length)
		if use_complex_model:
			features, labels = complex_string_to_state_action(i, edge_length, use_win_rates)

		if features != [] and labels != []:
			training_features.append(features)
			training_labels.append(labels)
		else:
			print ('Warning: Incomplete or corrupted data found and rejected from training data')
		
		if len(training_features) % 10000 == 0:
			print (len(training_features),'training features loaded')

	training_features = np.array(training_features).reshape(len(training_features),edge_length,edge_length,2)
	training_labels = np.array(training_labels).reshape(len(training_labels),board_size)

	print (len(training_features),'training features and',len(training_labels),'training labels loaded')

	model = build_normal_model(edge_length, loss_function)
	if use_little_model:
		model = build_little_model(edge_length, loss_function)

	#print (training_features)
	#print (training_labels)
	model.fit(training_features, training_labels, batch_size=128, epochs=training_epochs, verbose=1, validation_split=0.1)

	filename_base = trained_model_file
	open(filename_base,'w').write(model.to_json())
	model.save_weights(filename_base + ' weights')

def simulation_speed_test(sgf_file_location, edge_length):
	games = randomize(open(sgf_file_location,'r').read().split('\n'))
	print (len(games),'games loaded')

	debug = open('games with errors','w')

	errors = 0
	recorded = 0
	games_with_errors = []
	moves_seen = 0

	start_time = time.time()
	
	for i in games:
		recorded += 1
		if recorded % 100 == 0:
			elapsed_time = time.time() - start_time
			time_per_game = int(1000000*elapsed_time / float(recorded))
			time_per_move = int(1000000000*elapsed_time / float(moves_seen))
			print ('t/(m,g): ('+str(time_per_move)+'ns,'+str(time_per_game)+'us)\tExporting Game', recorded,'\tErrors:',errors,'\tGames With Errors:',len(games_with_errors),'\tState-Action Pairs:',moves_seen)

		game = i.split(',')

		player = 1
		board = Board(edge_length)
		this_game_has_errors = False
		for j in game:
			move = sgf_to_coords(j)
			moves_seen += 1
			error = board.place_stone(move,player)
			if error:
				errors += 1
				this_game_has_errors = True
			player *= -1
		
		if this_game_has_errors:
			games_with_errors.append(i)
			debug.write(i+'\n')

	debug.close()

def review_games_with_errors():
	games_with_errors = open('games with errors','r').read().split('\n')[:-1]
				
	for game in games_with_errors:
		player = 1
		board = Board(N)
		for move in game.split(','):
			print
			error = board.place_stone(sgf_to_coords(move),player)
			print ('Error Status:',error,sgf_to_coords(move),player)
			board.assemble_displays()
			player *= -1
			#time.sleep(0.1)

def make_resnet():
	return ResnetBuilder().build_resnet_18((17,19,19),361)

def old_build_network(channels, convolutional_layers, fully_connected_layers, edge_length, loss_function, is_value_network):
	board_size = edge_length**2
	
	model = Sequential()

	model.add(Conv2D(convolutional_layers[0][0], convolutional_layers[0][1], activation='relu', padding='same', input_shape=(edge_length, edge_length, channels)))
	for cv in convolutional_layers[1:]:
		model.add(Conv2D(cv[0], cv[1], activation='relu', padding='same'))

	model.add(Flatten())

	model.add(Dense(fully_connected_layers[0], activation='relu'))
	model.add(Dropout(0.5))
	for fc in fully_connected_layers[1:]:
		model.add(Dense(fc, activation='relu'))
		model.add(Dropout(0.5))

	if is_value_network:
		model.add(Dense(1, activation='tanh'))
	else:
		model.add(Dense(board_size, activation='softmax'))

	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

	return model

def build_network(input_shape, convolutional_layers, fully_connected_layers, edge_length, loss_function, is_value_network):
	board_size = edge_length**2
	
	model = Sequential()

	model.add(Conv2D(convolutional_layers[0][0], convolutional_layers[0][1], activation='relu', padding='same', input_shape=input_shape))
	for cv in convolutional_layers[1:]:
		model.add(Conv2D(cv[0], cv[1], activation='relu', padding='same'))

	model.add(Flatten())

	for fc in fully_connected_layers:
		model.add(Dense(fc, activation='relu'))
		model.add(Dropout(0.5))

	if is_value_network:
		model.add(Dense(1, activation='tanh'))
	else:
		model.add(Dense(board_size, activation='softmax'))

	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

	return model

def build_policy_network(edge_length, channels):
	return build_network(channels, [[32, (3,3)]]*4, [32], edge_length, 'categorical_crossentropy', False)

def build_value_network(edge_length):
	return build_network(2, [[32, (3,3)]]*4, [128], edge_length, 'mean_squared_error', True)

def write_dataset_to_file(filename, state, action, outcome, edge):
	
	string_array = dataset_to_string_array(state, action, outcome, edge)

	save = open(filename,'w')
	for i in string_array:
		save.write(i + '\n')
	save.close()

def read_dataset_from_file(filename, edge):
	
	string_array = open(filename,'r').read().split('\n')[:-1]
	
	state, action, outcome = string_array_to_dataset(string_array, edge)
	
	return state, action, outcome

def evaluate_model_predictions(training_strings, model, batch_size, edge, total_channels):

	state, action, outcome = BoardUtils().string_array_to_dataset(training_strings, edge, total_channels)

	predictions = model.predict(state)
	
	top1, top5, top10 = 0, 0, 0
	for sample in range(len(state)):
		ground_truth = 0
		by_preference = []
		for i in range(len(predictions[sample])):
			by_preference.append((predictions[sample][i], i))

			# i needed to arrange the preferences anyway, so now i can hitch a ride to find what the ground truth is
			if action[sample][i] == 1:
				ground_truth = i
				
		by_preference.sort()
		by_preference.reverse()

		# now we have our sorted preferences and we have (hopefully) located the ground truth answer, let's see if they match
		for i in range(10):
			if i == 0 and by_preference[i][1] == ground_truth:		top1 += 1
			if i <  5 and by_preference[i][1] == ground_truth:		top5 += 1
			if i < 10 and by_preference[i][1] == ground_truth:		top10+= 1

	return top1/float(batch_size), top5/float(batch_size), top10/float(batch_size)

def minibatch_training(training_strings, model, epochs, training_batch_size, evaluation_batch_size, is_value_network, edge, total_channels, name):
	"""
	this function takes a huge batch of training data and randomly samples from it once per epoch to reduce overfitting
	"""
	
	messages = 'Epoch\tTop-1\tTop-5\tTop-10\n'
	for epoch in range(epochs+1):
		batch_size = 128#min(32*(2**(epoch/2)),32768)#so we'll max out our batch size after 20 epochs

		training_strings = randomize(training_strings)

		state, action, outcome = BoardUtils().string_array_to_dataset(training_strings[:training_batch_size], edge, total_channels)
		
		if epoch and epoch % 5 == 0:
			print ('Saving Model:',name+' '+str(epoch))
			save_model(name+' '+str(epoch), model)
				
		if is_value_network:
			model.fit(state, outcome, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.1)
		else:
			if evaluation_batch_size:
				top1, top5, top10 = evaluate_model_predictions(training_strings[training_batch_size:training_batch_size+evaluation_batch_size], model, evaluation_batch_size, edge, total_channels)
				messages += str(epoch)+'\t'+str(top1)+'\t'+str(top5)+'\t'+str(top10)+'\n'
				print (messages, 'batch size =',batch_size)

			model.fit(state, action, batch_size=batch_size, epochs=1, verbose=1)

def batch_evaluate_model_predictions(state, action, model, edge, total_channels):

	predictions = model.predict(state)
	
	top1, top5, top10 = 0, 0, 0
	for sample in range(len(state)):
		ground_truth = 0
		by_preference = []
		for i in range(len(predictions[sample])):
			by_preference.append((predictions[sample][i], i))

			# i needed to arrange the preferences anyway, so now i can hitch a ride to find what the ground truth is
			if action[sample][i] == 1:
				ground_truth = i
				
		by_preference.sort()
		by_preference.reverse()

		# now we have our sorted preferences and we have (hopefully) located the ground truth answer, let's see if they match
		for i in range(10):
			if i == 0 and by_preference[i][1] == ground_truth:		top1 += 1
			if i <  5 and by_preference[i][1] == ground_truth:		top5 += 1
			if i < 10 and by_preference[i][1] == ground_truth:		top10+= 1

	return top1/float(len(state)), top5/float(len(state)), top10/float(len(state))

from multiprocessing import Process, Queue

def remove_index_from_array(index,array):
	if len(array) <= 1:
		return []
	
	if index == 0:
		return array[1:]
	
	if index == len(array)-1:
		return array[:-1]
	
	return array[:index] + array[index+1:]

def string_array_to_dataset_worker(worker, bundle, q, edge, total_channels):

	if total_channels == 2:
		state, action, outcome = BoardUtils2ch().string_array_to_dataset(bundle, edge)
		q.put([state, action, outcome])
	elif total_channels == 17:
		state, action, outcome = BoardUtils17ch().string_array_to_dataset(bundle, edge)
		q.put([state, action, outcome])

class Generator():
	def __init__(self, source_path, capacity, channels):
		self.CHANNELS = channels
		self.source_path = source_path
		self.source = open(source_path,'r')
		self.capacity = capacity
		self.reserve = capacity
		print ('Generator started with',self.capacity,'samples')
		
	def generate(self, requested_samples):

		edge = 19
		threadcount = 4
		bundle_size = 1

		state = np.zeros(requested_samples*edge*edge*self.CHANNELS, dtype=bool).reshape(requested_samples, edge, edge, self.CHANNELS)
		action = np.zeros(requested_samples*edge*edge, dtype=int).reshape(requested_samples, edge*edge)
		outcome = np.zeros(requested_samples).reshape(requested_samples)

		q = Queue()

		process = []
		
		counter = 0
		start_time = time.time()
		process_is_alive = False
		next_file_check = time.time()
		outstanding_samples = requested_samples
		while 0 < outstanding_samples or 0 < q.qsize() or process_is_alive:
			
			if next_file_check < time.time():
				if os.path.exists('./workers'):
					lines = open('workers','r').read().split('\n')[:-1]
					for line in lines:
						if line.find('workers=') != -1 and len(line.split('=')) == 2:
							w = line.split('=')[1]
							if w.isdigit():
								if threadcount != int(w):
									threadcount = int(w)
									print ('workers changed to ',threadcount)
						elif line.find('bundle=') != -1 and len(line.split('=')) == 2:
							b = line.split('=')[1]
							if b.isdigit():
								if bundle_size != int(b):
									bundle_size = int(b)
									print ('bundle size changed to',bundle_size)

				next_file_check += 10
				
			process_is_alive = False
			
			for i in range(len(process)-1,-1,-1):
				if process[i].is_alive() == False:
					process[i].join()
					process = remove_index_from_array(i, process)
				else:
					process_is_alive = True

			if len(process) < threadcount and outstanding_samples:

				bundle = []
				for i in range(bundle_size):
					if outstanding_samples > 0:
						if self.reserve:
							bundle.append(self.source.readline())
							outstanding_samples -= 1
							self.reserve -= 1
						else:
							self.source = open(self.source_path,'r')
							self.reserve = self.capacity

				process.append(Process(target=string_array_to_dataset_worker, args=(len(process), bundle, q, 19, self.CHANNELS)))
				process[-1].start()

			if q.qsize() != 0:
				result = q.get()

				r_state = result[0]
				r_action = result[1]
				r_outcome = result[2]
				
				for i in range(len(r_state)):
					state[counter] = r_state[i]
					action[counter] = r_action[i]
					outcome[counter] = r_outcome[i]
					counter += 1

					if counter % 100000 == 0:
						print ('Written string', counter, outstanding_samples, q.qsize(), str(counter / (time.time() - start_time))[:6])

		return state, action, outcome

def batch_training_with_generators(g_train, g_val, model, epochs, training_batch_size, validation_batch_size, batch_size, name):

	print ('Training Model Summary')
	model.summary()

	#print ('Trainable Layers\nName\tTrainable')
	#for layer in model.layers:
	#	print (layer.name, '\t', layer.trainable)

	messages = 'Epoch\tTop-1\tTop-5\tTop-10\n'
	for e in range(epochs):

		state, action, outcome = g_train.generate(training_batch_size)
		model.fit(state, action, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.1)

		state, action, outcome = g_val.generate(validation_batch_size)
		top1, top5, top10 = batch_evaluate_model_predictions(state, action, model, 19, 17)

		messages += str(e)+'\t'+str(top1)+'\t'+str(top5)+'\t'+str(top10)+'\n'
		print (messages, 'batch size =',batch_size)

		print ('Saving Model:',name+' '+str(e))
		save_model(name+' '+str(e), model)

def sgf_collection_to_training_dataset(worker, bundle, q, edge, total_channels):

	if total_channels == 2:
		state, action, outcome = BoardUtils2ch().sgf_array_to_dataset(bundle, edge)
		string_array = BoardUtils2ch().dataset_to_string_array(state, action, outcome, edge)
		q.put(string_array)
	elif total_channels == 17:
		state, action, outcome = BoardUtils17ch().sgf_array_to_dataset(bundle, edge)
		string_array = BoardUtils17ch().dataset_to_string_array(state, action, outcome, edge)
		q.put(string_array)

def threaded_sgf_file_to_training_dataset_file( sgf_file, training_dataset_file, edge, total_channels, threadcount ):

	saves = open(training_dataset_file,'w')
	collection = open(sgf_file,'r').read().split('\n')[:-1]

	print (len(collection),'games loaded into dataset.')

	q = Queue()

	process = []
	bundle_size = 1
	
	counter = 0
	counter_reset = 0
	start_time = time.time()
	process_is_alive = False
	next_file_check = time.time()
	while len(collection) or 0 < q.qsize() or process_is_alive:
		
		if next_file_check < time.time():
			if os.path.exists('./workers'):
				reset = False
				lines = open('workers','r').read().split('\n')[:-1]
				for line in lines:
					if line.find('workers=') != -1 and len(line.split('=')) == 2:
						w = line.split('=')[1]
						if w.isdigit():
							if threadcount != int(w):
								threadcount = int(w)
								print ('workers changed to ',threadcount)
								reset = True
					elif line.find('bundle=') != -1 and len(line.split('=')) == 2:
						b = line.split('=')[1]
						if b.isdigit():
							if bundle_size != int(b):
								bundle_size = int(b)
								print ('bundle size changed to',bundle_size)
								reset = True

				if reset:
					start_time = time.time()
					counter_reset = counter

			next_file_check += 10
			
		process_is_alive = False
		
		for i in range(len(process)-1,-1,-1):
			if process[i].is_alive() == False:
				process[i].join()
				process = remove_index_from_array(i, process)
			else:
				process_is_alive = True

		if len(process) < threadcount and len(collection):

			bundle = []
			for i in range(bundle_size):
				if len(collection):
					bundle.append(collection.pop())

			process.append(Process(target=sgf_collection_to_training_dataset, args=(len(process), bundle, q, edge, total_channels)))
			process[-1].start()

		if q.qsize() != 0:
			game = q.get()

			for state in game:
				saves.write(state + '\n')

				counter += 1

				if counter % 1000 == 0:
					print ('Written string', counter, len(collection), q.qsize(), str((counter - counter_reset) / (time.time() - start_time))[:6])

		time.sleep(0.01)

	saves.close()

def copy_and_freeze_layers( static_model, update_model, layers_to_copy, layers_to_freeze ):
	print ('Static Model Summary')
	static_model.summary()

	for layer in layers_to_copy:
		update_model.layers[layer].set_weights(static_model.layers[layer].get_weights())
		print ('Layer',update_model.layers[layer].name,'copied')

	for layer in layers_to_freeze:
		update_model.layers[layer].trainable = False
		print ('Layer',update_model.layers[layer].name,' - Trainable:',update_model.layers[layer].trainable)

	update_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print ('Update Model Summary')
	update_model.summary()


lr_t 	= 0.002
lr_ft 	= lr_t / 10
lr_vft	= lr_ft / 10
lr_vvft	= lr_vft / 10

def easy_pretraining(start_depth, depth_step, static_policy, model_conv_layers, model_dense_layers, g_train, training_batch_size, batch_size, epochs, threadcount):

	print ('Beginning Pretraining for:',model_conv_layers, model_dense_layers)

	edge = 19
	total_channels = 17
		
	static_policy_path = None
	static_layers = start_depth

	messages = 'Epoch\tTop-1\tTop-5\tTop-10\n'
	for i in range(start_depth+depth_step,len(model_conv_layers)+1,depth_step):

		policy_network = build_network(total_channels, model_conv_layers[:i], model_dense_layers, edge, 'categorical_crossentropy', False)

		if static_policy != None:
			copy_and_freeze_layers(static_policy, policy_network, range(static_layers), range(static_layers))

		print ('Training Model Summary')
		policy_network.summary()

		print ('Trainable Layers\nName\tTrainable')
		for layer in policy_network.layers:
			print (layer.name, '\t', layer.trainable)
		
		if len(range(static_layers)):
			state, action, outcome = [], [], []
			state, action, outcome = g_train.generate(training_batch_size)
			policy_network.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_t), metrics=['accuracy'])
			policy_network.fit(state, action, batch_size=batch_size, epochs=epochs, verbose=1)

			for layer in policy_network.layers:
				layer.trainable = True
				print (layer.name, '\t', layer.trainable)

		state, action, outcome = [], [], []
		state, action, outcome = g_train.generate(training_batch_size)
		policy_network.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_ft), metrics=['accuracy'])
		policy_network.fit(state, action, batch_size=batch_size, epochs=epochs, verbose=1)

		state, action, outcome = [], [], []
		state, action, outcome = g_train.generate(training_batch_size//10)
		top1, top5, top10 = batch_evaluate_model_predictions(state, action, policy_network, edge, total_channels)
		messages += str(i)+'\t'+str(top1)+'\t'+str(top5)+'\t'+str(top10)+'\n'
		print (messages)

		static_policy_path = 'policy network - kgs - pretrain layer '+str(i)

		print ('Saving Model:',static_policy_path)
		save_model(static_policy_path, policy_network)

		static_layers = i
		static_policy = load_model(static_policy_path)

def swap(array, a, b):
	temp = array[a]
	array[a] = array[b]
	array[b] = temp

# Python 3 program for left rotation of matrix by 90 
# degree without using extra space 

D = 19
R = D
C = R

# After transpose we swap elements of column 
# one by one for finding left rotation of matrix 
# by 90 degree 
def reverseColumns(arr): 
	for i in range(C): 
		j = 0
		k = C-1
		while j < k: 
			t = arr[j][i] 
			arr[j][i] = arr[k][i] 
			arr[k][i] = t 
			j += 1
			k -= 1

# Function for do transpose of matrix 
def transpose(arr): 
	for i in range(R): 
		for j in range(i, C): 
			t = arr[i][j] 
			arr[i][j] = arr[j][i] 
			arr[j][i] = t 

# Function for print matrix 
def printMatrix(arr): 
	for i in range(R): 
		for j in range(C): 
			print(str(arr[i][j]), end =" ") 
		print() 

# Function to anticlockwise rotate matrix 
# by 90 degree 
def rotate90(arr): 
	transpose(arr) 
	reverseColumns(arr) 

def flip(a):
	for i in range(R):
		for j in range(C//2):
			t = a[i][j]
			a[i][j] = a[i][C-j-1]
			a[i][C-j-1] = t
	return a

def generate_symmetries(array):
	all_symmetries = []
	for i in range(4):
		all_symmetries.append(deepcopy(array))
		all_symmetries.append(flip(deepcopy(array)))
		rotate90(array)
	return all_symmetries

sgf_games = []#open('go_kgs_6d+_games','r').read().split('\n')[:-1]
#print (len(sgf_games),'sgf games loaded')
#sgf_symm = open('go_kgs_6d+_symmetries','w')

_iter = 0
for game in sgf_games:
	moves = BoardUtils().sgf_to_moves(game)
	
	array = []
	for i in range(D):
		axis = []
		for j in range(D):
			axis.append([])
		array.append(axis)

	passes = []
	for m in range(len(moves)):
		if moves[m] == None:
			passes.append((m,None))
		else:
			y = moves[m][0]
			x = moves[m][1]
			array[y][x].append(m)
	
	symmetries = generate_symmetries(array)

	for sym in symmetries:
		moves = []
		for p in passes:
			moves.append(p)
		
		for i in range(D):
			for j in range(D):
				for k in sym[i][j]:
					moves.append((k,(i,j)))
		moves.sort()
		
		move_tuples = []
		for m in moves:
			move_tuples.append(m[1])
		
		sgf_symm.write(BoardUtils().moves_to_sgf(move_tuples)+'\n')
		#print (BoardUtils().moves_to_sgf(move_tuples))

	_iter += 1

	if _iter % 1000 == 0:
		print (_iter)

def shuffle_big_file_test(bigfile, buckets, rounds):

	for r in range(rounds):
		# create an array of 100 files
		# randomly sample into those files until the main file is depleted
		# load each temporary file in turn, randomly permuting it and replacing it
		# at the OS level, concatenate them back onto the original file location
		# repeat this process several times to ensure thorough mixing
		
		t_files = []
		for i in range(buckets):
			t_files.append(open('shuffle_'+str(i),'w'))
		
		counter = 0
		with open(bigfile) as input_file:
			for entry in input_file:
				if len(entry.split()) == 3:
					t_files[int(npr()*len(t_files))].write(entry)
					
					counter += 1
					
					if counter % 1000000 == 0:
						print (r,counter,'entries written')
				else:
					print ('malformed packet found at',counter)
					counter += 1
		
		for t_fo in t_files:
			t_fo.close()
		
		for i in range(buckets):
			if i%10 == 0:
				print (r,'shuffling number',i)
			
			shuffle = open('shuffle_'+str(i),'r').read().split('\n')[:-1]
			for j in range(len(shuffle)):
				swap(shuffle, j, int(npr()*len(shuffle)))
			
			save = open('shuffle_'+str(i),'w')
			for j in range(len(shuffle)):
				save.write(shuffle[j]+'\n')
			save.close()

		save = open(bigfile, 'w')
		for i in range(buckets):
			shuffle = open('shuffle_'+str(i),'r').read().split('\n')[:-1]
			for j in range(len(shuffle)):
				save.write(shuffle[j] + '\n')
		save.close()

def split_big_file_tvt(bigfile, training_limit, validation_limit, test_limit):

	ds_train 	= open(bigfile+'_train'		,'w')
	ds_validate = open(bigfile+'_validate'	,'w')
	ds_test 	= open(bigfile+'_test'		,'w')

	counter = 0
	with open(bigfile) as input_file:
		for entry in input_file:
			if len(entry.split()) == 3:
				if counter < training_limit:				ds_train.write(entry)
				elif counter < validation_limit:			ds_validate.write(entry)
				elif counter < test_limit:					ds_test.write(entry)
				else:										break;
			else:
				print ('malformed packet found at',counter)

			counter += 1
			
			if counter % 100000 == 0:
				print (counter,'entries written')

	ds_train.close()
	ds_validate.close()
	ds_test.close()

def build_resnet(input_shape, num_outputs, filters, blocks, layers_per_block):
	
	input 	= Input(shape=input_shape)

	conv 	= Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding="same")(input)
	norm	= BatchNormalization(axis=3)(conv)
	bk_input= Activation("relu")(norm)
	
	for block in range(blocks):
		conv	= Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding="same")(bk_input)
		norm	= BatchNormalization(axis=3)(conv)
		relu	= Activation("relu")(norm)

		conv	= Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding="same")(relu)
		norm	= BatchNormalization(axis=3)(conv)
		skip	= add([norm, bk_input])
		bk_input= Activation("relu")(skip)

	conv	= Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding="same")(bk_input)
	norm	= BatchNormalization(axis=3)(conv)
	relu	= Activation("relu")(norm)

	flatten = Flatten()(relu)
	fclayer = Dense(units=num_outputs, activation="softmax")(flatten)

	return Model(inputs=input, outputs=fclayer)
	
def go_clustering():
	g_train		= Generator('19x19_kgs_6d+_symmetries_dataset_140M_samples_3_randomization_cycles_test', 5000000)

	model = load_model('policy network - kgs 6ds - 64x8 - with 48 node bottleneck 3.14 0.4067')

	model.summary()
	
	state, action, outcome = g_train.generate(10000)
	
	print (state.shape, action.shape, outcome.shape, sum(sum(action)))

	predictions = model.predict(state, verbose=1)
	
	correct_predictions = []
	right_answer = []

	for p in range(len(predictions)):
		local_prediction = [(predictions[p][x],x) for x in range(len(predictions[p]))]
		local_prediction.sort()
		local_prediction.reverse()
		
		local_action = [(action[p][x],x) for x in range(len(action[p]))]
		local_action.sort()
		local_action.reverse()

		#print (local_prediction,'\n',local_action)
		
		if local_prediction[0][1] == local_action[0][1]:
			correct_predictions.append(state[p])
			right_answer.append(local_prediction[0][1])

	print (len(correct_predictions))
	
	model.pop()
	model.pop()

	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	
	model.summary()

	correct_predictions = np.array(correct_predictions).reshape(len(correct_predictions), 19, 19, 17)
	predictions = model.predict(correct_predictions, verbose=1)

	print (predictions.shape)

	state, correct_predictions = [], []

	from sklearn.decomposition import PCA, KernelPCA

	#pca = KernelPCA(n_components=10).fit_transform(predictions)
	pca = PCA(n_components=10).fit_transform(predictions)

	from sklearn import manifold

	tsne = manifold.TSNE()
	viz = tsne.fit_transform(pca)

	print (len(viz))

	save = open('exported cluster datafile for go','w')
	for i in range(len(viz)):
		save.write(str(viz[i][0])+'\t'+str(viz[i][1])+'\t'+str(right_answer[i])+'\n')
	save.close()



if __name__ == '__main__':

	#go_clustering()

	liberty_channels = 0
	total_channels = 17
	CHANNELS = total_channels
	THREADS = 4

	#threaded_sgf_file_to_training_dataset_file('go_kgs_6d+_symmetries', '19x19_kgs_6d+_symmetries_2ch_dataset', 19, 2, THREADS)

	# the idea is that you just leave it shuffling and then you stop it at some point once you think it's nice and mixed up
	saz_file = '19x19_kgs_6d+_symmetries_dataset'
	buckets = 1000
	rounds = 100
	#shuffle_big_file_test(saz_file,buckets,rounds)

	million = 1000*1000
	saz_file = '19x19_kgs_6d+_symmetries_dataset_140M_samples_3_randomization_cycles'
	training_limit = 	125*million
	validation_limit = 	training_limit + 10*million
	test_limit = 		validation_limit + 5*million
	#split_big_file_tvt(saz_file, training_limit, validation_limit, test_limit)

	feat_plains = 32
	conv_layers = 4

	filters = feat_plains
	resnet_blocks = conv_layers
	layers_per_block = 1
	
	#model_conv_layers = [[feat_plains,(3,3)]]*conv_layers + [[4,(1,1)]]
	#model_dense_layers = []
	model_conv_layers = [[feat_plains,(3,3)]]*conv_layers + [[2,(1,1)]]
	model_dense_layers = []
	training_batch_size = 100000
	validation_batch_size = training_batch_size//10
	batch_size = 128
	epochs = (2*million)//training_batch_size
	edge = 19
	'''
	g_train		= Generator('19x19_kgs_6d+_symmetries_2ch_dataset', 1200000, 2)
	
	#g_train 	= Generator('19x19_kgs_6d+_symmetries_dataset_140M_samples_3_randomization_cycles_train',		125*million)
	#g_val 		= Generator('19x19_kgs_6d+_symmetries_dataset_140M_samples_3_randomization_cycles_validate',	10*million)
	#g_test		= Generator('19x19_kgs_6d+_symmetries_dataset_140M_samples_3_randomization_cycles_test',		5*million)
	
	#policy_network = load_model('policy network - kgs 6ds - 16x4 - 357 673 779')
	policy_network = build_network((19,19,2), model_conv_layers, model_dense_layers, 19, 'categorical_crossentropy', False)
	#policy_network = build_resnet((19,19,17), 361, filters, resnet_blocks, layers_per_block)

	#easy_pretraining(0, 1, None, model_conv_layers, model_dense_layers, g_train, training_batch_size, batch_size, 1, THREADS)
	
	#policy_network = load_model('policy network - kgs - 0 - 64x6 - 8')
	#policy_network = build_network(total_channels, model_conv_layers, model_dense_layers, 19, 'categorical_crossentropy', False)

	#copy_and_freeze_layers(static_network, policy_network, range(14), [])
	
	#for layer in policy_network.layers:
	#	layer.trainable = True


	learning_rate = lr_t
	for _iter in range(10):
		print ('learning rate set to',learning_rate)
		policy_network.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
		batch_training_with_generators(g_train, g_train, policy_network, epochs, training_batch_size, validation_batch_size, batch_size, 'policy network - kgs - '+str(_iter)+' - '+str(feat_plains)+'x'+str(conv_layers)+' -')
		#batch_training_with_generators(g_train, g_val, policy_network, epochs, training_batch_size, validation_batch_size, batch_size, 'policy network - kgs - '+str(feat_plains)+'x'+str(conv_layers)+' -')
		learning_rate /= 10
	'''


	# *****************************************
	#             more strategies
	# *****************************************

	def return_uniform_preferences(board):
		
		columns = board.board.shape[1]
		board_size = board.board.size
		local_board = copy(board.board)

		preferences = []
		for i in range(board_size):
			row = i//columns
			col = i%columns
			score = 1
			preferences.append((score, (row, col)))
		preferences.sort()
		preferences.reverse()
		
		return preferences

	def calculate_value_estimate(model, board, player):
		columns = board.board.shape[1]
		edge_length = columns
		board_size = board.board.size
		local_board = copy(board.board)

		state_action_string = state_action_to_string(local_board, player, (0,0))
		features, junk = string_to_state_action(state_action_string, edge_length)
		features = np.array(features).reshape(1, edge_length, edge_length, 2)
		prediction = model.predict(features)
		prediction = np.array(prediction).reshape(len(prediction))
		#print (prediction)
		return prediction

	def calculate_preferences_for_themachine(model, board, player):
		
		columns = board.board.shape[1]
		edge_length = columns
		board_size = board.board.size
		local_board = copy(board.board)

		features = BoardUtils().position_to_multistate(board, player, liberty_channels)
		#state_action_string = state_action_to_string(local_board, player, (0,0))
		#features, junk = string_to_state_action(state_action_string, edge_length)
		features = np.array(features).reshape(1, edge_length, edge_length, 2+2*liberty_channels)
		prediction = model.predict(features)
		prediction = prediction.reshape(board_size)
		
		preferences = []
		for i in range(len(prediction)):
			row = i//columns
			col = i%columns
			score = prediction[i]
			preferences.append((score, (row, col)))
		preferences.sort()
		preferences.reverse()
		
		return preferences

	def calculate_preferences_for_themachine_using_features(features, model, board, player):
		
		columns = board.board.shape[1]
		edge_length = columns
		board_size = board.board.size
		local_board = copy(board.board)

		#features = BoardUtils().position_to_multistate(board, player, liberty_channels)
		#state_action_string = state_action_to_string(local_board, player, (0,0))
		#features, junk = string_to_state_action(state_action_string, edge_length)
		#features = np.array(features).reshape(1, edge_length, edge_length, 2+2*liberty_channels)
		#print (features.shape)
		prediction = model.predict(features)
		#print (features.shape, prediction.shape)
		prediction = prediction.reshape(board_size)
		
		preferences = []
		for i in range(len(prediction)):
			row = i//columns
			col = i%columns
			score = prediction[i]
			preferences.append((score, (row, col)))
		preferences.sort()
		preferences.reverse()
		
		return preferences

	def weight_sensible_moves(preferences, board, player, moves_to_consider, limit_to_sensible_moves):
		
		my_legal_moves 		= board.P2_legal_moves
		his_legal_moves 	= board.P1_legal_moves
		his_last_liberties	= board.P1_last_liberties
		he_has_a_ko = None
		if board.ko:
			if board.ko[1] == P1_ID:
				he_has_a_ko = board.ko[0]
		if player == P1_ID:
			my_legal_moves 		= board.P1_legal_moves
			his_legal_moves 	= board.P2_legal_moves
			his_last_liberties	= board.P2_last_liberties
			he_has_a_ko = None
			if board.ko:
				if board.ko[1] == P2_ID:
					he_has_a_ko = board.ko[0]

		considerable_sum = 1e-6
		under_consideration = [(1e-6, None)]
		for p in preferences:
			w = p[0]
			r = p[1][0]
			c = p[1][1]
			if my_legal_moves[r][c]:
				if limit_to_sensible_moves:
					if (his_legal_moves[r][c] or his_last_liberties[r][c] or he_has_a_ko == (r,c)):
						considerable_sum += w
						under_consideration.append(p)
					else:
						considerable_sum += 0
						under_consideration.append((0,p[1]))
				else:
					considerable_sum += w
					under_consideration.append(p)
			
		weight_adjusted_moves = []
		for i in under_consideration[:moves_to_consider]:
			w = i[0] / float(considerable_sum)
			m = i[1]
			weight_adjusted_moves.append((w,m))
		
		weight_adjusted_moves.sort()
		weight_adjusted_moves.reverse()

		return weight_adjusted_moves

	def wwtmd(model, board, player, accumulation_target):

		preferences = calculate_preferences_for_themachine(model, board, player)
		
		under_consideration = weight_sensible_moves(preferences, board, player, 500, True)

		if under_consideration == None:
			return None

		#print ('\n')
		#for c in under_consideration[:10]:
		#	print (c)

		if accumulation_target:
			subset = 0
			accumulator = under_consideration[subset][0]
			while accumulator < accumulation_target:
				subset += 1
				accumulator += under_consideration[subset][0]
			
			selection = int(npr()*subset)
			return under_consideration[selection][1]

		return under_consideration[0][1]

	def wwtmd_from_moves(features, model, board, player, accumulation_target):

		preferences = calculate_preferences_for_themachine_using_features(features, model, board, player)
		under_consideration = weight_sensible_moves(preferences, board, player, 500, True)

		if under_consideration == None:
			return None

		#print ('\n')
		#for c in under_consideration[:10]:
		#	print (c)

		if accumulation_target:
			subset = 0
			accumulator = under_consideration[subset][0]
			while accumulator < accumulation_target:
				subset += 1
				accumulator += under_consideration[subset][0]
			
			selection = int(npr()*subset)
			return under_consideration[selection][1]

		return under_consideration[0][1]

	def faster_sensible_move_generator(board, player):
		edge = board.board.shape[0]
		size = board.board.size

		my_legal_moves = None
		his_legal_moves = None
		his_last_liberties = None
		he_has_a_ko = None
		
		if player == P1_ID:
			my_legal_moves 		= board.P1_legal_moves
			his_legal_moves 	= board.P2_legal_moves
			his_last_liberties	= board.P2_last_liberties
			he_has_a_ko = None
			if board.ko:
				if board.ko[1] == P2_ID:
					he_has_a_ko = board.ko[0]
		else:
			my_legal_moves 		= board.P2_legal_moves
			his_legal_moves 	= board.P1_legal_moves
			his_last_liberties	= board.P1_last_liberties
			he_has_a_ko = None
			if board.ko:
				if board.ko[1] == P1_ID:
					he_has_a_ko = board.ko[0]

		sensible_moves = [None]
		for r in range(edge):
			for c in range(edge):
				if my_legal_moves[r][c] and (his_legal_moves[r][c] or his_last_liberties[r][c] or he_has_a_ko == (r,c)):
					sensible_moves.append((r,c))

		selection = int(npr()*len(sensible_moves))
		return sensible_moves[selection]

	def generate_random_move(board, player):
		# assign the same value to all the positions
		preferences = return_uniform_preferences(board)
		
		# check to see which ones are sensible
		under_consideration = weight_sensible_moves(preferences, board, player, 500, True)
		
		if under_consideration == None:
			return None
		
		# and randomly select one
		selection = int(npr()*len(under_consideration))
		return under_consideration[selection][1]

	def random_rollout(board, player):
		edge_length = board.board.shape[0]

		ctr = 0
		passed = 0
		turn = player
		while passed < 2 and ctr < board.board.size:

			move = faster_sensible_move_generator(board, turn)
			errors = board.do_move(move, turn)

			if move == None:	passed += 1
			else:				passed = 0

			turn *= -1
			ctr += 1

		P1_score = 0
		P2_score = 0

		#factor this score thing into the board class - it's kind of a clunky thing to have here
		for y in range(edge_length):
			for x in range(edge_length):
				P1_score += board.P1_legal_moves[y][x] + (1 if board.board[y][x] == P1_ID else 0)
				P2_score += board.P2_legal_moves[y][x] + (1 if board.board[y][x] == P2_ID else 0)

		if P1_score < P2_score:
			return (0,1,1)
		elif P2_score < P1_score:
			return (1,0,1)
		else:
			return (0,0,1)

	def random_rollout_with_rave(board, player):
		edge_length = board.board.shape[0]

		ctr = 0
		passed = 0
		turn = player
		while passed < 2 and ctr < board.board.size:

			move = faster_sensible_move_generator(board, turn)
			errors = board.do_move(move, turn)

			if move == None:	passed += 1
			else:				passed = 0

			turn *= -1
			ctr += 1

		P1_score = 0
		P2_score = 0

		P1_amaf = np.zeros(board.board.size).reshape(edge_length, edge_length)
		P2_amaf = np.zeros(board.board.size).reshape(edge_length, edge_length)

		#factor this score thing into the board class - it's kind of a clunky thing to have here
		for y in range(edge_length):
			for x in range(edge_length):
				P1_score += board.P1_legal_moves[y][x] + (1 if board.board[y][x] == P1_ID else 0)
				P2_score += board.P2_legal_moves[y][x] + (1 if board.board[y][x] == P2_ID else 0)
				
				if board.board[y][x] == P1_ID:		P1_amaf[y][x] = 1
				if board.board[y][x] == P2_ID:		P2_amaf[y][x] = 1

		if P1_score < P2_score:
			return (0,1,1), [], P2_amaf
		elif P2_score < P1_score:
			return (1,0,1), P1_amaf, []
		else:
			return (0,0,1), [], []

	def random_rollout_with_a_chance_of_neural_net(model, chance_of_nn, board, player):
		edge_length = board.board.shape[0]

		ctr = 0
		passed = 0
		turn = player
		while passed < 2 and ctr < board.board.size:

			move = None
			if npr() < chance_of_nn:
				move = wwtmd(model, board, turn, 0.10)
			else:
				move = faster_sensible_move_generator(board, turn)

			errors = board.do_move(move, turn)

			if move == None:	passed += 1
			else:				passed = 0

			turn *= -1
			ctr += 1

		P1_score = 0
		P2_score = 0

		P1_amaf = np.zeros(board.board.size).reshape(edge_length, edge_length)
		P2_amaf = np.zeros(board.board.size).reshape(edge_length, edge_length)

		#factor this score thing into the board class - it's kind of a clunky thing to have here
		for y in range(edge_length):
			for x in range(edge_length):
				P1_score += board.P1_legal_moves[y][x] + (1 if board.board[y][x] == P1_ID else 0)
				P2_score += board.P2_legal_moves[y][x] + (1 if board.board[y][x] == P2_ID else 0)
				
				if board.board[y][x] == P1_ID:		P1_amaf[y][x] = 1
				if board.board[y][x] == P2_ID:		P2_amaf[y][x] = 1

		if P1_score < P2_score:
			return (0,1,1), [], P2_amaf
		elif P2_score < P1_score:
			return (1,0,1), P1_amaf, []
		else:
			return (0,0,1), [], []

	# this still makes us attractive for selection even if we have opponents with Qi == 1, Ui == 1
	SIMULATED_ANNEALING = False

	FECUNDITY = 1
	EQUIVALENCE_PARAMETER = 1000
	EXPLORATION_TERM = 2**(1/2.0)
	if SIMULATED_ANNEALING:
		EXPLORATION_TERM = 2.0

	WAIT = 0.5
	THREADS = 1

	def multithread_collector_for_random_rollout(board, player, q, time_limit):
		p1wins, p2wins, total = 0,0,0
		while time.time() < time_limit:
			result = random_rollout(deepcopy(board), player)
			p1wins += result[0]
			p2wins += result[1]
			total += result[2]
		q.put([p1wins,p2wins,total])

	def multithread_manager_for_random_rollout(board, player, threadcount):
		# the quick version is to start "threads" threads and have variables to receive their results, then sum them all up and call it a day

		process = [None] * threadcount

		q = Queue()
		for i in range(threadcount):
			process[i] = Process(target=multithread_collector_for_random_rollout, args=(deepcopy(board), player, q, time.time() + WAIT))
		
		for i in range(threadcount):
			process[i].start()
		
		time.sleep(WAIT)

		all_ready = 0
		while all_ready < threadcount:
			for i in range(threadcount):
				if not process[i].is_alive():
					all_ready += 1
			time.sleep(0.01)

		for i in range(threadcount):
			process[i].join()

		p1_wins,p2_wins,total_games = 0,0,0
		while not q.empty():
			r = q.get()
			p1_wins += r[0]
			p2_wins += r[1]
			total_games += r[2]
			#print (i, p1_wins, p2_wins, total_games)
		
		return (p1_wins, p2_wins, total_games)

	class Amaf():
		def __init__(self, size, edge):
			self.table = np.zeros(2*size,dtype=np.intc).reshape(2, edge, edge)
			self.games = 0

	class go_mcts_node():
		def __init__(self, parent, policy_network, value_network, rollout_policy, current_player, next_player, current_board, current_move, distance_from_root, using_rave, chance_of_nn):
			"""
			Make a move and save the whole board object
			
			"""
			# keep track of networks
			self.policy_network = policy_network
			self.value_network = value_network
			self.rollout_policy = rollout_policy

			# start things off by seeing what the network thinks of this position
			preferences = []
			if self.policy_network != None:
				preferences = calculate_preferences_for_themachine(policy_network, current_board, next_player)
			else:
				preferences = return_uniform_preferences(current_board)
			self.legal_move_preferences = weight_sensible_moves(preferences, current_board, next_player, 361, True)
			
			self.value_estimates = []
			if self.value_network != None:
				self.value_estimates.append(calculate_value_estimate(self.value_network, current_board, current_player)[0])

			# set up our parent and children nodes - the children are start out uninitialized to save memory and will be expanded as needed, but there is one per board position and there's another for "pass"
			self.parent = parent
			self.children = [None]*len(self.legal_move_preferences)
			self.children_map = np.array([None]*current_board.board.size).reshape(current_board.board.shape[0], current_board.board.shape[0])

			# as well as game related details
			self.current_player = current_player
			self.current_board = deepcopy(current_board)
			self.current_move = copy(current_move)
			self.next_player = next_player

			# and mcts stats
			self.P1_wins = 0
			self.P2_wins = 0
			self.total_games = 0

			# and rave stats
			self.amaf = Amaf(current_board.board.size, current_board.board.shape[0])
			self.using_rave = using_rave

			self.chance_of_nn = chance_of_nn

			#print ('Depth:',distance_from_root, self.amaf.games)
			self.distance_from_root = distance_from_root
			self.max_depth = distance_from_root
			
		def get_child_state(self):
			#state = []
			#for child in self.children:
			#	if child != None:
			#		state.append((child.current_move, child.get_performance_tuple(), float(sum(child.value_estimates))/len(child.value_estimates)))
			#return state
			
			state = []

			for child in self.children:
				if child != None and child.current_move == None:
					if self.value_network != None:
						state.append((child.current_move, child.get_performance_tuple(), float(sum(child.value_estimates))/len(child.value_estimates)))
					else:
						state.append((child.current_move, child.get_performance_tuple()))

			for lm in self.legal_move_preferences:
				policy_value = lm[0]
				this_move = lm[1]
				
				if this_move != None:
					row = this_move[0]
					col = this_move[1]
					child = self.children_map[row][col]
					if child != None:
						if self.value_network != None:
							state.append((child.current_move, child.get_performance_tuple(), policy_value, float(sum(child.value_estimates))/len(child.value_estimates)))
						else:
							state.append((child.current_move, child.get_performance_tuple(), policy_value))
			return state

		def get_node_state(self):
			return ''
			
			if len(self.children) == 0:
				return None
			
			edge = self.current_board.board.shape[0]
			size = self.current_board.board.size
			
			rec_total_games = np.zeros(size,dtype=float).reshape(edge, edge)
			rec_win_ratios  = np.zeros(size).reshape(edge, edge)
			for child in self.children:
				outcomes = child.get_performance_tuple()
				player = child.current_player
				move = child.current_move
				
				if move != None:
					wins = 0
					if player == P1_ID:
						wins = outcomes[0]
					else:
						wins = outcomes[1]
					
					total_games = outcomes[2]
					
					win_ratio = 0
					if total_games:
						win_ratio = wins / float(total_games)
					
					row = move[0]
					col = move[1]
					
					rec_total_games[row][col] = total_games
					rec_win_ratios[row][col] = win_ratio
			
			rec_total_games = rec_total_games.reshape(size)
			rec_total_games /= max(rec_total_games)
			
			rec_win_ratios = rec_win_ratios.reshape(size)

			string = str(rec_total_games[0])
			for p in rec_total_games[1:]:
				string += ',' + str(p)
			
			string += '\t' + str(rec_win_ratios[0])
			for p in rec_win_ratios[1:]:
				string += ',' + str(p)

			return string

		def get_performance_tuple(self):
			return (self.P1_wins, self.P2_wins, self.total_games)
		
		def assemble_legal_moves(self):
			return copy(self.legal_move_preferences)
		
		def deallocate(self, new_root):
			self.children_map = []
			#self.current_board = []
			self.legal_move_preferences = []
			self.amaf = None
			
			for child in self.children:
				if child != None and child != new_root:
					child.deallocate(None)
			
			self.children = []
		
		def simulate_move(self, current_move, player):

			new_root = None
			if current_move != None:
				new_root = self.children_map[current_move[0]][current_move[1]]
			
			return new_root

		def advance_game(self, current_move, player):
			"""
			Advance the game by one move.  The "next_player" ID will be verified and then I'll try to match the move with a legal move and return that child.
			
			"""
			
			if player != self.next_player:
				print ('Warning: The wrong player seems to be placing a stone')
			
			new_root = None
			if current_move != None:
				new_root = self.children_map[current_move[0]][current_move[1]]
			
			if new_root == None:
				if self.current_board.this_move_is_legal(current_move, player):
					
					new_board = deepcopy(self.current_board)
					new_board.do_move(current_move, self.next_player)

					new_root = go_mcts_node(self, self.policy_network, self.value_network, self.rollout_policy, self.next_player, self.current_player, new_board, current_move, self.distance_from_root + 1, self.using_rave, self.chance_of_nn)

			self.deallocate(new_root)
			
			#new_root.parent = None
			
			return new_root

		#Selection - will be handled by the mcts driver class
		def selection(self, fecundity, temperature):
			"""
			1) If we're at search depth, then run a simulation and update your superordinate nodes
			
			2) Otherwise, organize your children to figure out which one is most in need of attention, call it with a decremented search_depth and the opposite player
			
			If you don't have children, then have some and go back to step 2
			
			"""

			if fecundity == 0:
				self.run_simulation()
			else:
				# calculate scores for all children
				# pick the highest scoring child
				# if that child exists, call its selection function, passing along the temperature and fecundity
				# if the child does not exist, generate a random number against the fecundity to decide whether to spawn the child
				#	if you spawn a child, set fecundity to zero and call the child
				#	if you do not, then call self.run_simulation() from here

				child_scores = []
				for i in range(len(self.legal_move_preferences)):
					child_scores.append((self.calculate_child_score(i, temperature), i))

				child_scores.sort()
				child_scores.reverse()
				
				child_index = child_scores[0][1]
				if self.children[child_index] == None:
					if npr() < fecundity and self.current_move != None:
						self.expand_one(child_index)
						self.children[child_index].selection(0, temperature)
					else:
						self.run_simulation()
				else:
					self.children[child_index].selection(fecundity, temperature)

		def get_most_simulated(self, top_n):
			candidates = []
			for child in self.children:
				if child != None:
					candidates.append(child)

			if len(candidates):
				potential_boards = []
				for child in candidates:
					simulation_count = child.total_games
					potential_boards.append((simulation_count, child))
				potential_boards.sort()
				potential_boards.reverse()

				top_moves = potential_boards[:top_n]
				relevant_simulations = 0
				for tm in top_moves:
					relevant_simulations += tm[0]
				
				accumulator_target = npr() * relevant_simulations
				running_total = 0
				for tm in top_moves:
					running_total += tm[0]
					if accumulator_target <= running_total:
						selected_move = tm[1]
						his_move = selected_move.current_move
						player_ID = self.current_player
						
						return his_move, player_ID

				#most_simulated_child = potential_boards[0][1]
				#his_move = most_simulated_child.current_move
				#player_ID = self.current_player
				
				#return his_move, player_ID
			
			return None, None
		
		#Expand
		def expand_one(self, child_index):
			if self.children[child_index] == None:
				this_move = self.legal_move_preferences[child_index][1]
				
				new_board = deepcopy(self.current_board)
				new_board.do_move(this_move, self.next_player)
				
				self.children[child_index] = go_mcts_node(self, self.policy_network, self.value_network, self.rollout_policy, self.next_player, self.next_player * -1, new_board, this_move, self.distance_from_root + 1, self.using_rave, self.chance_of_nn)
				if this_move != None:
					self.children_map[this_move[0]][this_move[1]] = self.children[child_index]

		def expand_all(self):
			for i in range(len(self.legal_move_preferences)):
				self.expand_one(i)

		#Simulation
		def run_simulation(self):
			"""
			run simulation, update my score, and update the nodes above me in the tree
			
			"""
			if self.value_network != None:
				self.add_value_estimate()
				return
			
			#result, p1_amaf, p2_amaf = random_rollout_with_rave(deepcopy(self.current_board), self.next_player)
			result, p1_amaf, p2_amaf = random_rollout_with_a_chance_of_neural_net(self.rollout_policy, self.chance_of_nn, deepcopy(self.current_board), self.next_player)

			# start by updating myself
			self.update_game_record(result, p1_amaf, p2_amaf, self.distance_from_root)

			# next, if i happen to be the winner of this simulation, update that moves list with my move before sending it up the tree
			if self.current_move != None:
				if self.current_player == P1_ID and p1_amaf != []:
					self.amaf.table[0][self.current_move[0]][self.current_move[1]] += 1
				if self.current_player == P2_ID and p2_amaf != []:
					self.amaf.table[1][self.current_move[0]][self.current_move[1]] += 1

			self.update_superordinate_nodes(result, p1_amaf, p2_amaf, self.distance_from_root)
			
		def add_value_estimate(self):
			current_node = self
			while current_node.parent != None:
				current_node = current_node.parent
				# cool and everything, but try and make sure that there's some adjustment somewhere to account for the fact that what's good for one player is bad for the other
				# probably need to do a fair amount of work correlating who generates the value with the way it filters back up through the network
				# one nice way of auditing might include functions to display a tuple of who generated the value, the original value, who is receiving the value, and the new value
				# that way, you could be absolutely sure that, for instance, a move that is determined to be good for white is recorded as being bad for black
				if self.current_move != None:
					if self.current_player == current_node.current_player:
						current_node.value_estimates.append(self.value_estimates[0])
					else:
						current_node.value_estimates.append(-1*self.value_estimates[0])

					#print (self.current_player, self.value_estimates[0], current_node.current_player, self.value_estimates[0])
					#current_node.value_estimates.append(self.value_estimates[0])

				current_node.update_game_record((0,0,1), [], [], self.distance_from_root)

		#Backprop
		def update_superordinate_nodes(self, result, p1_amaf, p2_amaf, max_depth):
			current_node = self
			while current_node.parent != None:
				current_node = current_node.parent
				current_node.update_game_record(result, p1_amaf, p2_amaf, max_depth)

		def update_game_record(self, result, p1_amaf, p2_amaf, max_depth):
			self.P1_wins += result[0]
			self.P2_wins += result[1]
			self.total_games += result[2]
			
			if self.using_rave and self.amaf != None:
				if p1_amaf != []:
					for r in range(self.current_board.board.shape[0]):
						for c in range(self.current_board.board.shape[0]):
							self.amaf.table[0][r][c] += p1_amaf[r][c]
				if p2_amaf != []:
					for r in range(self.current_board.board.shape[0]):
						for c in range(self.current_board.board.shape[0]):
							self.amaf.table[1][r][c] += p2_amaf[r][c]
				self.amaf.games += 1

			self.max_depth = max(self.max_depth, max_depth)

		def calculate_child_score(self, move_index, temperature):
			
			"""
			Get our variables together so we can calculate a score.
			
			"""
			
			# get a copy of the move under consideration so we can check the mcts data if the child exists
			this_move = self.legal_move_preferences[move_index][1]

			# how much the neural network wants to play this move
			network_term = self.legal_move_preferences[move_index][0]

			value_term = 0
			total_mcts_games = 0
			if self.value_network != None:
				if self.children[move_index] != None:
					value_term = float(sum(self.children[move_index].value_estimates))/len(self.children[move_index].value_estimates)

			mcts_term = 0
			if self.children[move_index] != None:
				performance = self.children[move_index].get_performance_tuple()
				
				total_mcts_games = performance[2]
				if total_mcts_games:
					if self.next_player == P1_ID:
						mcts_term = performance[0] / float(total_mcts_games)
					else:
						mcts_term = performance[1] / float(total_mcts_games)

			rave_term = 0
			if this_move != None and self.amaf.games:
				if self.next_player == P1_ID:
					rave_term = self.amaf.table[0][this_move[0]][this_move[1]] / float(self.amaf.games)
				else:
					rave_term = self.amaf.table[1][this_move[0]][this_move[1]] / float(self.amaf.games)

			if self.value_network == None:
				value_term = mcts_term
				if self.using_rave:
					k = 3000#self.current_board.board.size
					weight = (k/float(3*total_mcts_games + k))**0.5
					value_term = (1-weight)*mcts_term + weight*rave_term

			UCT = EXPLORATION_TERM
			if total_mcts_games:
				radicand = ( math.log(self.total_games) / math.log(math.e) ) / float(total_mcts_games)
				UCT = EXPLORATION_TERM*(radicand**(1/2.0))

			#if UCT != EXPLORATION_TERM:
			#	print ('policy:',network_term,'value:',value_term, 'explore:',temperature*UCT,'action value:',network_term + value_term + temperature*UCT)
			return network_term + value_term + temperature*UCT

	class go_mcts():
		def __init__(self, policy_network, value_network, rollout_policy, current_board, current_player, next_player, use_simulated_annealing, use_rave, fecundity, chance_of_nn):

			'''
			class go_mcts_node():
				def __init__(self, parent, model, Qi, current_player, next_player, current_board, current_move, moves_to_consider, distance_from_root):
			'''

			if rollout_policy == None:
				chance_of_nn = 0

			self.root = go_mcts_node(None, policy_network, value_network, rollout_policy, current_player, next_player, deepcopy(current_board), None, 0, use_rave, chance_of_nn)
			self.root.expand_all()
			
			self.use_simulated_annealing = use_simulated_annealing
			self.use_rave = use_rave
			self.fecundity = fecundity
			
			self.total_time = 0
			self.total_simulations = 0

			self.game_in_progress = True
		
		def get_moves_forecast(self):
			moves_forecast = []
			walker = self.root
			while walker != None:
				move, player = walker.get_most_simulated(1)
				if player != None:
					moves_forecast.append((move, -1*player))
				walker = walker.simulate_move(move, player)
			return moves_forecast

		def get_mcts_board(self):
			if self.root != None:
				return self.root.current_board
			else:
				print ('Error: There is no root node.')
			return None

		def get_next_player(self):
			if self.root == None:
				return None
			else:
				return self.root.next_player
		
		def get_legal_moves(self):
			if self.root == None:
				return None
			else:
				return self.root.assemble_legal_moves()

		def get_favorite_moves(self, n_moves):
			if self.root == None:
				return []
			
			simulation_preference = []
			for status in self.root.get_child_state():
				move = status[0]
				simulations = status[1]
				simulation_preference.append((simulations[2], move))
			
			simulation_preference.sort()
			simulation_preference.reverse()
			return simulation_preference[:n_moves]

		def get_mcts_scores(self):
			if self.root == None:
				return None
			else:
				return self.root.assemble_mcts_scores()
		
		def get_node_state(self):
			return self.root.get_node_state()

		def generate_move(self, search_time_budget, top_n):
			"""
			Impose a time limit on how long a decision can take.  Then return the most simulated move and update the root node.
			"""

			if self.root == None:
				return None

			start_time = time.time()
			time_limit = start_time + search_time_budget

			countdown = time.time() + 10
			simulation_count = 0
			while time.time() < time_limit:
				
				temperature = 1
				if self.use_simulated_annealing:
					temperature = (time_limit - time.time())/(search_time_budget+1e-10)

				self.root.selection(self.fecundity, temperature)
				simulation_count += 1

				if countdown <= time.time():
					print ('Time Until Decision:',time_limit - time.time(),'\tSimulation Count:',simulation_count,'\tTemperature:',temperature,'\tTree Depth:',self.root.distance_from_root,self.root.max_depth)
					countdown = time.time() + 10
			'''
			print ('Player 1 Last Liberties:\n',self.root.current_board.P1_last_liberties)
			print ('Player 2 Last Liberties:\n',self.root.current_board.P2_last_liberties)
			last_liberties = self.root.current_board.last_liberties
			last_liberties.sort()
			for i in last_liberties:
				print (i)
			'''

			total_simulations = 0
			print ('RAVE Map:',self.root.amaf.games,'\n',self.root.amaf.table[0],'\n',self.root.amaf.table[1])
			print ('Node state for root:',self.root)
			child_state = self.root.get_child_state()
			child_state.sort()
			for i in child_state:
				print (i)
				total_simulations += i[1][2]
			self.total_time += time.time() - start_time
			self.total_simulations += total_simulations
			simulation_count = total_simulations

			print ('Simulations Run:',simulation_count,'Cumulative Performance:',float(self.total_simulations)/self.total_time)

			move, player = self.root.get_most_simulated(top_n)
			
			return move

		def undo_move(self):
			if self.root.parent != None:
				self.root = self.root.parent
			else:
				print ('Error: Cannot Undo.  No valid parent.')

		def register_move(self, current_move, current_player):
			"""
			Enable the tree to be updated as other players make their moves
			"""

			self.root = self.root.advance_game(current_move, current_player)
			self.root.expand_all()

			if self.root == None:
				self.game_in_progress = False
			
			self.display_board()
		
		def display_board(self):
			if self.root != None:
				print (self.root.get_performance_tuple())
				self.root.current_board.assemble_displays()
				#territory, score = BoardUtils().tromp_taylor_scoring(self.root.current_board)
				#print ('Score:',score,'\nTerritory:\n',territory)

	class MCTSUtils():
		def __init__(self):
			pass
		
		def run_simulation(self, policy_network, value_network, edge, use_simulated_annealing, use_rave, think_time, fecundity, chance_of_nn, top_n):

			moves = []

			rollout_policy = None

			session = go_mcts(policy_network, value_network, rollout_policy, Board(edge), None, 1, use_simulated_annealing, use_rave, fecundity, chance_of_nn)
			
			turn = 1
			passed = 0
			while passed < 2 and turn <= 2*(edge*edge):
				player = 0
				errors = 0
				
				if turn & 1:	player = 1
				else:			player = -1

				moves.append(session.generate_move(think_time, top_n))
				session.register_move(moves[-1], player)
				print ('Move played above was',moves[-1],'by',player)

				if moves[-1] == None:		passed += 1
				else:						passed = 0

				turn += 1

			return BoardUtils().moves_to_sgf(moves)
		
		def training_loop(self):
			# game pool - most recent 10,000
			# generate  - 500 games of self-play per training loop
			# using     - the top 5 networks as determined by win rate in a 6-way tournament (if there are <6 networks, then generate some random ones)
			# train on  - say, train for 20 epochs over 10000 state-action pairs randomly sampled per epoch
			# eliminate - toss the new network in against the 5 strongest networks and play 10,000 games in a 6-way tournament, keeping the top-5 by win rate.
			
			# this will require 
			pass

		def maigo_zero(self, policy_network, value_network, edge, batch_size, pool_size, training_loops, games_per_loop, think_time, top_n):
			"""
			Automate the self-play training loop
			"""

			if policy_network != None:			policy_network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			if value_network != None:			value_network.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

			games_history = open('latest_games_thread_02','r').read().split('\n')[:-1]
			print (len(games_history),'historical game records loaded')

			save = open('latest_games_thread_02','w')
			for loop in range(training_loops):
				for game in range(games_per_loop):
					print ('Game Number:',game)
					games_history.append(self.run_simulation(policy_network, value_network, edge, SIMULATED_ANNEALING, True, think_time, 1, 0, top_n))
					save.write(games_history[-1]+'\n')

				if pool_size < len(games_history):
					state, action, outcome = BoardUtils().sgf_array_to_dataset(games_history[-pool_size:], edge, liberty_channels)
					
					saz_tuple = []
					for i in range(len(state)):
						saz_tuple.append((state[i], action[i], outcome[i]))
					
					saz_tuple = randomize(saz_tuple)
					
					state, action, outcome = [], [], []
					for s, a, z in saz_tuple[:batch_size]:
						state.append(s)
						action.append(a)
						outcome.append(z)
					
					state = np.array(state).reshape(len(state), edge, edge, total_channels)
					action = np.array(action).reshape(len(action), edge*edge)
					outcome = np.array(outcome).reshape(len(outcome))
					
					if policy_network != None:
						policy_network.fit(state, action, epochs=1, verbose=1)
						save_model('latest_policy', policy_network)
						save_model('policy network - zero - '+str(len(games_history)), policy_network)

					if value_network != None:
						value_network.fit(state, outcome, epochs=1, verbose=1)
						save_model('latest_value', value_network)
						save_model('value network - zero - '+str(len(games_history)), value_network)
			save.close()


	#def maigo_zero(self, policy_network, value_network, edge, batch_size, pool_size, training_loops, games_per_loop, think_time, top_n):

	policy_network = None#load_model('policy network - 5x5 32x6+32 - 01-11k')
	value_network = None#load_model('latest_value')
	edge = 9
	batch_size = 1000
	pool_size = 50000
	training_loops = 1<<20
	games_per_loop = 2500
	think_time = 20
	top_n = 3

	#MCTSUtils().maigo_zero(policy_network, value_network, edge, batch_size, pool_size, training_loops, games_per_loop, think_time, top_n)

	WIDTH = 800
	HEIGHT = 800

	h = '0123456789ABCDEF'
	def tohex(r,g,b):
		r = min(r,255)
		g = min(g,255)
		b = min(b,255)

		string = '#'
		string += h[r//16] + h[r%16]
		string += h[g//16] + h[g%16]
		string += h[b//16] + h[b%16]

		return string

	class display( Frame ):

		def redraw_and_clear(self):

			self.canvas.delete('all')

			for i in range(1,self.board.N+1):
				self.canvas.create_line(i*self.wstep-self.wstep/2,0,i*self.wstep-self.wstep/2,HEIGHT)
				self.canvas.create_line(0,i*self.hstep-self.hstep/2,WIDTH,i*self.hstep-self.hstep/2)

			for y in range(self.board.N):
				for x in range(self.board.N):
					if self.star_points[y][x]:
						self.canvas.create_oval(x*self.wstep+self.wstep/2-self.wstep/8,y*self.hstep+self.hstep/2-self.hstep/8,x*self.wstep+self.wstep/2+self.wstep/8,y*self.hstep+self.hstep/2+self.hstep/8,fill='black')
						
			for y in range(self.board.N):
				for x in range(self.board.N):
					if self.board.board[y][x]:
						if self.board.board[y][x] == P1_ID:
							self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill='black')
						elif self.board.board[y][x] == P2_ID:
							self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill='white')
						self.canvas.create_text(x*self.wstep+self.wstep/2,y*self.hstep+self.hstep/2,text=str(self.move_numbers[y][x]),fill='red')
					else:
						if self.show_liberties:
							r = 128+16*(self.board.P2_liberty_map[y][x] & LIBERTY_MASK)
							g = 128+16*(self.board.P1_liberty_map[y][x] & LIBERTY_MASK)
							b = 128
							if 384 < r+g+b:
								c = tohex(r,g,b)
								self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill=c)

						if self.show_legal_moves:
							if self.board.P1_legal_moves[y][x] and self.board.P2_legal_moves[y][x]:
								self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill='yellow')
							elif self.board.P1_legal_moves[y][x]:
								self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill='green')
							elif self.board.P2_legal_moves[y][x]:
								self.canvas.create_oval(x*self.wstep,y*self.hstep,x*self.wstep + self.wstep,y*self.hstep + self.hstep,fill='red')
						#self.canvas.create_text(x*self.wstep+self.wstep/2,y*self.hstep+self.hstep/2,text=str(y)+','+str(x),fill='red')

			moves_forecast = []#self.mcts.get_moves_forecast()

			ctr = 1
			for m, p in moves_forecast:
				color = 'black'
				if p == P2_ID:
					color = 'white'
				
				if m != None:
					row = m[0]
					col = m[1]
					self.canvas.create_oval(col*self.wstep+self.wstep/2-self.wstep/4,row*self.hstep+self.hstep/2-self.hstep/4,col*self.wstep+self.wstep/2+self.wstep/4,row*self.hstep+self.hstep/2+self.hstep/4,fill=color)
					self.canvas.create_text(col*self.wstep+self.wstep/2,row*self.hstep+self.hstep/2,text=str(ctr), fill='magenta')
				ctr += 1

			for fm in self.mcts_favorite_moves:
				if fm[1] != None:
					y = fm[1][0]
					x = fm[1][1]
					self.canvas.create_oval(x*self.wstep+self.wstep/2-self.wstep/8,y*self.hstep+self.hstep/2-self.hstep/8,x*self.wstep+self.wstep/2+self.wstep/8,y*self.hstep+self.hstep/2+self.hstep/8,fill='red')
				
			mcts_legal_moves = []
			if self.cell_score_source % 3 == 1:
				#print ('Cell Scores Sourced From Neural Network')
				#mcts_legal_moves = self.mcts.get_legal_moves()
				
				player = 1
				if self.turn % 2 == 0:
					player = -1
				
				sgf = BoardUtils().moves_to_sgf(self.moves_stack)
				if len(sgf) == 0:
					sgf = BoardUtils().moves_to_sgf([(0,0)])

				if self.model.input_shape[-1] == 2:
					mcts_legal_moves = calculate_preferences_for_themachine(self.model, self.board, player)
				elif self.model.input_shape[-1] == 17:
					features, junk1, junk2 = BoardUtils().sgf_to_multichannel_dataset(sgf, 19, 16)
					features = features[-1].reshape(1,19,19,17)
					mcts_legal_moves = calculate_preferences_for_themachine_using_features(features, self.model, self.board, player)
				mcts_legal_moves = weight_sensible_moves(mcts_legal_moves, self.board, player, 20, False)
				
			elif self.cell_score_source % 3 == 2:
				#print ('Cell Scores Sourced From MCTS')
				#mcts_legal_moves = BoardUtils().populate_value_map(self.value_network, self.board, self.mcts.get_legal_moves(), self.mcts.get_next_player())
				#mcts_legal_moves = calculate_preferences_for_themachine(self.territory_network, self.board, self.mcts.get_next_player())
				territory, score = BoardUtils().tromp_taylor_scoring(self.board)
				
				mcts_legal_moves = []
				for r in range(self.board.edge):
					for c in range(self.board.edge):
						mcts_legal_moves.append((territory[r][c],(r,c)))

			weights = []
			moves = []
			
			if mcts_legal_moves != None:
				for m in mcts_legal_moves:
					pass#print (m)

			if mcts_legal_moves != None and len(mcts_legal_moves):
				for m in mcts_legal_moves:
					weights.append(m[0])
					moves.append(m[1])

				minimum = min(weights)
				domain = max(weights) - min(weights)
				median = domain / 2.0
				
				for i in range(len(weights)):
					this_weight = weights[i]
					this_weight -= minimum
					this_weight /= float(domain+1e-10)
					this_weight *= 100
					this_weight = int(this_weight)
					dist_from_median = abs(this_weight - 50)*2
					
					#dist_from_median = int(100*abs(float(median - this_weight)/(domain/2.0)))
					
					#print (dist_from_median)
					
					r,g,b = 128,128,128
					if this_weight < 50:	# then we're going to shift more toward blue
						r -= dist_from_median
						g -= dist_from_median
						b += dist_from_median
					else:
						r += dist_from_median
						g -= dist_from_median
						b -= dist_from_median

					#print ('Color Inputs:',r,g,b)
					color = tohex(r,g,b)
					if this_weight < 10:
						color = 'beige'
					
					move = moves[i]
					if move != None:
						if self.board.board[move[0]][move[1]] == 0:
							row = move[0]
							col = move[1]
							
							self.canvas.create_oval(col*self.wstep+self.wstep/2-self.wstep/4,row*self.hstep+self.hstep/2-self.hstep/4,col*self.wstep+self.wstep/2+self.wstep/4,row*self.hstep+self.hstep/2+self.hstep/4,fill=color)
							self.canvas.create_text(col*self.wstep+self.wstep/2,row*self.hstep+self.hstep/2,text=str(this_weight))

			# display moves with errors
			for row, col in self.moves_debug:
				self.canvas.create_oval(col*self.wstep+self.wstep/2-self.wstep/4,row*self.hstep+self.hstep/2-self.hstep/4,col*self.wstep+self.wstep/2+self.wstep/4,row*self.hstep+self.hstep/2+self.hstep/4,fill='magenta')

			self.canvas.update_idletasks()

		def computer_move(self):
			played = False
			accumulator = max(0.8-float(self.turn)/100, 0.10)
			if self.turn & 1 and player_1 != 'human':
				position = None
				if player_1 == 'srandom':
					position = wwsrgd(self.board, 1)
				if player_1 == 'themachine':
					if self.black_model.input_shape[-1] == 2:
						position = wwtmd(self.black_model, self.board, 1, accumulator)
					elif self.black_model.input_shape[-1] == 17:
						position = wwtmd_from_moves(self.features, self.black_model, self.board, 1, accumulator)
					#position = self.mcts.generate_move(self.think_time, 1)
				self.moves_stack.append(position)
				errors = self.board.do_move(position,1)
				#self.mcts_favorite_moves = self.mcts.get_favorite_moves(5)
				#self.mcts.register_move(position,1)
				if position != None:
					self.move_numbers[position[0]][position[1]] = self.move_counter
					played = True
				self.move_counter += 1
				if errors == 0:
					if self.handicap:
						self.handicap -= 1
						print (self.handicap,'handicaps left')
					else:
						self.add_board_to_deep_features()
						self.turn += 1
					print ('Player 1 has moved to',position,'using accumulator',accumulator)
				else:
					print ('There was an error in computer_move() with Black',position)
					self.board.assemble_displays()
					self.moves_debug.append(position)

			elif self.turn & 1 == 0 and player_2 != 'human':
				position = None
				if player_2 == 'srandom':
					position = wwsrgd(self.board, -1)
				if player_2 == 'themachine':
					if self.white_model.input_shape[-1] == 2:
						position = wwtmd(self.white_model, self.board, -1, accumulator)
					elif self.white_model.input_shape[-1] == 17:
						position = wwtmd_from_moves(self.features, self.white_model, self.board, -1, accumulator)
					#position = self.mcts.generate_move(self.think_time, 1)
				self.moves_stack.append(position)
				errors = self.board.do_move(position,-1)
				#self.mcts_favorite_moves = self.mcts.get_favorite_moves(5)
				#self.mcts.register_move(position,-1)
				if position != None:
					self.move_numbers[position[0]][position[1]] = self.move_counter
					played = True
				self.move_counter += 1
				if errors == 0:
					self.add_board_to_deep_features()
					self.turn += 1
					print ('Player 2 has moved to',position,'using accumulator',accumulator)
				else:
					print ('There was an error in computer_move() with White',position)
					self.board.assemble_displays()
					self.moves_debug.append(position)

			P1_score = 0
			P2_score = 0

			#factor this score thing into the board class - it's kind of a clunky thing to have here
			for y in range(N):
				for x in range(N):
					P1_score += self.board.P1_legal_moves[y][x] + (1 if self.board.board[y][x] == P1_ID else 0)
					P2_score += self.board.P2_legal_moves[y][x] + (1 if self.board.board[y][x] == P2_ID else 0)
			
			territory, tromp_taylor_score = BoardUtils().tromp_taylor_scoring(self.board)
			
			player = 1
			if self.turn % 1 == 1:
				player = -1
			
			#value_estimate = calculate_value_estimate(self.value_network, self.board, player)

			print ('Player 1:',P1_score,'Player 2:',P2_score,'Tromp-Taylor:',tromp_taylor_score)#,'Value Estimate:',value_estimate)
			
			self.redraw_and_clear()
			time.sleep(0.1)
			if played:
				self.computer_move()
				
		def keyboard(self, event):
			if event.char.isdigit():
				if int(event.char) == 0:
					self.think_time *= 10
				else:
					self.think_time = int(event.char)
				print ('think time:',self.think_time)
			if event.keysym == 'c':
				self.cell_score_source += 1
			if event.keysym == 'e':
				self.show_eyes ^= 1
			if event.keysym == 'l':
				self.show_liberties ^= 1
			if event.keysym == 'm':
				self.show_legal_moves ^= 1
			if event.keysym == 'p':
				if player_1 == 'human':
					self.mcts.register_move(None,1)
				elif player_2 == 'human':
					self.mcts.register_move(None,-1)

				self.turn += 1

				self.computer_move()

			if event.keysym == 's':
				self.setting_starpoints ^= 1
				print ('Setting Starpoints:',self.setting_starpoints)

			if event.keysym == 'r':
				if len(self.games_stack) == 0:
					print ('Error: No Games in Games Stack')
				else:
					print ('Selecting random game')
					
					selection = int(npr()*len(self.games_stack))
					
					print ('Selected game',selection)
					
					moves_stack = BoardUtils().sgf_to_moves(self.games_stack[selection])

					self.turn = 1
					self.move_counter = 1
					self.move_numbers = np.zeros(N**2,dtype=np.intc).reshape(N,N)
					
					self.board_stack = []
					self.board_pointer = 0
					
					player = 1
					self.board = Board(N)
					for m in moves_stack[:2*int(npr()*(len(moves_stack)/2))]:
						self.board_stack.append(deepcopy(self.board))
						self.board.do_move(m, player)

						if m != None:
							self.move_numbers[m[0]][m[1]] = self.move_counter

						self.move_counter += 1
						self.turn += 1
						player *= -1

			if event.keysym == 'b' or event.keysym == 'w':
				if event.keysym == 'b':
					if self.edit_mode == EDIT_BLACK:	self.edit_mode = 0
					else:								self.edit_mode = EDIT_BLACK

				if event.keysym == 'w':
					if self.edit_mode == EDIT_WHITE:	self.edit_mode = 0
					else:								self.edit_mode = EDIT_WHITE

				if self.edit_mode == EDIT_BLACK:		print ('Edit Mode: Black')
				elif self.edit_mode == EDIT_WHITE:		print ('Edit Mode: White')
				else:
					print ('Edit Mode: Off')
					if self.turn & 1:
						print ('Black\'s Turn')
					else:
						print ('White\'s Turn')

			if event.keysym == 'n':
				print ('Resetting Game')
				
				moves_stack = []

				self.turn = 1
				self.move_counter = 1
				self.move_numbers = np.zeros(N**2,dtype=np.intc).reshape(N,N)
				
				self.board_stack = []
				self.board_pointer = 0
				
				player = 1
				self.board = Board(N)

			if event.keysym == 'Right':
				if len(self.board_stack):
					self.board_pointer += 1
					if len(self.board_stack) <= self.board_pointer:
						self.board_pointer = len(self.board_stack)-1
					
					self.board = self.board_stack[self.board_pointer]

			if event.keysym == 'Left':
				if len(self.board_stack):
					self.board_pointer -= 1
					if self.board_pointer < 0:
						self.board_pointer = 0
					
					self.board = self.board_stack[self.board_pointer]

			if event.keysym == 'Up':
				if len(self.board_stack):
					self.board_pointer += 10
					if len(self.board_stack) <= self.board_pointer:
						self.board_pointer = len(self.board_stack)-1
					
					self.board = self.board_stack[self.board_pointer]

			if event.keysym == 'Down':
				if len(self.board_stack):
					self.board_pointer -= 10
					if self.board_pointer < 0:
						self.board_pointer = 0
					
					self.board = self.board_stack[self.board_pointer]

				#self.mcts.undo_move()
				#self.board = self.mcts.get_mcts_board()
				#print (self.board, self.mcts.root.current_board)

			if event.keysym == 'Up':
				pass

			if event.keysym == 'Down':
				pass

			self.redraw_and_clear()

		def mouse_update(self, event):
			self.MouseX = event.x
			self.MouseY = event.y

		def add_board_to_deep_features(self):
			
			channels = 16
			
			self.board_history.append(copy(self.board.board))

			# this channel format is to mimic the deepmind paper
			self.features = np.zeros(N*N*(channels+1), dtype=np.intc).reshape(1, N, N, channels+1)
			
			for chan in range((channels>>1)):
				# if we don't have enough history, then leave it and it'll show up as zeros
				if chan < len(self.board_history):
					for r in range(N):
						for c in range(N):
							if 		self.board_history[-(1+chan)][r][c] == P1_ID:
								self.features[0][r][c][2*chan + 0] = 1
							elif 	self.board_history[-(1+chan)][r][c] == P2_ID:
								self.features[0][r][c][2*chan + 1] = 1

			player = P1_ID
			if self.turn & 1:
				player = P2_ID

			if player == P1_ID:
				pass # because we were just going to fill it with zeros, but it's already zeros
			elif player == P2_ID:
				for r in range(N):
					for c in range(N):
						self.features[0][r][c][channels] = 1 # note that 'channels' == the 'channels+1'th channel
			'''

			# make sure to always use P1_ID for the player here because the channel orientation is fixed in the deep feature stack
			new_features = BoardUtils().position_to_state(self.board, P1_ID)
			
			# shift the existing moves (in kind of a stupid way)
			for r in range(N):
				for c in range(N):
					for p in range(14):
						self.features[0][r][c][15-p] = self.features[0][r][c][13-p]

					self.features[0][r][c][0] = new_features[r][c][0]
					self.features[0][r][c][1] = new_features[r][c][1]

					if self.turn & 1 == 0:
						self.features[0][r][c][16] = 1
					else:
						self.features[0][r][c][16] = 0
			'''
		def mouse_press(self, event):
			self.left_mouse_pressed = 1

			self.MouseX = event.x
			self.MouseY = event.y

			x = int((self.MouseX) / self.wstep)
			y = int((self.MouseY) / self.hstep)

			if self.edit_mode:
				if self.edit_mode == EDIT_BLACK:	self.board.do_move((y,x),1)
				elif self.edit_mode == EDIT_WHITE:	self.board.do_move((y,x),-1)
			else:
				if self.setting_starpoints:
					self.star_points[y][x] ^= 1
				else:
					if self.turn & 1 and player_1 == 'human':
						errors = self.board.do_move((y,x),1)
						if errors == 0:
							if self.handicap:
								self.handicap -= 1
								print (self.handicap,'handicaps left')
							else:
								self.add_board_to_deep_features()
								self.turn += 1

							self.mcts_favorite_moves = []
							#self.mcts.register_move((y,x),1)
							self.move_numbers[y][x] = self.move_counter
							self.move_counter += 1
							self.moves_stack.append((y,x))
						else:
							print ('There was an error with the human player')
					elif self.turn & 1 == 0 and player_2 == 'human':
						errors = self.board.do_move((y,x),-1)
						if errors == 0:
							self.add_board_to_deep_features()
							self.turn += 1
							self.mcts_favorite_moves = []
							#self.mcts.register_move((y,x),-1)
							self.move_numbers[y][x] = self.move_counter
							self.move_counter += 1
							self.moves_stack.append((y,x))
						else:
							print ('There was an error with the human player')

			self.redraw_and_clear()
			self.computer_move()
			self.redraw_and_clear()

		def mouse_release(self, event):
			self.left_mouse_pressed = 0
			self.redraw_and_clear()

		def __init__(self):
			self.turn = 1
			self.board = Board(N)
			self.handicap = handicap

			self.games_stack = []#open('go_kgs_6d+_games','r').read().split('\n')[:-1]
			self.board_history = []

			self.features = np.zeros(19*19*17).reshape(1,19,19,17) # that's 8 positions of 19x19 for each player, black/white, black/white, etc.

			self.moves_debug = []
			self.moves_stack = []
			self.moves_pointer = 0
			self.board_stack = []
			self.board_pointer = 0
			self.edit_mode = 0

			self.think_time = THINKTIME

			self.move_counter = 1
			self.move_numbers = np.zeros(N**2,dtype=np.intc).reshape(N,N)
			
			#TODO - factor rotational symmetry into dataset (4x increase in data points)
			#TODO - add channels to represent the last 8 moves for each player
			#TODO - add a channel to say who's turn it is
			
			preferred_network = 'policy network - kgs 6ds - 0.0996 4.35'
			
			self.model = load_model(preferred_network)
			self.value_network = None#load_model('latest_value')
			self.territory_network = build_model(N)#load_model('territory network 10')

			self.black_model = load_model('policy network - kgs 6ds - 0.0517 4.87')
			self.white_model = load_model('policy network - kgs 6ds - 0.2525 3.65')

			self.mcts = None#go_mcts(self.model, None, self.model, deepcopy(self.board), None, 1, SIMULATED_ANNEALING, True, 1, 0)
			self.mcts_favorite_moves = []
			
			self.show_eyes = 0
			self.show_liberties = 0
			self.show_legal_moves = 0
			self.cell_score_source = 1
			self.setting_starpoints = False
			self.star_points = np.zeros(N**2,dtype=np.intc).reshape(N,N)

			self.wstep = WIDTH / float(self.board.N)
			self.hstep = HEIGHT / float(self.board.N)
			self.swidth = min(self.wstep,self.hstep)

			self.left_mouse_pressed = 0

			self.MouseX = 0
			self.MouseY = 0

			Frame.__init__(self)
			self.master.title('Go!')
			self.master.rowconfigure(0,weight=1)
			self.master.columnconfigure(0,weight=1)
			self.grid()

			self.canvas=Canvas(self,width=WIDTH, height=HEIGHT, bg='beige')
			self.canvas.grid(row=0,column=0)

			self.bind_all('<KeyPress>', self.keyboard)
			self.bind_all('<Motion>', self.mouse_update)
			self.bind_all('<ButtonPress-1>', self.mouse_press)
			self.bind_all('<ButtonRelease-1>', self.mouse_release)

			self.computer_move()
			self.redraw_and_clear()
	display().mainloop()






