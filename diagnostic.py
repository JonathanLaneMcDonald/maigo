
from keras.models import load_model
from board import Board
from model import build_agz_model

from tkinter import *
from copy import deepcopy

from datasets import *
from datasets import _char_to_smallnum_

WIDTH = 450
HEIGHT = 450

def color_from_tanh(tanh):
	h = '0123456789ABCDEF'
	loss = int(abs(tanh) * 255)
	color = h[(255-loss)//16]+h[(255-loss)%16]
	if tanh < 0:
		return '#' + color*2 + 'FF'
	else:
		return '#FF' + color*2

class display(Frame):

	def redraw_and_clear(self):

		self.canvas.delete('all')

		# draw the board in the background
		if len(self.frame_stack) and self.frame < len(self.frame_stack):
			player_to_move = 1 if self.frame % 2 == 0 else -1
			model_inputs = game_state_to_model_inputs(self.frame_stack[self.frame][0], player_to_move)
			model_inputs = np.array([model_inputs])
			model_inputs = np.moveaxis(model_inputs, 1, -1)
			policy, value = self.model.predict(model_inputs)
			model_inputs = np.moveaxis(model_inputs, -1, 1)[0]
			policy /= np.amax(policy)
			policy = np.reshape(np.array(policy[0]), (self.edge, self.edge))
			value = value[0]

			print('player to move:', player_to_move, 'value of position:', value)

			display_matrix = np.zeros((self.edge, self.edge))
			if self.channel_selection == 1:		display_matrix = model_inputs[0]
			if self.channel_selection == 2:		display_matrix = model_inputs[1]
			if self.channel_selection == 3:		display_matrix = model_inputs[2]
			if self.channel_selection == 4:		display_matrix = model_inputs[3]
			if self.channel_selection == 5:		display_matrix = model_inputs[4]
			if self.channel_selection == 6:		display_matrix = model_inputs[5]
			if self.channel_selection == 7:		display_matrix = model_inputs[6]
			if self.channel_selection == 8:		display_matrix = model_inputs[7]
			if self.channel_selection == 9:		display_matrix = policy

			for y in range(self.edge):
				for x in range(self.edge):
					self.canvas.create_rectangle(x * self.wstep, y * self.hstep, (x+1) * self.wstep, (y+1) * self.hstep, fill=color_from_tanh(display_matrix[y][x]), outline="")

		# draw the lines on the board
		for i in range(1, self.edge + 1):
			self.canvas.create_line(i * self.wstep - self.wstep / 2, 0, i * self.wstep - self.wstep / 2, HEIGHT)
			self.canvas.create_line(0, i * self.hstep - self.hstep / 2, WIDTH, i * self.hstep - self.hstep / 2)

		# draw the stones on the board
		if len(self.frame_stack) and self.frame < len(self.frame_stack):
			for y in range(self.edge):
				for x in range(self.edge):
					if self.frame_stack[self.frame][0].is_black(y * self.edge + x):
						self.canvas.create_oval(x * self.wstep + self.wstep // 7, y * self.hstep + self.hstep // 7, x * self.wstep + self.wstep - self.wstep // 7, y * self.hstep + self.hstep - self.hstep // 7, fill='black')
					elif self.frame_stack[self.frame][0].is_white(y * self.edge + x):
						self.canvas.create_oval(x * self.wstep + self.wstep // 7, y * self.hstep + self.hstep // 7, x * self.wstep + self.wstep - self.wstep // 7, y * self.hstep + self.hstep - self.hstep // 7, fill='white')

			if self.frame_stack[self.frame][1]:
				move = self.frame_stack[self.frame][1]
				y, x = move // self.edge, move % self.edge
				self.canvas.create_oval(x * self.wstep + self.wstep // 4, y * self.hstep + self.hstep // 4, x * self.wstep + self.wstep - self.wstep // 4, y * self.hstep + self.hstep - self.hstep // 4, fill='red')

		text = str(self.game) + ':' + str(self.frame)
		self.canvas.create_text(40, 20, text=text, fill='red')

		self.canvas.update_idletasks()

	def load_games(self):
		self.game_stack = [x for x in open('10k vanilla mcts games', 'r').read().split('\n') if len(x)]
		print(len(self.game_stack), 'games loaded')
		self.game = 0

	def replay_driver(self):

		if len(self.game_stack) and self.game < len(self.game_stack):
			pass
		else:
			return

		print('********************************************************************************')

		self.frame = 0
		self.frame_stack = []

		game = parse_game_record(self.game_stack[self.game])['moves']

		player = 1
		board = Board(self.edge)
		self.frame_stack.append((deepcopy(board), -1))
		self.frame = len(self.frame_stack) - 1
		for j in game:
			move = _char_to_smallnum_(j)

			if not board.place_stone(move, player):
				print('error:',player, move, len(self.frame_stack))
			player *= -1

			self.frame_stack.append((deepcopy(board), move))
			self.frame = len(self.frame_stack) - 1

		self.frame = 0

		self.redraw_and_clear()

	def keyboard(self, event):
		if event.keysym == 'Prior':
			if self.game + 10 < len(self.game_stack):
				self.game += 10
				self.replay_driver()

		if event.keysym == 'Next':
			if 0 < self.game - 10:
				self.game -= 10
				self.replay_driver()

		if event.keysym == 'Up':
			if self.game + 1 < len(self.game_stack):
				self.game += 1
				self.replay_driver()

		if event.keysym == 'Down':
			if 0 < self.game:
				self.game -= 1
				self.replay_driver()

		if event.keysym == 'Right':
			if self.frame + 1 < len(self.frame_stack):
				self.frame += 1

		if event.keysym == 'Left':
			if 0 < self.frame:
				self.frame -= 1

		if event.keysym.isdigit():
			self.channel_selection = int(event.keysym)

		self.redraw_and_clear()

	def __init__(self):
		self.edge = 9

		self.board = Board(self.edge)

		self.game = 0
		self.game_stack = []

		self.frame = 0
		self.frame_stack = []

		self.wstep = WIDTH / float(self.edge)
		self.hstep = HEIGHT / float(self.edge)

		Frame.__init__(self)
		self.master.title('Go!')
		self.master.rowconfigure(0, weight=1)
		self.master.columnconfigure(0, weight=1)
		self.grid()

		self.canvas = Canvas(self, width=WIDTH, height=HEIGHT, bg='beige')
		self.canvas.grid(row=0, column=0)

		self.bind_all('<KeyPress>', self.keyboard)

		#self.model = load_model('crappy go model b4c32 4500.h5')
		self.model = build_agz_model(4, 32, (9, 9, 8))
		self.channel_selection = 0

		self.redraw_and_clear()

		self.load_games()
		self.replay_driver()

display().mainloop()
