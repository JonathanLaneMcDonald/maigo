
from board import Board

from tkinter import *
from copy import deepcopy



WIDTH = 800
HEIGHT = 800

class display(Frame):

	def redraw_and_clear(self):

		self.canvas.delete('all')

		for i in range(1, self.edge + 1):
			self.canvas.create_line(i * self.wstep - self.wstep / 2, 0, i * self.wstep - self.wstep / 2, HEIGHT)
			self.canvas.create_line(0, i * self.hstep - self.hstep / 2, WIDTH, i * self.hstep - self.hstep / 2)

		if len(self.frame_stack) and self.frame < len(self.frame_stack):
			for y in range(self.edge):
				for x in range(self.edge):
					if self.frame_stack[self.frame][0].is_black(y * self.edge + x):
						self.canvas.create_oval(x * self.wstep, y * self.hstep, x * self.wstep + self.wstep, y * self.hstep + self.hstep, fill='black')
					elif self.frame_stack[self.frame][0].is_white(y * self.edge + x):
						self.canvas.create_oval(x * self.wstep, y * self.hstep, x * self.wstep + self.wstep, y * self.hstep + self.hstep, fill='white')

			if self.frame_stack[self.frame][1]:
				y, x = self.frame_stack[self.frame][1]
				self.canvas.create_oval(x * self.wstep + self.wstep // 4, y * self.hstep + self.hstep // 4, x * self.wstep + self.wstep - self.wstep // 4, y * self.hstep + self.hstep - self.hstep // 4, fill='red')

		text = str(self.game) + ':' + str(self.frame)
		self.canvas.create_text(40, 20, text=text, fill='red')

		self.canvas.update_idletasks()

	def load_games(self):
		self.game_stack = [x for x in open('games with errors', 'r').read().split('\n') if len(x)]
		print(len(self.game_stack), 'games loaded')
		self.game = 0

	def error_driver(self):

		if len(self.game_stack) and self.game < len(self.game_stack):
			pass
		else:
			return

		print('********************************************************************************')

		self.frame = 0
		self.frame_stack = []

		game = self.game_stack[self.game].split(',')

		player = 1
		board = Board(self.edge)
		for j in game:
			move = sgf_to_coords(j)

			linearized_move = None
			if move != None:
				linearized_move = self.edge*move[0] + move[1]
			if not board.place_stone(linearized_move, player):
				print('error:',player, move, len(self.frame_stack))
			player *= -1

			self.frame_stack.append((deepcopy(board), move))
			self.frame = len(self.frame_stack) - 1

		self.redraw_and_clear()

	def keyboard(self, event):
		if event.keysym == 'Prior':
			if self.game + 10 < len(self.game_stack):
				self.game += 10
				self.error_driver()

		if event.keysym == 'Next':
			if 0 < self.game - 10:
				self.game -= 10
				self.error_driver()

		if event.keysym == 'Up':
			if self.game + 1 < len(self.game_stack):
				self.game += 1
				self.error_driver()

		if event.keysym == 'Down':
			if 0 < self.game:
				self.game -= 1
				self.error_driver()

		if event.keysym == 'Right':
			if self.frame + 1 < len(self.frame_stack):
				self.frame += 1

		if event.keysym == 'Left':
			if 0 < self.frame:
				self.frame -= 1

		self.redraw_and_clear()

	def __init__(self):
		self.edge = 19

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

		self.redraw_and_clear()

		self.load_games()
		self.error_driver()

display().mainloop()
