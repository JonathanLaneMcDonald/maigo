
import os
import numpy as np
from numpy.random import random
from datasets import create_training_dataset_from_frames

class ReplayBuffer:

	update_interval = 100 # every k games, we create a training batch and send it to the model
	batch_size = 2048 # that update has this many samples
	checkpoint_interval = 10_000 # save a checkpoint every k games

	historical_move_window = 10_000_000
	distribution = lambda x: int(min(ReplayBuffer.historical_move_window, x) * (random() ** (1 / 2)))

	def __init__(self, completed_to_buffer, training_data_to_model):

		real_game_filename = '9x9 real games'
		training_frames_filename = '9x9 training frames'

		self.real_game_buffer = []
		if os.path.exists(real_game_filename):
			self.real_game_buffer = [x for x in open(real_game_filename, 'r').read().split('\n') if len(x)]
			print(len(self.real_game_buffer), 'games loaded from file')
		self.real_game_writer = open(real_game_filename, 'a')

		self.training_frame_buffer = []
		if os.path.exists(training_frames_filename):
			self.training_frame_buffer = [x for x in open(training_frames_filename, 'r').read().split('\n') if len(x)]
			print(len(self.training_frame_buffer), 'training frames loaded from file')
		self.training_frame_writer = open(training_frames_filename, 'a')

		self.completed_games_to_buffer = completed_to_buffer
		self.training_data_to_model = training_data_to_model

		self.monitor()

	def monitor(self):
		print("ReplayBuffer is entering monitoring loop")
		while True:
			if not self.completed_games_to_buffer.empty():
				real_game, training_frames = self.completed_games_to_buffer.get()

				self.real_game_buffer.append(real_game)
				self.real_game_writer.write(real_game)

				self.training_frame_buffer += training_frames
				for frame in training_frames:
					self.training_frame_writer.write(frame + '\n')

				if len(self.real_game_buffer) % ReplayBuffer.update_interval == 0:
					model_inputs, policy_targets, value_targets = create_training_dataset_from_frames(
						self.training_frame_buffer,
						ReplayBuffer.batch_size,
						ReplayBuffer.historical_move_window
					)

					save_checkpoint = len(self.real_game_buffer) % ReplayBuffer.checkpoint_interval == 0
					self.training_data_to_model.put((
						len(self.real_game_buffer),
						model_inputs,
						policy_targets,
						value_targets,
						save_checkpoint
					))




