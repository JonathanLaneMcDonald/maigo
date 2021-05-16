
from model import build_agz_model

import numpy as np

class ModelManager:
	def __init__(self, resnet_bc, tasks_to_model_queue, results_to_mcts_queue_array):

		blocks, channels = resnet_bc
		self.model = build_agz_model(blocks, channels, (9, 9, 8))

		self.tasks_to_model = tasks_to_model_queue
		self.results_to_mcts = results_to_mcts_queue_array

		self.monitor()

	def monitor(self):
		print("ModelManager is entering monitoring loop")
		while True:

			'''
			batch_size = 512
			if len(real_games) % 10 == 0:
				model_inputs, policy_targets, value_targets = create_training_dataset_from_frames(training_frames, batch_size)
				model.fit(model_inputs, [policy_targets, value_targets], batch_size=batch_size, epochs=1, verbose=1)
				save_model(model, "b" + str(blocks) + "c" + str(channels) + "@" + str(len(real_games)) + ".h5", save_format="h5")
				save_model(model, "current go model.h5", save_format="h5")
			'''

			if not self.tasks_to_model.empty():

				originator_random_number = []
				originating_processes = []
				originating_nodes = []
				originating_parents = []
				inference_tasks = []
				while not self.tasks_to_model.empty():
					random_number, originating_process, originating_node, parent, inference_task = self.tasks_to_model.get()

					originator_random_number.append(random_number)
					originating_processes.append(originating_process)
					originating_nodes.append(originating_node)
					originating_parents.append(parent)
					inference_tasks.append(inference_task)

				#print("ModelManager pulled", len(originating_processes),"states off the queue")

				policy_targets, value_targets = self.model.predict(np.moveaxis(np.array(inference_tasks), 1, -1))

				for i in range(len(originating_processes)):
					#print("placing result on queue", i, "for node", originating_nodes[i])
					self.results_to_mcts[originating_processes[i]].put((originator_random_number[i], originating_nodes[i], originating_parents[i], policy_targets[i], value_targets[i]))
