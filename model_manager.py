
from model import build_agz_model

import numpy as np

class ModelManager:
	def __init__(self, resnet_bc, tasks_to_model_queue, results_to_mcts_queue_array, max_batch_size=128):

		blocks, channels = resnet_bc
		self.model = build_agz_model(blocks, channels, (9, 9, 8))

		self.tasks_to_model = tasks_to_model_queue
		self.results_to_mcts = results_to_mcts_queue_array
		self.max_batch_size = max_batch_size

		self.monitor()

	def monitor(self):
		print("ModelManager is entering monitoring loop")
		while True:
			if not self.tasks_to_model.empty():

				originator_random_number = []
				originating_processes = []
				originating_nodes = []
				inference_tasks = []
				while not self.tasks_to_model.empty() and len(originating_processes) < self.max_batch_size:
					random_number, originating_process, originating_node, inference_task = self.tasks_to_model.get()

					originator_random_number.append(random_number)
					originating_processes.append(originating_process)
					originating_nodes.append(originating_node)
					inference_tasks.append(inference_task)

				print("ModelManager pulled", len(originating_processes),"states off the queue")

				policy_targets, value_targets = self.model.predict(np.moveaxis(np.array(inference_tasks), 1, -1))

				for i in range(len(originating_processes)):
					self.results_to_mcts[originating_processes[i]].put((originator_random_number[i], originating_nodes[i], policy_targets[i], value_targets[i]))
