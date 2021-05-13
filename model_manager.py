
import time
from multiprocessing import Process, Queue

import numpy as np

class ModelManager:
	def __init__(self, model, batch_size, max_cooldown=0.001):
		self.model = model
		self.batch_size = batch_size
		self.last_batch_time = time.time()
		self.max_cooldown = max_cooldown
		self.queue = Queue()

		self.process = Process(target=self.monitor, args=())
		self.process.daemon = True
		self.process.start()

	def get_queue(self):
		return self.queue

	def monitor(self):
		while True:
			if self.queue.qsize() >= self.batch_size or (not self.queue.empty() and time.time()-self.last_batch_time >= self.max_cooldown):
				self.last_batch_time = time.time()

				current_batch_size = min(self.batch_size, self.queue.qsize())

				originator_info = {} # expecting {'model inputs', 'originator queue', 'originating node'}
				batch_model_inputs = np.zeros((current_batch_size, 8, 9, 9), dtype=np.intc)
				for i in range(current_batch_size):
					originator_info[i] = self.queue.get()
					batch_model_inputs[i] = originator_info['model inputs']

				policy_targets, value_targets = self.model.predict(np.moveaxis(batch_model_inputs, 1, -1))

				for i in range(current_batch_size):
					originator_info[i]['originator queue'].put(
						{
							'policy': policy_targets[i],
							'value': value_targets[i],
							'node': originator_info[i]['originating node']
						}
					)

