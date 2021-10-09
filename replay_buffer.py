



class ReplayBuffer:
	"""
	1) create a directory for self-play game data, maybe call it "self-play game data"
	2) read all historical games into a buffer (RAM should be fine even with a few million games)
	3) create a new file int(time.time()) to store games generated during this session
	4) when a game string is submitted, add it to the stack in memory and to the open file, flush file buffer
	5) be able to tell listeners when the total games count == 0 % some value
	6) given a Game implementation, be able to produce a training batch of some size from the k most recent games
	"""

	def __init__(self):
		pass




