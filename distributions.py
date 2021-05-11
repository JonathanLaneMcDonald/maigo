
from numpy.random import random

result = [0]*100
for i in range(1000000):
	result[int(100*(random()**(1/3)))] += 1
print('\n'.join([str(x) + ' ' + str(result[x]) for x in range(len(result))]))









