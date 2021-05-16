
from numpy.random import random

result = {-x:0 for x in range(21)}
for i in range(1000000):
	result[-int(1+20*(1-random()**(1/5)))] += 1
print('\n'.join([str(x) for x in result.items()]))









