
D = 9
R = D
C = R

# Function for print matrix
def printMatrix(arr):
	for i in range(R):
		for j in range(C):
			print(str(arr[i][j]), end =" ")
		print()

board = [
	[' ', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
	[' ', ' ', ' ', '.', '!', '?', ' ', ' ', ' '],
	[' ', ' ', ' ', '.', '?', '?', ' ', ' ', ' '],
	[' ', ' ', ' ', '.', '.', '.', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', '|', '|', '|', '|', '|', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]

import numpy

array = numpy.array([board])
print('array shape:', array.shape)
for i in range(4):
	array = numpy.rot90(array, axes=(1,2))
	print("rotated",array.shape)
	printMatrix(array[0])
	flipped = numpy.flip(array, axis=-1)
	print("flipped",flipped.shape)
	printMatrix(flipped[0])

array = numpy.moveaxis(array, 0, -1)
print("moveaxis -> ",array.shape)
