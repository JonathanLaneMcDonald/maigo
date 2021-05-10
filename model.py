
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization, Convolution2D, Activation, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply
from keras.layers import Flatten

def add_bn_relu_convolve_block(x, filters, kernel_size=(3,3)):
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Convolution2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
	return x

def add_squeeze_exite_block(x, filters):
	y = GlobalAveragePooling2D()(x)
	y = Dense(filters // 4, activation='relu', use_bias=False)(y)
	y = Dense(filters, activation='sigmoid', use_bias=False)(y)
	y = Reshape((1, 1, filters))(y)
	x = Multiply()([x, y])
	return x

def build_agz_model(blocks, filters, input_shape=(9, 9, 4)):

	game_input = Input(shape=input_shape)
	meta_input = Input(shape=(1))

	# start by projecting the input into the space we'll need for residual connections
	game_projection = Convolution2D(filters=filters, kernel_size=(5,5), padding='same')(game_input)
	meta_projection = Dense(filters)(meta_input)
	meta_projection = Reshape((1,1,filters))(meta_projection)
	x = Add()([game_projection, meta_projection])

	for _ in range(blocks):
		y = add_bn_relu_convolve_block(x, filters)
		y = add_squeeze_exite_block(y, filters)
		y = add_bn_relu_convolve_block(y, filters)
		x = Add()([x,y])

	policy_head = x
	policy_head = add_bn_relu_convolve_block(policy_head, filters, (1,1))
	policy_head = add_squeeze_exite_block(policy_head, filters)
	policy_head = add_bn_relu_convolve_block(policy_head, 1, (1,1))
	policy = Flatten()(policy_head)
	policy = Activation('softmax', name='policy')(policy)

	value_head = x
	value_head = add_bn_relu_convolve_block(value_head, filters, (1,1))

	ownership = add_bn_relu_convolve_block(value_head, 1, (1,1))
	ownership = Reshape((input_shape[0], input_shape[1]))(ownership)
	ownership = Activation('tanh', name='ownership')(ownership)

	value = GlobalAveragePooling2D()(value_head)
	value = Dense(filters, 'relu')(value)
	value = Dense(1, activation='tanh', name='value')(value)

	model = Model([game_input, meta_input], [policy, ownership, value])
	model.compile(
		loss={'policy': 'categorical_crossentropy',
			  'ownership': 'mse',
			  'value': 'mse'},
		loss_weights={'policy': 1.0,
					  'ownership': 0.02,
					  'value': 1.0},
		optimizer='adam'
	)
	model.summary()

	return model




