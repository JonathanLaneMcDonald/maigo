
from keras.models import Model
from keras.layers import Input, BatchNormalization, Convolution2D, Activation, Add, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.initializers import RandomNormal


def add_bn_relu_convolve_block(x, filters, kernel_size=(3, 3)):
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Convolution2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=RandomNormal(stddev=0.01))(x)
	return x


def build_agz_model(blocks, filters, input_shape):

	inputs = Input(shape=input_shape)

	x = add_bn_relu_convolve_block(inputs, filters)

	for _ in range(blocks):
		y = add_bn_relu_convolve_block(x, filters)
		y = add_bn_relu_convolve_block(y, filters)
		x = Add()([x, y])

	policy = x
	policy = add_bn_relu_convolve_block(policy, 2, (1, 1))
	policy = Flatten()(policy)
	policy = Dense(input_shape[0]*input_shape[1]+1, activation='softmax', name='policy')(policy)

	value = x
	value = add_bn_relu_convolve_block(value, 1, (1, 1))
	value = Flatten()(value)
	value = Dense(2*filters, activation='relu')(value)
	value = Dense(1, activation='tanh', name='value')(value)

	model = Model(inputs, [policy, value])
	model.compile(
		loss={'policy': 'sparse_categorical_crossentropy',
			  'value': 'mse'},
		loss_weights={'policy': 1.0,
					  'value': 1.0},
		optimizer=Adam(learning_rate=0.0002),
		metrics=["accuracy"]
	)
	model.summary()

	return model


def build_rollout_policy(blocks, filters, input_shape):

	inputs = Input(shape=input_shape)

	x = inputs
	for block in range(blocks):
		x = Convolution2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
		x = BatchNormalization()(x)

	x = Convolution2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(x)

	x = Flatten()(x)

	outputs = Dense(input_shape[0]*input_shape[1]+1, activation='softmax')(x)

	model = Model(inputs, outputs)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
	model.summary()

	return model




