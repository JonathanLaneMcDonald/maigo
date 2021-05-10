
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization, Convolution2D, Activation, Add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply
from keras.layers import Flatten, GlobalMaxPooling2D, Concatenate
from keras.optimizers import Adam

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

def build_agz_model(blocks, filters, input_shape):

	game_input = Input(shape=input_shape)
	meta_input = Input(shape=(2))

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

	scoring_and_outcome_gap = GlobalAveragePooling2D()(value_head)
	scoring_and_outcome_gmp = GlobalMaxPooling2D()(value_head)
	scoring_and_outcome = Concatenate()([scoring_and_outcome_gap, scoring_and_outcome_gmp])

	final_score = Dense(2*input_shape[0]*input_shape[1], activation='softmax', name='score')(scoring_and_outcome)

	value = Dense(1, activation='tanh', name='value')(scoring_and_outcome)

	model = Model([game_input, meta_input], [policy, ownership, final_score, value])
	model.compile(
		loss={'policy': 'sparse_categorical_crossentropy',
			  'ownership': 'mse',
			  'score': 'sparse_categorical_crossentropy',
			  'value': 'mse'},
		loss_weights={'policy': 1.0,
					  'ownership': 0.02,
					  'score': 0.02,
					  'value': 1.0},
		optimizer=Adam(lr=0.0001)
	)
	model.summary()

	return model




