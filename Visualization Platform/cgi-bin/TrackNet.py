from keras.models import *
from keras.layers import *
# from group_norm import GroupNormalization

def TrackNet( input_height, input_width ): # input_height = 360, input_width = 640

	imgs_input = Input(shape=(9,input_height,input_width))

	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer14
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer18
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer21
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	#layer24
	x =  Conv2D( 1 , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
#	x = ( BatchNormalization())(x)
# 	x = GroupNormalization(groups=4,axis=1)(x)

	o_shape = Model(imgs_input , x ).output_shape
	#print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#layer24 output shape: 1, 360, 640

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	#reshape the size to (360, 640)
	output = (Reshape((  OutputHeight  ,  OutputWidth   )))(x)

	


	model = Model( imgs_input , output)
#	model input unit:9*360*640, output unit:360*640
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#show model's details
	#model.summary()

	return model




