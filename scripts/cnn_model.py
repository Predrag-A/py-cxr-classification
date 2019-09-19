from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,\
    AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.models import Model


def identity_block(x, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    x_shortcut = x

    # First component of main path
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def convolutional_block(x, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    x_shortcut = x

    # MAIN PATH
    # First component of main path
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    # SHORTCUT PATH
    x_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='valid'
                        , name=conv_name_base + '10')(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def build_model(img_height, img_width, classes=2):

    input_shape = (img_height, img_width, 1)

    if K.image_data_format() == "channels_first":
        input_shape = (1, img_height, img_width)
    K.set_learning_phase(1)

    x_input = Input(input_shape)

    # Stage 1
    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)

    # MAX POOLING
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, f=3, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(x, f=3, filters=[64, 64, 256], stage=2, block='c')

    # MAX POOLING
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Stage 3
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')

    # Stage 4
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')

    # Stage 5
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='d')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='e')

    # AVERAGE POOLING
    x = AveragePooling2D(pool_size=(7,7), padding='same')(x)

    # Output Layer
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model
