from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,\
    AveragePooling2D, MaxPooling2D, Concatenate
from keras import backend as K
from keras.models import Model


def identity_block(data, kernel_size, filters, stage, block):
    """
    Adds an identity block to the input model
    An identity block contains a residual connection to the begining of the block    
    :param data: Functional API model
    :param kernel_size: Kernel size of the second conv operation
    :param filters: Dimensionality of the conv operation outputs
    :param stage: Id of the stage
    :param block: Id of the block
    :return: Model with added layers
    """

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    shortcut = data

    # First component of main path
    data = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2a')(data)
    data = Activation('relu')(data)

    # Second component of main path
    data = Conv2D(filters=f2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',
                  name=conv_name_base + '2b')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2b')(data)
    data = Activation('relu')(data)

    # Third component of main path
    data = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2c')(data)

    # Final step: Add shortcut value to main path, and pass it through a ReLu activation
    data = Add()([data, shortcut])
    data = Activation('relu')(data)

    return data


def convolutional_block(data, kernel_size, filters, stage, block, s=2):
    """
    Adds a convolutional block to the input model
    A convolutional block contains a residual connection to the begining of the block
    which has an added convolution layer to adjust the dimensionality of the connection  
    :param data: Functional API model
    :param kernel_size: Kernel size of the second conv operation
    :param filters: Dimensionality of the conv operation outputs
    :param stage: Id of the stage
    :param block: Id of the block
    :param s: Stride of the convolution operation applied to the residual connection
    :return: Model with added layers
    """

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    shortcut = data

    # Main path
    # First component of main path
    data = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2a')(data)
    data = Activation('relu')(data)

    # Second component of main path
    data = Conv2D(filters=f2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',
                  name=conv_name_base + '2b')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2b')(data)
    data = Activation('relu')(data)

    # Third component of main path
    data = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(data)
    data = BatchNormalization(axis=3, name=bn_name_base + '2c')(data)
    data = Activation('relu')(data)

    # Shortcut path
    shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                      name=conv_name_base + '10')(shortcut)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    # Final step: Add shortcut value to main path, and pass it through a ReLu activation
    data = Add()([data, shortcut])
    data = Activation('relu')(data)

    return data


def build_resnet(img_height, img_width, classes=2, additional_input=False, vec_dim=3):
    """
    Creates a modified ResNet-50 model using Keras functional API
    :param img_height: Height of input image
    :param img_width: Width of input image
    :param classes: Number of classification groups
    :param additional_input: Whether an additional input vector is used
    :param vec_dim: Dimension of the additional input vector
    :return: Created functional API model
    """

    input_shape = (img_height, img_width, 1)

    if K.image_data_format() == "channels_first":
        input_shape = (1, img_height, img_width)
    K.set_learning_phase(1)

    x_input = Input(input_shape)

    # Stage 1
    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)

    # Max Pooling
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')

    # Max Pooling
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Stage 3
    x = convolutional_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')

    # Stage 4
    x = convolutional_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')

    # Stage 5
    x = convolutional_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='d')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='e')

    # Average Pooling
    x = AveragePooling2D(pool_size=(7, 7), padding='same')(x)

    # Output Layer
    x = Flatten()(x)
    if additional_input:
        # Concatenate additional attributes to input if provided
        vec_input = Input((vec_dim,))
        x = Concatenate()([x, vec_input])
    x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    if additional_input:
        model = Model(inputs=[x_input, vec_input], outputs=x, name='ResNet50')
    else:
        model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model
