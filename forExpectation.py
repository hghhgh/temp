import tensorflow.keras as keras
from tensorflow.keras import backend as kb
from tensorflow.keras.layers import Input, add, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

from ExpectationLayers import *


def net1_expectation_binomial_residual_network(input_shape, classes_num, custom_activation, t4e, stack_n=5):
    l1 = 96
    l2 = 96
    l3 = 96
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation(custom_activation)(
            Activation('relu')(Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))))
        conv_1 = EConv2D(o_filters, t4e=t4e, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation(custom_activation)(Activation('relu')(
            Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))))
        conv_2 = EConv2D(o_filters, t4e=t4e, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = EConv2D(o_filters, t4e=t4e, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                 activation=kb.tanh,
                                 kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = EConv2D(filters=l1, t4e=t4e, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, l1, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, l2, True)
    for _ in range(1, stack_n):
        x = residual_block(x, l2, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, l3, True)
    for _ in range(1, stack_n):
        x = residual_block(x, l3, False)

    # x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x = Activation(custom_activation)(
        Activation('relu')(Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))))
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net1_expectation_binomial_residual_network'


def net1mnist_expectation_binomial_residual_network(input_shape, classes_num, custom_activation, t4e, stack_n=5):
    l1 = 16
    l2 = 32
    l3 = 64
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation(custom_activation)(
            Activation('relu')(Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))))
        conv_1 = EConv2D(o_filters, t4e=t4e, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation(custom_activation)(Activation('relu')(
            Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))))
        conv_2 = EConv2D(o_filters, t4e=t4e, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = EConv2D(o_filters, t4e=t4e, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                 activation=kb.tanh,
                                 kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = EConv2D(filters=l1, t4e=t4e, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, l1, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, l2, True)
    for _ in range(1, stack_n):
        x = residual_block(x, l2, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, l3, True)
    for _ in range(1, stack_n):
        x = residual_block(x, l3, False)

    # x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x = Activation(custom_activation)(
        Activation('relu')(Activation(keras.activations.tanh)(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))))
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net1_expectation_binomial_residual_network'
