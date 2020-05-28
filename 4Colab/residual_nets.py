from LayerAndLoss import *

# l1 = 24  # 16
# l2 = 48  # 32
# l3 = 96  # 64
l1 = 16
l2 = 32
l3 = 64


def net4_residual_network_multinom_loss(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net4_residual_network_multinom_loss'


def net5_residual_network(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net5_residual_network'


# in train only
def net6_binomial_residual_network(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        # o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        # o1 = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(x))
        o1 = BinomialDropout_Train_Only(.15)(Activation('relu')(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        # o2 = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(conv_1))
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        # conv_2 = MyActivation(custom_activation)(Activation('relu')(conv_2))
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            # projection = MyActivation(custom_activation)(Activation('relu')(projection))
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
            # block = MyActivation(custom_activation)(Activation('relu')(Activation('tanh')(block)))
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # x = MyActivation(custom_activation)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net6_binomial_residual_network'


# in train and test
def net7_binomial_residual_network(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        # o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        # o1 = Activation(custom_activation_t10)(Activation('relu')(x))
        o1 = BinomialDropout_Train_Test(.15)(Activation('relu')(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        # o2 = Activation(custom_activation_t10)(Activation('relu')(conv_1))
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        # conv_2 = Activation(custom_activation)(Activation('relu')(conv_2))
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            # projection = Activation(custom_activation)(Activation('relu')(projection))
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
            # block = Activation(custom_activation)(Activation('relu')(Activation('tanh')(block)))
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # x = Activation(custom_activation)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net7_binomial_residual_network'


# in train and test
def net7b_binomial_residual_network(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        # o1 = Activation(custom_activation_t10)(Activation('relu')(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        # o2 = Activation(custom_activation_t10)(Activation('relu')(conv_1))
        # o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        o2 = BinomialDropout_Train_Only(.35)(Activation('relu')(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        # conv_2 = Activation(custom_activation)(Activation('relu')(conv_2))
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            # projection = Activation(custom_activation)(Activation('relu')(projection))
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
            # block = Activation(custom_activation)(Activation('relu')(Activation('tanh')(block)))
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # x = Activation(custom_activation)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net7b_binomial_residual_network'


# in train only
def net8_binomial_residual_network(input_shape, classes_num, stack_n=5):
    weight_decay = 1e-4
    img_input = Input(shape=input_shape)

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        # o1 = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        # o2 = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(conv_1))
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        # conv_2 = Binomial_In_Train_Only(custom_activation)(Activation('relu')(conv_2))
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=kb.tanh,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            # projection = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(projection))
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
            block = Binomial_In_Train_Only(custom_activation_t10)(Activation('relu')(Activation('tanh')(block)))
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=l1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=kb.tanh,
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

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # x = Binomial_In_Train_Only(custom_activation)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    resnet = Model(img_input, x)

    # set optimizer
    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet, 'net8_binomial_residual_network'
