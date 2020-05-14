from keras.layers import Dropout

from LayerAndLoss import *

cl1 = 24  # 16
cl2 = 36  # 32
cl3 = 48  # 64
cl4 = 60  # 64
cl5 = 72  # 64


def net1__binomial_crossentropy(input_shape, num_classes):
    # Building up our CNN
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(cl1, kernel_size=(3, 3), activation=kb.tanh, input_shape=input_shape))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl1, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh, strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(BinomialDropout_Train_Only(0.25))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh, padding='same'))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))  # , padding='same'
    model.add(Activation('relu'))

    model.add(BinomialDropout_Train_Only(0.25))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Flatten layer
    model.add(Flatten())

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))

    model.add(BinomialDropout_Train_Only(0.25))

    # Apply Softmax
    model.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd,
                  metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])

    return model, 'net1__binomial_crossentropy'


def net1b__binomial_crossentropy(input_shape, num_classes):
    weight_decay = 1e-4

    # Building up our CNN
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(cl1, kernel_size=(3, 3), activation=kb.tanh, input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl1, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh, strides=(2, 2)))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # model.add(Dropout(0.25))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh, padding='same'))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(
        Conv2D(cl4, (3, 3), activation=kb.tanh))  # , padding='same'
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    BatchNormalization(momentum=0.9, epsilon=1e-5)
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    model.add(GlobalAveragePooling2D())
    # Apply Softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay)))

    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model, 'net1b__binomial_crossentropy'


def net2_binomial_multinom_loss(input_shape, num_classes):
    # Building up our CNN
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(cl1, kernel_size=(3, 3), activation=kb.tanh, input_shape=input_shape))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl1, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh, strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh, padding='same'))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))  # , padding='same'
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Flatten layer
    model.add(Flatten())

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))
    model.add(Binomial_In_Train_Only(custom_activation_t10))

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))

    # Apply Softmax
    model.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss=multinom_loss, optimizer=sgd,
                  metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])

    return model, 'net2_binomial_multinom_loss'


def net3_multinom_loss(input_shape, num_classes):
    # Building up our CNN
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(cl1, kernel_size=(3, 3), activation=kb.tanh, input_shape=input_shape))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl1, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl2, (3, 3), activation=kb.tanh, strides=(2, 2)))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh, padding='same'))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl3, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl4, (3, 3), activation=kb.tanh))  # , padding='same'
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Conv2D(cl5, (3, 3), activation=kb.tanh))
    model.add(Activation('relu'))

    # Flatten layer
    model.add(Flatten())

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))

    # Fully connected Layer
    model.add(Dense(512, activation=kb.tanh))
    model.add(Activation('relu'))

    # Apply Softmax
    model.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss=multinom_loss, optimizer=sgd,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model, 'net3_multinom_loss'
