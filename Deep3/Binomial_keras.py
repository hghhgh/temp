import datetime
import itertools
import json
import sys

import matplotlib.pyplot as plt
import pickle

import gc

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

from forBernoulli import *
from forExpectation import *


# -----------------------------------------------------------

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch, per=1):
    per = .5
    if epoch < per * 50:
        return 0.1
    if epoch < per * 75:
        return 0.05
    if epoch < per * 125:
        return 0.01
    if epoch < per * 200:
        return 0.005
    if epoch < per * 275:
        return 0.001
    if epoch < per * 350:
        return 0.0005
    if epoch < per * 425:
        return 0.0001
    if epoch < per * 500:
        return 0.00005
    return 0.00001


def fit_model_exp(problem, name, batch_size, t, epochs, model, verbose, x_test, x_train, y_test, y_train):
    datagen, x_test, x_train = preProcessData(problem, x_test, x_train)

    # set callback
    import os
    if not os.path.exists(name + '_weights'):
        os.makedirs(name + '_weights')
    cbks = [TensorBoard(log_dir=name + '_TensorBoard', histogram_freq=0), LearningRateScheduler(scheduler),
            keras.callbacks.ModelCheckpoint(name + '_weights/{epoch:08d}.h5', save_weights_only=True, period=10)]

    # Fit our model
    history = model.fit_generator(rep_input_flow(x_train, y_train, batch_size=batch_size, t=t),
                                  epochs=epochs, verbose=verbose, steps_per_epoch=len(y_train) // batch_size + 1,
                                  callbacks=cbks, validation_data=(x_test, y_test))
    # Evaluate our model
    # score1 = model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=t), verbose=2)
    score = model.evaluate(x_test, y_test, verbose=2)

    # Return
    return score, history, model


def fit_model_no_augment(problem, name, batch_size, epochs, model, verbose, x_test, x_train, y_test, y_train):
    # color preprocessing
    if problem != 'mnist':
        x_train, x_test = color_preprocessing(x_train, x_test)

    # set callback
    import os
    if not os.path.exists(name + '_weights'):
        os.makedirs(name + '_weights')
    cbks = [TensorBoard(log_dir=name + '_TensorBoard', histogram_freq=0), LearningRateScheduler(scheduler),
            keras.callbacks.ModelCheckpoint(name + '_weights/{epoch:08d}.h5', save_weights_only=True, period=10)]

    # Fit our model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        callbacks=cbks, validation_data=(x_test, y_test))
    # history = model.fit(x_train, y_train, batch_size=batch_size,
    #                     epochs=epochs, verbose=verbose, steps_per_epoch=len(y_train) // batch_size + 1,
    #                     callbacks=cbks, validation_data=(x_test, y_test),
    #                     validation_steps=len(y_test) // batch_size + 1)
    # Evaluate our model
    score = model.evaluate(x_test, y_test, verbose=2)

    # Return
    return score, history, model


def preProcessData(problem, x_test, x_train):
    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    if problem != 'mnist':
        x_train, x_test = color_preprocessing(x_train, x_test)
    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 # zca_whitening=True,
                                 # rotation_range=25,
                                 # shear_range=0.125,
                                 # zoom_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)
    return datagen, x_test, x_train


def saveResult(model, modelname, history, score):
    # serialize weights to HDF5
    model.save(modelname + '.h5')
    model.save_weights(modelname + "_weights.h5")
    print("Saved model to disk")
    print(('Test loss:', score[0]))
    print(('Test accuracy:', score[1]))
    try:
        f = open(modelname + "_model.txt", "w")
        model.summary(line_length=100, print_fn=f.write)
        f.close()
        f = open(modelname + "_scores.txt", "w")
        f.write(str(score))
        f.close()
        f = open(modelname + "_loss_history.txt", "w")
        f.write(str(history.history))
        f.close()
        f = open(modelname + "_loss_history.json", "w")
        f.write(json.dump(history.history))
        f.close()
        plot_model(model, to_file=modelname + '_model.png')

        # serialize model to JSON
        model_json = model.to_json()
        with open(modelname + "_model.json", "w") as json_file:
            json_file.write(model_json)

    except:
        pass
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(modelname + '_accuracy.png')
    # plt.show()
    #
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(modelname + '_loss.png')
    # plt.show()


def saveTime(duTime, modelname):
    with open(modelname + "_TrainDuration.txt", "w") as json_file:
        json_file.write(json.dumps(duTime))


def do_train_exp(problem, modelfunction, custom_activation, t, epochs=100, loadWeightsFrom='', loadModelFrom=''):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    if problem == 'mnist':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # the data, split between train and test sets
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif problem == 'cifar10':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
    elif problem == 'cifar100':
        batch_size = 32
        num_classes = 100
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Check for any available GPUs
    kb.tensorflow_backend._get_available_gpus()

    model, funame = modelfunction(input_shape, num_classes, custom_activation, t4e=t)
    nam = 'results/' + str(now) + problem + '_t' + str(t) + '_' + funame + '/e'
    if t == 0:
        nam = 'results/' + str(now) + problem + '_linear_' + funame + '/e'
        t = 1
    print(('----> Problem: ', nam))
    print((model.summary()))
    if len(loadWeightsFrom) > 1:
        model.load_weights(loadWeightsFrom)
        print('Weights loaded !')
    if len(loadModelFrom) > 1:
        model.load(loadModelFrom)
        print('Model loaded !')

    # calc_confusion_matrix(problem, model, x_test, y_test, batch_size, x_train)
    stTime = datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')
    score, history, model = fit_model_exp(problem, nam, batch_size, t, epochs, model, 2, x_test, x_train, y_test,
                                          y_train)
    endTime = datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')
    duTime = {'startTime': stTime, 'endTime': endTime}
    saveTime(duTime, nam)
    saveResult(model, nam, history, score)
    calc_confusion_matrix(problem, model, nam, x_test, y_test, batch_size, x_train)
    print('-------------------------------------------------------------------------------------------------------------------------')


def do_multitest_exp(problem, modelfunction, loadWeightsFrom):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    if problem == 'mnist':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # the data, split between train and test sets
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif problem == 'cifar10':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
    elif problem == 'cifar100':
        batch_size = 32
        num_classes = 100
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Check for any available GPUs
    datagen, x_test, x_train = preProcessData(problem, x_test, x_train)

    print('startTime: ' + datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_'))
    res = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, keras.activations.linear, t4e=1)
    model.load_weights(loadWeightsFrom)
    # model = load_model(loadWeightsFrom, custom_objects={'custom_activation_t1':custom_activation_t1, 'EConv2D':EConv2D})
    print('linear ...')
    res['linear'] = {'res': model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=1),
                                                     steps=len(y_train) // batch_size + 1, verbose=2),
                     'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=1)
    model.load_weights(loadWeightsFrom)
    print('t1 ...')
    res['t1'] = {'res': model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=1),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=2)
    model.load_weights(loadWeightsFrom)
    print('t2 ...')
    res['t2'] = {'res': model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=2),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=3)
    model.load_weights(loadWeightsFrom)
    print('t3 ...')
    res['t3'] = {'res': model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=3),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=5)
    model.load_weights(loadWeightsFrom)
    print('t5 ...')
    res['t5'] = {'res': model.evaluate_generator(rep_input_flow(x_test, y_test, batch_size=batch_size, t=5),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    batch_size = None
    num_classes = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    input_shape = None

    gc.collect()

    return res


def do_multitest_no_augment(problem, modelfunction, loadWeightsFrom):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    if problem == 'mnist':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # the data, split between train and test sets
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif problem == 'cifar10':
        batch_size = 32
        num_classes = 10
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
    elif problem == 'cifar100':
        batch_size = 32
        num_classes = 100
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Check for any available GPUs
    datagen, x_test, x_train = preProcessData(problem, x_test, x_train)

    print('startTime: ' + datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_'))
    res = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, keras.activations.linear)
    model.load_weights(loadWeightsFrom)
    print('linear ...')
    res['linear'] = {'res': model.evaluate(x_test, y_test, verbose=2),
                     'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1)
    model.load_weights(loadWeightsFrom)
    print('t1 ...')
    res['t1'] = {'res': model.evaluate(x_test, y_test, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t2)
    model.load_weights(loadWeightsFrom)
    print('t2 ...')
    res['t2'] = {'res': model.evaluate(x_test, y_test, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t3)
    model.load_weights(loadWeightsFrom)
    print('t3 ...')
    res['t3'] = {'res': model.evaluate(x_test, y_test, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t5)
    model.load_weights(loadWeightsFrom)
    print('t5 ...')
    res['t5'] = {'res': model.evaluate(x_test, y_test, verbose=2),
                 'time': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    batch_size = None
    num_classes = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    input_shape = None

    gc.collect()

    return res


def do_train_no_augment(problem, postfix, modelfunction, custom_activation, epochs=100, loadWeightsFrom='',
                        loadModelFrom=''):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    if problem == 'mnist':
        batch_size = 64
        num_classes = 10
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # the data, split between train and test sets
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif problem == 'cifar10':
        batch_size = 64
        num_classes = 10
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
    elif problem == 'cifar100':
        batch_size = 64
        num_classes = 100
        img_rows, img_cols = 32, 32
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        if kb.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Check for any available GPUs
    kb.tensorflow_backend._get_available_gpus()

    model, funame = modelfunction(input_shape, num_classes, custom_activation)
    nam = 'results/' + str(now) + problem + '_' + postfix + '_' + funame + '_no_augments/e'
    print(('----> Problem: ', nam))
    print((model.summary()))
    if len(loadWeightsFrom) > 1:
        model.load_weights(loadWeightsFrom)
        print('Weights loaded !')
    if len(loadModelFrom) > 1:
        model.load(loadModelFrom)
        print('Model loaded !')

    # calc_confusion_matrix(problem, model, x_test, y_test, batch_size, x_train)
    score, history, model = fit_model_no_augment(problem, nam, batch_size, epochs, model, 2, x_test, x_train, y_test,
                                                 y_train)
    saveResult(model, nam, history, score)
    calc_confusion_matrix(problem, model, nam, x_test, y_test, batch_size,
                          x_train)  # problem, model, modelname, x_test, y_tes, batch_size, x_train
    print('-------------------------------------------------------------------------------------------------------------------------')


def calc_confusion_matrix(problem, model, modelname, x_test, y_tes, batch_size, x_train):
    datagen, x_test, x_train = preProcessData(problem, x_test, x_train)
    # model.load_weights('results/StartAt_190213-210251_cifar10_net2f_bernoulli_residual_network/e_weights/00000500.h5')
    # from keras.models import load_model
    # model = load_model('results/StartAt_190213-210251_cifar10_net2f_bernoulli_residual_network/e.h5',
    #                    custom_objects={'custom_activation_t1':custom_activation_t1})
    predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=2)
    print(score)

    y_test1 = numpy.argmax(y_tes, axis=1)
    y_pred = numpy.argmax(predictions, axis=1)
    cnf_matrix = confusion_matrix(y_test1, y_pred)
    plot_confusion_matrix(modelname, cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # report = metrics.classification_report(1, y_pred, target_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(report)


def plot_confusion_matrix(modelname, cm, classes, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(modelname + '_test_confusion.png')


def TrainCIFAR10(db, maxepochs):
    #  -------------------------------------------------- Expectation Binomial ----------------------------------------
    do_train_exp(db, net1_expectation_binomial_residual_network, custom_activation_t1, t=1, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1_expectation_binomial_residual_network, custom_activation_t1, t=2, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1_expectation_binomial_residual_network, custom_activation_t1, t=3, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1_expectation_binomial_residual_network, custom_activation_t1, t=4, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1_expectation_binomial_residual_network, keras.activations.linear, t=0, epochs=maxepochs,
                 loadWeightsFrom='')
    #  ---------------------------------------------------------- Binomial -------------------------------------------

    do_train_no_augment(db, 't1', net2_bernoulli_residual_network, custom_activation_t1, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't2', net2_bernoulli_residual_network, custom_activation_t2, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't3', net2_bernoulli_residual_network, custom_activation_t3, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't5', net2_bernoulli_residual_network, custom_activation_t5, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 'linear', net2_bernoulli_residual_network, keras.activations.linear, maxepochs,
                        loadWeightsFrom='')  #


def TestCifar10():
    db = 'cifar10'
    #  -------------------------------------------------- Expectation Binomial -----------------------------------------
    exp_multitest = {}
    print('exp: linear')
    exp_multitest['linear'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # linear
                                               loadWeightsFrom='results/StartAt_190428-164814_cifar10_t1_net1_expectation_binomial_residual_network/e_weights/00000210.h5')

    print('exp: t1')
    exp_multitest['t1'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t1
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t1_net1_expectation_binomial_residual_network/e_weights/00000190.h5')
    print('exp: t2')
    exp_multitest['t2'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t2
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t2_net1_expectation_binomial_residual_network/e_weights/00000210.h5')
    print('exp: t3')
    exp_multitest['t3'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t3
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t3_net1_expectation_binomial_residual_network/e_weights/00000160.h5')

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + '_exp_multitest_' + serie + '.pkl', 'wb') as f:
        pickle.dump(exp_multitest, f, pickle.HIGHEST_PROTOCOL)
    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {}
    print('no_augment: linear')
    noaug_multitest['linear'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # linear
                                                        loadWeightsFrom='results/StartAt_190428-164814_cifar10_linear_net2_bernoulli_residual_network_no_augments/e_weights/00000080.h5')
    print('no_augment: t1')
    noaug_multitest['t1'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t1
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t1_net2_bernoulli_residual_network_no_augments/e_weights/00000220.h5')
    print('no_augment: t2')
    noaug_multitest['t2'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t2
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t2_net2_bernoulli_residual_network_no_augments/e_weights/00000240.h5')
    print('no_augment: t3')
    noaug_multitest['t3'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t3
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t3_net2_bernoulli_residual_network_no_augments/e_weights/00000240.h5')
    print('no_augment: t5')
    noaug_multitest['t5'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t5
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t5_net2_bernoulli_residual_network_no_augments/e_weights/00000110.h5')

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'noaug_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(noaug_multitest, f, pickle.HIGHEST_PROTOCOL)


def TrainMNIST(db, maxepochs):
    #  -------------------------------------------------- Expectation Binomial ----------------------------------------
    do_train_exp(db, net1mnist_expectation_binomial_residual_network, custom_activation_t1, t=1, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1mnist_expectation_binomial_residual_network, custom_activation_t1, t=2, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1mnist_expectation_binomial_residual_network, custom_activation_t1, t=3, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1mnist_expectation_binomial_residual_network, custom_activation_t1, t=4, epochs=maxepochs,
                 loadWeightsFrom='')
    do_train_exp(db, net1mnist_expectation_binomial_residual_network, keras.activations.linear, t=0, epochs=maxepochs,
                 loadWeightsFrom='')
    #  ---------------------------------------------------------- Binomial -------------------------------------------

    do_train_no_augment(db, 't1', net2mnist_bernoulli_residual_network, custom_activation_t1, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't2', net2mnist_bernoulli_residual_network, custom_activation_t2, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't3', net2mnist_bernoulli_residual_network, custom_activation_t3, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 't5', net2mnist_bernoulli_residual_network, custom_activation_t5, maxepochs,
                        loadWeightsFrom='')  #
    do_train_no_augment(db, 'linear', net2mnist_bernoulli_residual_network, keras.activations.linear, maxepochs,
                        loadWeightsFrom='')  #


def TestMNIST():
    db = 'mnist'
    #  -------------------------------------------------- Expectation Binomial -----------------------------------------
    exp_multitest = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}
    print('exp: linear')
    exp_multitest['linear'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # linear
                                               loadWeightsFrom='results/StartAt_190528-094208_mnist_linear_net1_expectation_binomial_residual_network/e_weights/00000070.h5'),

    print('exp: t1')
    exp_multitest['t1'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t1
                                           loadWeightsFrom='results/StartAt_190528-094208_mnist_t1_net1_expectation_binomial_residual_network/e_weights/00000120.h5'),
    print('exp: t2')
    exp_multitest['t2'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t2
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t2_net1_expectation_binomial_residual_network/e_weights/00000160.h5'),
    print('exp: t3')
    exp_multitest['t3'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t3
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t3_net1_expectation_binomial_residual_network/e_weights/00000200.h5'),
    print('exp: t4')
    exp_multitest['t4'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t4
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t4_net1_expectation_binomial_residual_network/e_weights/00000140.h5'),

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'exp_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(exp_multitest, f, pickle.HIGHEST_PROTOCOL)
    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}
    print('no_augment: linear')
    noaug_multitest['linear'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # linear
                                                        loadWeightsFrom='results/StartAt_190529-114807_mnist_linear_net2_bernoulli_residual_network_no_augments/e_weights/00000230.h5'),
    print('no_augment: t1')
    noaug_multitest['t1'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t1
                                                    loadWeightsFrom='results/StartAt_190528-094208_mnist_t1_net2_bernoulli_residual_network_no_augments/e_weights/00000120.h5'),
    print('no_augment: t2')
    noaug_multitest['t2'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t2
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t2_net2_bernoulli_residual_network_no_augments/e_weights/00000230.h5'),
    print('no_augment: t3')
    noaug_multitest['t3'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t3
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t3_net2_bernoulli_residual_network_no_augments/e_weights/00000040.h5'),
    print('no_augment: t5')
    noaug_multitest['t5'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t5
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t5_net2_bernoulli_residual_network_no_augments/e_weights/00000210.h5'),

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'noaug_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(noaug_multitest, f, pickle.HIGHEST_PROTOCOL)


# kb.set_floatx('float16')
# kb.set_epsilon(1e-4)  # default is 1e-7
tf.random.set_seed(73142193)  # For reproducibility
now = datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')

# db =  'cifar10', 'cifar100', 'mnist'
maxepochs = 250

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
cdb = sys.argv[1]
withTrain = sys.argv[2]
testCount = int(sys.argv[3])

orig_stdout = sys.stdout
f = open('MultiRes_' + cdb + '_' + now + '.txt', 'w')
sys.stdout = f

if cdb == 'cifar10':
    if withTrain == '1':
        TrainCIFAR10('cifar10', maxepochs)
    for i in range(testCount):
        TestCifar10()

if cdb == 'mnist':
    if withTrain == '1':
        TrainMNIST('mnist', maxepochs)
    for i in range(testCount):
        TestMNIST()

sys.stdout = orig_stdout
f.close()
