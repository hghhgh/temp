import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10, cifar100, mnist
from tensorflow.keras import backend as kb

def color_preprocessing(x_train, x_test):
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  mean = [125.307, 122.95, 113.865]
  std = [62.9932, 62.0887, 66.7048]
  for i in range(3):
      x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
      x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
  return x_train, x_test

def loadProblem(problem):
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
      batch_size = 96
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

  if problem != 'mnist':
    x_train, x_test = color_preprocessing(x_train, x_test)
        
  return x_train, y_train, x_test, y_test, input_shape, num_classes, batch_size
