import os
import threading
import warnings

from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras import backend as K, activations, initializers, regularizers, constraints
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
# from keras_preprocessing.image import IteratorType


class EConv2D(Conv2D):
    def __init__(self, filters, t4e, H=1., **kwargs):
        super(EConv2D, self).__init__(filters, **kwargs)
        self.iteration = 1
        self.inp = 0
        self.T4E = t4e

    def call(self, inputs):
        if type(self.inp) == 'int' or self.iteration >= self.T4E:
            self.inp = inputs
            self.iteration = 1
        else:
            self.inp += (self.inp * self.iteration + inputs) / (self.iteration + 1.)
            self.iteration += 1
        outputs = K.conv2d(
            self.inp,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        return config


class EDense(Dense):
    def __init__(self, t4e, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(EDense, self).__init__(**kwargs)
        self.iteration = 1
        self.inp = 0
        self.T4E = t4e

    def call(self, inputs):
        if type(self.inp) == 'int' or self.iteration >= self.T4E:
            self.inp = inputs
            self.iteration = 1
        else:
            self.inp += (self.inp * self.iteration + inputs) / (self.iteration + 1.)
            self.iteration += 1

        output = K.dot(self.inp, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def rep_input_flow(x, y=None, batch_size=32, t=1):
    # return NumpyArrayIterator(x, y, batch_size=batch_size)
    idx = np.asarray(range(y.shape[0]))
    while True:
        # Select files (paths/indices) for the batch
        batch_idx = np.random.choice(a=idx, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_idx in batch_idx:
            input = x[input_idx, ...]
            output = y[input_idx, ...]
            for rep in range(t):
                batch_input.append(input)
                batch_output.append(output)
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)
