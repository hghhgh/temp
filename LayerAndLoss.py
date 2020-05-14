import tensorflow.keras as keras
import numpy
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as kb
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten, Input, add, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import random_ops, math_ops
from tensorflow.python.ops.nn_ops import _get_noise_shape


@tf.custom_gradient
def custom_activation_t1(x):
    s = tf.shape(x)
    bi = kb.random_binomial(s, x)

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t2(x):
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    # ps = kb.expand_dims(x, axis=-1)
    # vals = [ps, ps]
    # aps = tf.concat(vals, axis=-1, name='concat')
    # s = tf.shape(aps)
    # bo = kb.random_binomial(s, aps)
    # bi = kb.sum(bo, axis=-1) / 2.
    s = tf.shape(x)
    bi = (kb.random_binomial(s, x) + kb.random_binomial(s, x)) / 2.

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t3(x):
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    s = tf.shape(x)
    bo = kb.random_binomial(s, x)
    bo += kb.random_binomial(s, x)
    bo += kb.random_binomial(s, x)
    bi = bo / 3.

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t4(x):
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    s = tf.shape(x)
    bo = kb.random_binomial(s, x)
    bo += kb.random_binomial(s, x)
    bo += kb.random_binomial(s, x)
    bo += kb.random_binomial(s, x)
    bi = bo / 4.

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t5(x):
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    ps = kb.expand_dims(x, axis=-1)
    vals = [ps, ps, ps, ps, ps]
    aps = tf.concat(vals, axis=-1, name='concat')
    s = tf.shape(aps)
    bo = kb.random_binomial(s, aps)
    bi = kb.sum(bo, axis=-1) / 5.

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t10(x):
    t = 10
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    ps = kb.expand_dims(x, axis=-1)
    vals = []
    for i in range(t):
        vals.append(ps)
    aps = tf.concat(vals, axis=-1, name='concat')
    s = tf.shape(aps)
    bo = kb.random_binomial(s, aps)
    bi = kb.sum(bo, axis=-1) / t

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t20(x):
    t = 20
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    ps = kb.expand_dims(x, axis=-1)
    vals = []
    for i in range(t):
        vals.append(ps)
    aps = tf.concat(vals, axis=-1, name='concat')
    s = tf.shape(aps)
    bo = kb.random_binomial(s, aps)
    bi = kb.sum(bo, axis=-1) / t

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t50(x):
    t = 50
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    ps = kb.expand_dims(x, axis=-1)
    vals = []
    for i in range(t):
        vals.append(ps)
    aps = tf.concat(vals, axis=-1, name='concat')
    s = tf.shape(aps)
    bo = kb.random_binomial(s, aps)
    bi = kb.sum(bo, axis=-1) / t

    def grad(dy):
        return 1. * dy

    return bi, grad


@tf.custom_gradient
def custom_activation_t(x):
    t = 100
    # t = kb.get_value(kb.cast(kb.random_uniform([1], 1, 100), "int32"))[0]
    ps = kb.expand_dims(x, axis=-1)
    vals = []
    for i in range(t):
        vals.append(ps)
    aps = tf.concat(vals, axis=-1, name='concat')
    s = tf.shape(aps)
    bo = kb.random_binomial(s, aps)
    bi = kb.sum(bo, axis=-1) / t

    def grad(dy):
        return 1. * dy

    return bi, grad


# @tf.custom_gradient
# def custom_activation(x):
#     t = 100.
#     # t = K.get_value(K.random_uniform([1], 1, 100))[0]
#     ps = kb.expand_dims(x, axis=-1)
#     aps = tf.concat([ps, 1 - ps], axis=-1, name='concat')
#     dist = tf.distributions.Multinomial(total_count=t, probs=aps)
#     xt = dist.sample(1)
#     bi = kb.sum(xt, axis=0)
#     # bi = xt[0,...,0] / t
#
#     def grad(dy):
#         return 1. * dy
#
#     return bi, grad


def multinom_loss(y_true, y_pred):
    t = 50.
    tar = (t - 1) / t
    oth = 1 - tar
    my_true = y_true * (tar - oth) + oth
    dist = tf.distributions.Multinomial(total_count=t, probs=my_true)
    yt = dist.sample(1) / t
    return kb.categorical_crossentropy(yt, y_pred)


class Binomial_In_Train_Only(Layer):
    def __init__(self, activation, **kwargs):
        super(Binomial_In_Train_Only, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    # def call(self, inputs):
    #     return self.activation(inputs)

    def call(self, inputs, training=None):
        return kb.in_train_phase(self.activation(inputs), inputs,
                                 training=training)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Binomial_In_Train_Only, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BinomialDropout_Train_Only(Layer):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!1 -> use : tf.boolean_mask
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(BinomialDropout_Train_Only, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = kb.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            seed = numpy.random.randint(10e6)
            noise_shape = _get_noise_shape(inputs, noise_shape)
            keep_prob = 1. - self.rate
            # uniform [keep_prob, 1.0 + keep_prob)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=inputs.dtype)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = math_ops.floor(random_tensor)

            binomidx = 1 - binary_tensor
            binom = binomidx * custom_activation_t10(inputs)
            dropped_inputs = binary_tensor * inputs + binom

            return kb.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(BinomialDropout_Train_Only, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BinomialDropout_Train_Test(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(BinomialDropout_Train_Test, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = kb.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            seed = numpy.random.randint(10e6)
            noise_shape = _get_noise_shape(inputs, noise_shape)
            keep_prob = 1. - self.rate
            # uniform [keep_prob, 1.0 + keep_prob)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=inputs.dtype)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = math_ops.floor(random_tensor)

            binomidx = 1 - binary_tensor
            binom = binomidx * custom_activation_t10(inputs)
            dropped_inputs = binary_tensor * inputs + binom

            return dropped_inputs
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(BinomialDropout_Train_Test, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def top3_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
