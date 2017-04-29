import numpy as np
import tensorflow as tf


alphanum_char_set = list("abcdefghijklmnopqrstuvwxyz0123456789")
#This is len 32 (orig paper distinguished between minus and a dash)
punc_char_set = list("`~!@#$%^&*()_+-={}|[]\\:\";'<>?,./")
default_char_set = alphanum_char_set + punc_char_set + [" "]


def one_hot(object, object_vector):
    a = np.array(object_vector) == object
    return a.astype(float)

#Weight generating functions                                                                               
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Layer generating wrappers
def conv(x, W):
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def pool(x, pool_size):
    return tf.nn.pool(x, window_shape =[pool_size], pooling_type = "MAX",\
                          padding = 'SAME', strides = [pool_size],)
