import numpy as np
import tensorflow as tf


#Character sets
alphanum_char_set = list("abcdefghijklmnopqrstuvwxyz0123456789")
#This is len 32 (orig paper distinguished between minus and a dash)
punc_char_set = list("`~!@#$%^&*()_+-={}|[]\\:\";'<>?,./")
default_char_set = alphanum_char_set + punc_char_set + [" "]



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


#Preprocessing

def one_hot(object, object_vector):
    a = np.array(object_vector) == object
    return a.astype(float)




#Amazon

amazon_polarity_labels = ['1', '2']


def str2lab_am1(s):
    return s

def str2featarr_am1(sent_string, char_set, sent_max_len = 512):

    sent_string = sent_string.lower()

    orig_chars_reg = sent_string[:sent_max_len] + " "*(sent_max_len - len(sent_string))
    arr = np.zeros((0,len(char_set)))
    for char in orig_chars_reg:
        arr = np.vstack((arr, one_hot(char, char_set)))
    return arr
