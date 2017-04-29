import numpy as np
import nltk
import tensorflow as tf
from preprocessing import pp1, get_data_array_from_sent, get_label_from_pathname,\
    get_label_array_from_pathname
from utilities import weight_variables, bias_variables, conv, pool, default_char_set, one_hot
import os
from dataset import Dataset
#from future import division, print_function


#Set these
DATA_DIRECTORY = "./guten_data/"
CHAR_SET = default_char_set
PATHNAME_TO_LABEL_MAPPER = get_label_from_pathname
INPUT_WIDTH = 512
PREPROCESSOR = pp1
###
BATCH_SIZE = 128
STEP_SIZE = .01 #orig paper halved this every 3 epochs for 10 times
MOMENTUM = .9
NUM_ITERS = 2000
###
CONV_FILTER_SIZE_A = 7
CONV_FILTER_SIZE_B = 3
CONV_OUT_NUM_FEATS = 512
POOL_SIZE = 4
CONN_NUM_FEATS = 512

#These should probably be determined
CHAR_SET_SIZE = len(CHAR_SET)
ALL_LABELS =  [PATHNAME_TO_LABEL_MAPPER(DATA_DIRECTORY + filename) for\
                   filename in os.listdir(DATA_DIRECTORY)]
NUM_LABELS = len(ALL_LABELS)

def main():

    #(0) Placeholders

    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, INPUT_WIDTH, CHAR_SET_SIZE])
    y = tf.placeholder(tf.float32, shape = [BATCH_SIZE, NUM_LABELS])

    out0 = x


    #(1) First Conv Layer

    in1 = out0

    conv_weights_1 = weight_variables([CONV_FILTER_SIZE_A, CHAR_SET_SIZE, CONV_OUT_NUM_FEATS])
    conv1 = conv(in1, conv_weights_1) #shape is
    #[BATCH_SIZE, INPUT_WIDTH, CONV_OUT_NUM_FEATS]
    
    thresh_bias_1 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh1 = tf.nn.relu(conv1 + thresh_bias_1)#shape is still \
        #[BATCH_SIZE, INPUT_WIDTH, CONV_OUT_NUM_FEATS]

    pool1 = pool(thresh1, pool_size = POOL_SIZE) #shape is now \
        #[BATCH_SIZE, INPUT_WIDTH/POOL_SIZE, CONV_OUT_NUM_FEATS]

    out1 = pool1


    #(2) Second Conv Layer

    in2 = out1

    conv_weights_2 = weight_variables([CONV_FILTER_SIZE_A,\
                                          CONV_OUT_NUM_FEATS, CONV_OUT_NUM_FEATS])
    conv2 = conv(in2, conv_weights_2)#shape is \
        #[BATCH_SIZE, INPUT_WIDTH/POOL_SIZE, CONV_OUT_NUM_FEATS]

    thresh_bias_2 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh2 = tf.nn.relu(conv2 + thresh_bias_2)#shape is still \
        #[BATCH_SIZE, INPUT_WIDTH/POOL_SIZE, CONV_OUT_NUM_FEATS]

    pool2 = pool(thresh2, POOL_SIZE) #shape is now
    #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**2), CONV_OUT_NUM_FEATS

    out2 = pool2


    #(2.1) Third Conv Layer

    in2_1 = out2

    conv_weights_2_1 = weight_variables([CONV_FILTER_SIZE_B,\
                                          CONV_OUT_NUM_FEATS, CONV_OUT_NUM_FEATS])
    conv2_1 = conv(in2_1, conv_weights_2_1)#shape is \
        #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**2), CONV_OUT_NUM_FEATS]

    thresh_bias_2_1 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh2_1 = tf.nn.relu(conv2_1 + thresh_bias_2_1)

    out2_1 = thresh2_1


    #(2.2) Fourth Conv Layer

    in2_2 = out2_1

    conv_weights_2_2 = weight_variables([CONV_FILTER_SIZE_B,\
                                          CONV_OUT_NUM_FEATS, CONV_OUT_NUM_FEATS])
    conv2_2 = conv(in2_2, conv_weights_2_2)#shape is \
        #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**2), CONV_OUT_NUM_FEATS]

    thresh_bias_2_2 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh2_2 = tf.nn.relu(conv2_2 + thresh_bias_2_2)

    out2_2 = thresh2_2


    #(2.3) Fifth Conv Layer

    in2_3 = out2_2

    conv_weights_2_3 = weight_variables([CONV_FILTER_SIZE_B,\
                                          CONV_OUT_NUM_FEATS, CONV_OUT_NUM_FEATS])
    conv2_3 = conv(in2_3, conv_weights_2_3)#shape is \
        #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**2), CONV_OUT_NUM_FEATS]

    thresh_bias_2_3 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh2_3 = tf.nn.relu(conv2_3 + thresh_bias_2_3)

    out2_3 = thresh2_3


    #(2.4) Sixth Conv Layer

    in2_4 = out2_3

    conv_weights_2_4 = weight_variables([CONV_FILTER_SIZE_B,\
                                          CONV_OUT_NUM_FEATS, CONV_OUT_NUM_FEATS])
    conv2_4 = conv(in2_4, conv_weights_2_4)#shape is \
        #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**2), CONV_OUT_NUM_FEATS]

    thresh_bias_2_4 = bias_variables([CONV_OUT_NUM_FEATS])
    thresh2_4 = tf.nn.relu(conv2_4 + thresh_bias_2_4)

    pool2_4 = pool(thresh2_4, POOL_SIZE) #shape is now \
        #[BATCH_SIZE, INPUT_WIDTH/(POOL_SIZE**3), CONV_OUT_NUM_FEATS]

    out2_4 = thresh2_4


    #(3)  First connected layer

    in3 = out2_4

    conn_input_size = int(in3.shape[1])* int(in3.shape[2])
    reshaped = tf.reshape(in3, [BATCH_SIZE, conn_input_size])
    conn_weights_1 = weight_variables([conn_input_size, CONN_NUM_FEATS])
    
    conn_1 = tf.matmul(reshaped, conn_weights_1) #shape is 

    thresh_bias_conn_1 = bias_variables([CONN_NUM_FEATS])
    thresh_conn_1 = tf.nn.relu(conn_1 + thresh_bias_conn_1)
    #[BATCH_SIZE, CONN_NUM_FEATS]    
    out3 = thresh_conn_1


    #(4)  Second connected layer

    in4 = out3

    conn_weights_2 = weight_variables([CONN_NUM_FEATS, CONN_NUM_FEATS])
    
    conn_2 = tf.matmul(in4, conn_weights_2) #shape is

    thresh_bias_conn_2 = bias_variables([CONN_NUM_FEATS])
    thresh_conn_2 = tf.nn.relu(conn_2 + thresh_bias_conn_2)
    #[BATCH_SIZE, CONN_NUM_FEATS]
    out4 = thresh_conn_2


    #(5) Readout layer

    in5 = out4

    readout_weights = weight_variables([CONN_NUM_FEATS, NUM_LABELS])
    readout_bias = bias_variables([NUM_LABELS])
    readout = tf.matmul(in5, readout_weights) + readout_bias
    thresh_readout = tf.nn.softmax(readout)
    out5 = thresh_readout


    # Error and Optimizer

    out = out5

    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    train_step = tf.train.MomentumOptimizer(STEP_SIZE, MOMENTUM).minimize(error)

    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #RUN MODEL
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    data = Dataset(DATA_DIRECTORY)

    for i in range(NUM_ITERS):

        pathnames, lines = data.next_batch(BATCH_SIZE)

        label_batch_list = [get_label_array_from_pathname(pathname, ALL_LABELS) for pathname in pathnames]
        feature_batch_list = [get_data_array_from_sent(line, char_set = CHAR_SET,\
                                                           preprocessor = PREPROCESSOR,\
                                                           sent_max_len = INPUT_WIDTH) for line in lines]
        label_batch = np.array(label_batch_list)
        feature_batch = np.array(feature_batch_list)
        feed_dict =  {x: feature_batch, y: label_batch}

        train_accuracy = accuracy.eval(session = sess, feed_dict = feed_dict)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(session = sess, feed_dict = feed_dict)
        
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: test_feature_tensors, y: test_target_vectors}))

    sess.close()
