from __future__ import division, print_function
import numpy as np
import nltk
import tensorflow as tf
import os
import csv

from utilities import weight_variables, bias_variables, conv, pool, default_char_set, one_hot

#for amazon dataset
from utilities import amazon_polarity_labels, str2featarr_am1, str2lab_am1


#Set these
CHAR_SET = default_char_set
###

DATA_FILE = "./data/amazon_polarity/train.csv"
LABEL_INDEX = 0
TEXT_INDEX = 2
STRING_TO_LABEL = str2lab_am1
STRING_TO_FEATURE_ARRAY = str2featarr_am1
ALL_LABELS = amazon_polarity_labels
MODEL_SAVE_PATH = "./saved_models/model0.ckpt"
STARTING_MODEL_PATH = "./saved_models/model0.ckpt"

### OPTIMIZATION 
BATCH_SIZE = 128
STEP_SIZE = .01 #orig paper halved this every 3 epochs for 10 times
MOMENTUM = .9
NUM_ITERS = 30000

### MODEL PARAMETERS
INPUT_WIDTH = 512
CONV_FILTER_SIZE_A = 7
CONV_FILTER_SIZE_B = 3
CONV_OUT_NUM_FEATS = 512
POOL_SIZE = 4
CONN_NUM_FEATS = 512
DROPOUT_PROB = .5


def main(start_with_saved = True):

    CHAR_SET_SIZE = len(CHAR_SET)
    NUM_LABELS = len(ALL_LABELS)

    #(0) Placeholders

    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, INPUT_WIDTH, CHAR_SET_SIZE])
    y = tf.placeholder(tf.float32, shape = [BATCH_SIZE, NUM_LABELS])
    dropout_prob = tf.placeholder(tf.float32)

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

    drop_1 = tf.nn.dropout(thresh_conn_1, dropout_prob)

    out3 = drop_1


    #(4)  Second connected layer

    in4 = out3

    conn_weights_2 = weight_variables([CONN_NUM_FEATS, CONN_NUM_FEATS])
    
    conn_2 = tf.matmul(in4, conn_weights_2) #shape is

    thresh_bias_conn_2 = bias_variables([CONN_NUM_FEATS])
    thresh_conn_2 = tf.nn.relu(conn_2 + thresh_bias_conn_2)

    drop_2 = tf.nn.dropout(thresh_conn_2, dropout_prob)

    out4 = drop_2


    #(5) Readout layer

    in5 = out4

    readout_weights = weight_variables([CONN_NUM_FEATS, NUM_LABELS])
    readout_bias = bias_variables([NUM_LABELS])
    readout = tf.matmul(in5, readout_weights) + readout_bias
    out5 = readout


    # Error and Optimizer

    out = out5

    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    train_step = tf.train.MomentumOptimizer(STEP_SIZE, MOMENTUM).minimize(error)

    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #RUN MODEL
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    train_data_file = open(DATA_FILE, 'r')
    csv_reader = csv.reader(train_data_file)

    saver = tf.train.Saver()

    #Load saved model
    if start_with_saved:
        saver.restore(sess, STARTING_MODEL_PATH)
        print("Starting from model in {}.".format(STARTING_MODEL_PATH))

    for i in range(NUM_ITERS):

        label_batch_list = list()
        feature_batch_list = list()

        #Get next batch

        for _ in range(BATCH_SIZE):

            try:
                next_list = csv_reader.next()

            except StopIteration:
                train_data_file.close()
                train_data_file = open(DATA_FILE, 'r')
                csv_reader = csv.reader(train_data_file)
                next_list = csv_reader.next()

            label_batch_list.append(next_list[LABEL_INDEX])
            feature_batch_list.append(next_list[TEXT_INDEX])

        label_batch_list = [one_hot(STRING_TO_LABEL(s), ALL_LABELS) for s in label_batch_list]
        feature_batch_list = [STRING_TO_FEATURE_ARRAY(s, CHAR_SET, INPUT_WIDTH) for s in feature_batch_list]

        label_batch = np.array(label_batch_list)
        feature_batch = np.array(feature_batch_list)
        eval_dict =  {x: feature_batch, y: label_batch, dropout_prob: 1.}
        train_dict = {x: feature_batch, y: label_batch, dropout_prob: DROPOUT_PROB}

        train_accuracy = accuracy.eval(session = sess, feed_dict = eval_dict)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(session = sess, feed_dict = train_dict)

        #Periodically save contents of variables
        if i%100 == 0:
            saver.save(sess, MODEL_SAVE_PATH)
            print("Model saved in {}.".format(MODEL_SAVE_PATH))

    sess.close()
