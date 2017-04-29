import numpy as np
import tensorflow as tf
from utilities import one_hot

def pp1(sent):
    return sent.lower()

def get_data_array_from_sent(sent_string, char_set, preprocessor = pp1,  sent_max_len = 512):
    
    #general preprocessing, including, eg lowercasing
    sent_string = preprocessor(sent_string)

    orig_chars_reg = sent_string[:sent_max_len] + " "*(sent_max_len - len(sent_string))
    dtens = np.zeros((0,len(char_set)))
    for char in orig_chars_reg:
        dtens = np.vstack((dtens, one_hot(char, char_set)))
    return dtens

def get_label_from_pathname(pathname):
    filename = pathname.split('/')[-1]
    label = filename.split(".")[0] 
    return label

def get_label_array_from_pathname(pathname, all_labels):
    label = get_label_from_pathname(pathname)
    return one_hot(label, all_labels)
