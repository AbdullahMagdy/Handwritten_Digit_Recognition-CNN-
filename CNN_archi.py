import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops



# build nn archi


def conv_2d_layer(x,w,b,stride = 1):
    x = tf.nn.conv2d(x,w,strides= [1 , stride , stride , 1],padding= "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpooling_layer(x , k = 2):
    return tf.nn.max_pool(x,ksize = [1,k,k,1],strides = [1 ,k,k,1] ,padding="SAME")


def conv_net(x, weights, biases):
    conv1 = conv_2d_layer(x, weights['wc1'], biases['bc1'])
    conv1 = maxpooling_layer(conv1, k = 2)
    conv2 = conv_2d_layer(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpooling_layer(conv2, k = 2)
    conv3 = conv_2d_layer(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpooling_layer(conv3, k = 2)
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


weights = {
    'wc1' : tf.get_variable('W0', shape = (3, 3, 1, 32), initializer = tf.contrib.layers.xavier_initializer()),
    'wc2' : tf.get_variable('W1', shape = (3, 3, 32, 64), initializer = tf.contrib.layers.xavier_initializer()),
    'wc3' : tf.get_variable('W2', shape = (3, 3, 64, 128), initializer = tf.contrib.layers.xavier_initializer()),
    'wd1' : tf.get_variable('W3', shape = (4 * 4 * 128, 128), initializer = tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable('W4', shape = (128, 10), initializer = tf.contrib.layers.xavier_initializer())
    # 10 == > no.classes(0 .... 9)
}

biases = {
    'bc1': tf.get_variable('B0', shape = (32), initializer = tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape = (64), initializer = tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape = (10), initializer = tf.contrib.layers.xavier_initializer()),
}

