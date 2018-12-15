import tensorflow as tf
import os
import math
import numpy as np


def get_batch(data_array, i0, i1):
    """Provides a quick and dirty wrapping functionality"""
    if i0 > i1:
        a0 = data_array[i1:, :]
        a1 = data_array[:i0, :]
        return np.concatenate([a0, a1], 0)
    else:
        return data_array[i0:i1, :]



def train_net(data_array, num_neurons, batch_size, num_training_runs):
    #The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
    #  nonlinear function approximator, of the form [input]->[tan-sigmoid hidden layer] -> [linear weighted sum layer] -> output
    #data_array is a 2d array where each row corresponds to a given well and each column corresponds to a given data header
    #   the final column in data_array must be the values of the variable to predict, cum_365_prod for this exercise

    inp = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[1]]) #input shape is an arbitrary length vector to allow batch training
    correct_output = tf.placeholder(dtype=tf.float32)
    tan_sig_layer = tf.contrib.layers.fully_connected(inp, num_neurons, activation_fn=tf.nn.sigmoid)
    lin_layer = tf.contrib.layers.fully_connected(tan_sig_layer, 1, activation_fn=None)#Specify None for linear activation


    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(correct_output, lin_layer))))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.5).minimize(error)  # training step
    tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
    tf.summary.tensor_summary("Linear Layer", lin_layer)
    tf.summary.scalar("MSE", error)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        log_dir = '/tmp/oag_int/train/'
        os.system("rm {0} -r".format(log_dir))
        train_writer = tf.summary.FileWriter('/tmp/oag_int/train/',
                                             sess.graph)

        i0 = 0
        i1 = batch_size
        tf.global_variables_initializer().run() #initialize variables
        for i in range(num_training_runs): #training episodes
            batch = get_batch(data_array,i0, i1)
            in_arr = batch[:,:data_array.shape[0]-1] #grab all but the last column
            out_arr = batch[:, -1] #grab last column

            sess.run(train, feed_dict={inp: in_arr, correct_output: out_arr})
            MSE = sess.run(error, feed_dict={inp: in_arr, correct_output: out_arr})



            if i % 50 == 0:
                summary = sess.run(merged, feed_dict={inp: in_arr, correct_output: out_arr})
                train_writer.add_summary(summary, i)
                print("{0}.) Training MSE: {1}".format(i, MSE))


            i0 = (i0 + batch_size) % data_array.shape[0]#Wrap index if past bounds
            i1 = (i1 + batch_size) % data_array.shape[0]#Wrap index if past bounds
