import tensorflow as tf
import os
import numpy as np

def get_batch(data_array, i0, i1):
    """Provides a quick and dirty wrapping functionality"""
    if i0 > i1:
        a0 = data_array[i0:, :]
        a1 = data_array[:i1, :]
        return np.concatenate([a0, a1], 0)
    else:
        return data_array[i0:i1, :]

def train_net(data_array, input_placeholder, output_placeholder, error_tensor, train_tensor, summaries, batch_size, num_training_runs):

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
            in_arr = batch[:,:-1] #grab all but the last column
            out_arr = batch[:, -1] #grab last column

            sess.run(train_tensor, feed_dict={input_placeholder: in_arr, output_placeholder: out_arr})
            MSE = sess.run(error_tensor, feed_dict={input_placeholder: in_arr, output_placeholder: out_arr})



            if i % 50 == 0:
                summary = sess.run(summaries, feed_dict={input_placeholder: in_arr, output_placeholder: out_arr})
                train_writer.add_summary(summary, i)
                print("{0}.) Training MSE: {1}".format(i, MSE))


            i0 = (i0 + batch_size) % data_array.shape[0]#Wrap index if past bounds
            i1 = (i1 + batch_size) % data_array.shape[0]#Wrap index if past bounds
