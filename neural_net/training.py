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

def train_net(data_array, neural_net, run_params):

    with tf.Session() as sess:
        log_dir = '/tmp/oag_int/train/'
        os.system("rm {0} -r".format(log_dir))
        train_writer = tf.summary.FileWriter('/tmp/oag_int/train/',
                                             sess.graph)

        i0 = 0
        i1 = run_params.batch_size
        tf.global_variables_initializer().run() #initialize variables
        for i in range(run_params.num_training_steps):
            batch = get_batch(data_array,i0, i1)
            in_arr = batch[:,:-1] #grab all but the last column
            out_arr = batch[:, -1] #grab last column

            sess.run(neural_net.train_tensor, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})
            MSE = sess.run(neural_net.error_tensor, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})



            if i % run_params.log_period == 0:
                summary = sess.run(neural_net.summaries, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})
                train_writer.add_summary(summary, i)
                print("{0}.) Training MSE: {1}".format(i, MSE))


            i0 = (i0 + run_params.batch_size) % data_array.shape[0]#Wrap index if past bounds
            i1 = (i1 + run_params.batch_size) % data_array.shape[0]#Wrap index if past bounds
