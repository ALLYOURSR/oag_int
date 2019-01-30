import tensorflow as tf
import os
import numpy as np
from objects import ErrorTracker
import matplotlib.pyplot as plt

def get_batch(data_array, i0, i1):
    """Provides a quick and dirty wrapping functionality"""
    if i0 > i1:
        a0 = data_array[i0:, :]
        a1 = data_array[:i1, :]
        return np.concatenate([a0, a1], 0)
    else:
        return data_array[i0:i1, :]

def train_net(data_array, neural_net, run_params):
    """Trains a net according to the specified run_params on the given data_array, an [n,d+1] array where d is the number of data variables. Last column must be variable to predict"""
    log_dir = '/tmp/oag_int/train/'
    os.system("rm {0} -r".format(log_dir))
    train_writer = tf.summary.FileWriter('/tmp/oag_int/train/',
                                         neural_net.session.graph)

    error_tracker = ErrorTracker(10000)

    i0 = 0
    i1 = run_params.batch_size
    with neural_net.session.as_default():
        with neural_net.graph.as_default():
            tf.global_variables_initializer().run() #initialize variables

    i = 0

    all_errors_1st = []
    all_slopes = []
    while True:
    #for i in range(run_params.num_training_steps):
        batch = get_batch(data_array,i0, i1)
        in_arr = batch[:,:-1] #grab all but the last column
        out_arr = batch[:, -1] #grab last column

        neural_net.session.run(neural_net.train_tensor, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})
        MSE = neural_net.session.run(neural_net.error_tensor, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})
        error_tracker.add_error(MSE)


        if i % run_params.log_period == 0:
            summary = neural_net.session.run(neural_net.summaries, feed_dict={neural_net.input_placeholder: in_arr, neural_net.output_placeholder: out_arr})
            train_writer.add_summary(summary, i)
            current_average = error_tracker.get_current_average()
            slope = error_tracker.get_slope()

            all_errors_1st.append(MSE)
            all_slopes.append(slope)

            print("{:8d}. Training MSE: {:8f} -- Average MSE: {:8f} -- Slope: {:8f}".format(i, MSE, current_average, slope))

            if run_params.num_training_steps is None and slope > 0: #We're "regressing" on large epochs now
                break




        i0 = (i0 + run_params.batch_size) % data_array.shape[0]#Wrap index if past bounds
        i1 = (i1 + run_params.batch_size) % data_array.shape[0]#Wrap index if past bounds
        i += 1

        if run_params.num_training_steps is not None and i == run_params.num_training_steps:
            break

    fig, ax = plt.subplots(1, 3)
    xvals = range(len(all_errors_1st))
    ax[0].plot(xvals, all_errors_1st)
    ax[1].plot(xvals, all_slopes)

    ax[0].set_ylim(0, 120000)
    ax[1].set_ylim(-25, 5)

    ax[0].set_title("1st Order MSE")
    ax[2].set_title("Slope")

    return error_tracker.get_current_average()