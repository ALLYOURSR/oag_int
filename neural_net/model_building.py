import tensorflow as tf
from objects import NeuralNet

def build_net_basic(data_array, run_params):
    # The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
    #  nonlinear function approximator, of the form [input]->[tan-sigmoid hidden layer] -> [linear weighted sum layer] -> output
    # data_array is a 2d array where each row corresponds to a given well and each column corresponds to a given data header
    #   the final column in data_array must be the values of the variable to predict, cum_365_prod for this exercise

    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[
        1]-1])  # input shape is an arbitrary length vector to allow batch training
    output_placeholder = tf.placeholder(dtype=tf.float32)
    tan_sig_layer = tf.contrib.layers.fully_connected(input_placeholder, run_params.num_neurons, activation_fn=tf.nn.sigmoid)
    linear_output_layer = tf.contrib.layers.fully_connected(tan_sig_layer, 1,
                                                  activation_fn=None)  # Specify None for linear activation

    error = tf.sqrt(tf.reduce_mean(tf.square((output_placeholder - linear_output_layer))))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(run_params.train_rate).minimize(error)  # training step

    tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
    tf.summary.tensor_summary("Linear Layer", linear_output_layer)
    tf.summary.scalar("MSE", error)
    summaries = tf.summary.merge_all()

    return NeuralNet(input_placeholder, output_placeholder, linear_output_layer, error, train, summaries)

def build_net_bnorm(data_array, run_params):
    """Improved neural network with batch normalization. Converges to the same MSE range, but significantly more quickly."""
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[
        1]-1])  # input shape is an arbitrary length vector to allow batch training
    output_placeholder = tf.placeholder(dtype=tf.float32)
    bnorm_inp = tf.layers.batch_normalization(input_placeholder)
    tan_sig_layer = tf.contrib.layers.fully_connected(bnorm_inp, run_params.num_neurons, activation_fn=tf.nn.sigmoid)
    bnorm_tan_sig = tf.layers.batch_normalization(tan_sig_layer)
    linear_output_layer = tf.contrib.layers.fully_connected(bnorm_tan_sig, 1,
                                                  activation_fn=None)  # Specify None for linear activation

    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output_placeholder, linear_output_layer))))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(run_params.train_rate).minimize(error)  # training step
    tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
    tf.summary.tensor_summary("Linear Layer", linear_output_layer)
    tf.summary.scalar("MSE", error)
    summaries = tf.summary.merge_all()

    return NeuralNet(input_placeholder, output_placeholder, linear_output_layer, error, train, summaries)

