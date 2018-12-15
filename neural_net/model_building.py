import tensorflow as tf


def build_net_basic(data_array, num_neurons):
    # The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
    #  nonlinear function approximator, of the form [input]->[tan-sigmoid hidden layer] -> [linear weighted sum layer] -> output
    # data_array is a 2d array where each row corresponds to a given well and each column corresponds to a given data header
    #   the final column in data_array must be the values of the variable to predict, cum_365_prod for this exercise

    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[
        1]-1])  # input shape is an arbitrary length vector to allow batch training
    output_placeholder = tf.placeholder(dtype=tf.float32)
    tan_sig_layer = tf.contrib.layers.fully_connected(input_placeholder, num_neurons, activation_fn=tf.nn.sigmoid)
    lin_layer = tf.contrib.layers.fully_connected(tan_sig_layer, 1,
                                                  activation_fn=None)  # Specify None for linear activation

    error = tf.sqrt(tf.reduce_mean(tf.square((output_placeholder - lin_layer))))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error)  # training step

    tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
    tf.summary.tensor_summary("Linear Layer", lin_layer)
    tf.summary.scalar("MSE", error)
    summaries = tf.summary.merge_all()

    return input_placeholder, output_placeholder, error, train, summaries

def build_net_bnorm(data_array, num_neurons):
    """Improved neural network with batch normalization. Converges to the same MSE range, but significantly more quickly."""
    inp = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[
        1]])  # input shape is an arbitrary length vector to allow batch training
    correct_output = tf.placeholder(dtype=tf.float32)
    bnorm_inp = tf.layers.batch_normalization(inp)
    tan_sig_layer = tf.contrib.layers.fully_connected(bnorm_inp, num_neurons, activation_fn=tf.nn.sigmoid)
    bnorm_tan_sig = tf.layers.batch_normalization(tan_sig_layer)
    lin_layer = tf.contrib.layers.fully_connected(bnorm_tan_sig, 1,
                                                  activation_fn=None)  # Specify None for linear activation

    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(correct_output, lin_layer))))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.5).minimize(error)  # training step
    tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
    tf.summary.tensor_summary("Linear Layer", lin_layer)
    tf.summary.scalar("MSE", error)
    summaries = tf.summary.merge_all()

    return error, train, summaries

