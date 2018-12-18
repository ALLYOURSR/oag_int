import tensorflow as tf
from objects import NeuralNet
from enums import NeuralNetTypes

class NeuralNetFactory:
    def __init__(self):
        pass

    def build_net(self, neural_net_type:NeuralNetTypes, num_inputs, run_params):
        """Instantiates and returns a neural net"""
        if neural_net_type is NeuralNetTypes.Basic:
            return self._build_basic(num_inputs, run_params)
        elif neural_net_type is NeuralNetTypes.BatchNormalized:
            return self._build_bnorm(num_inputs, run_params)
        elif neural_net_type is NeuralNetTypes.Dual_Layer_Basic:
            return self._build_dual_layer(num_inputs, run_params)
        else:
            raise NotImplementedError("Neural net type {0} not implemented!".format(run_params.neural_net_type))

    def _build_basic(self, num_inputs, run_params):
        # The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
        #  nonlinear function approximator, of the form [input]->[tan-sigmoid hidden layer] -> [linear weighted sum layer] -> output
        # data_array is a 2d array where each row corresponds to a given well and each column corresponds to a given data header
        #   the final column in data_array must be the values of the variable to predict, cum_365_prod for this exercise

        graph = tf.Graph()
        with graph.as_default() as g:
            with g.name_scope("basic_net") as scope: #TODO: add unique naming scheme to allow for multiple nets of this type
                input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs])  # input shape is an arbitrary length vector to allow batch training
                output_placeholder = tf.placeholder(dtype=tf.float32)
                tan_sig_layer = tf.contrib.layers.fully_connected(input_placeholder, run_params.num_neurons_per_layer, activation_fn=tf.nn.sigmoid)
                linear_output_layer = tf.contrib.layers.fully_connected(tan_sig_layer, 1,
                                                              activation_fn=None)  # Specify None for linear activation

                error = tf.sqrt(tf.reduce_mean(tf.square((output_placeholder - linear_output_layer))))  # cost function, mse
                train = tf.train.GradientDescentOptimizer(run_params.train_rate).minimize(error)  # training step

                tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
                tf.summary.tensor_summary("Linear Layer", linear_output_layer)
                tf.summary.scalar("MSE", error)
                summaries = tf.summary.merge_all()

        return NeuralNet(input_placeholder, output_placeholder, linear_output_layer, error, train, summaries, graph, scope)

    def _build_dual_layer(self, num_inputs, run_params):
        # Two hidden layers
        graph = tf.Graph()
        with graph.as_default() as g:
            with g.name_scope("dual_layer") as scope:
                input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs])
                bnorm_inp = tf.layers.batch_normalization(input_placeholder)
                output_placeholder = tf.placeholder(dtype=tf.float32)
                tan_sig_layer0 = tf.contrib.layers.fully_connected(bnorm_inp, run_params.num_neurons_per_layer, activation_fn=tf.nn.sigmoid)
                tan_sig_layer1 = tf.contrib.layers.fully_connected(tan_sig_layer0, run_params.num_neurons_per_layer,
                                                                  activation_fn=tf.nn.sigmoid)
                linear_output_layer = tf.contrib.layers.fully_connected(tan_sig_layer1, 1, activation_fn=None)

                error = tf.sqrt(tf.reduce_mean(tf.square((output_placeholder - linear_output_layer))))
                train = tf.train.GradientDescentOptimizer(run_params.train_rate).minimize(error)

                tf.summary.tensor_summary("Tan Sigmoid0", tan_sig_layer0)
                tf.summary.tensor_summary("Tan Sigmoid1", tan_sig_layer1)
                tf.summary.tensor_summary("Linear Layer", linear_output_layer)
                tf.summary.scalar("MSE", error)
                summaries = tf.summary.merge_all()

        return NeuralNet(input_placeholder, output_placeholder, linear_output_layer, error, train, summaries, graph, scope)

    def _build_bnorm(self, num_inputs, run_params):
        """Improved neural network with batch normalization. Converges to the same MSE range, but significantly more quickly."""
        graph = tf.Graph()
        with graph.as_default() as g:
            with g.name_scope("bnorm_net") as scope:
                input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs])
                output_placeholder = tf.placeholder(dtype=tf.float32)
                bnorm_inp = tf.layers.batch_normalization(input_placeholder)
                tan_sig_layer = tf.contrib.layers.fully_connected(bnorm_inp, run_params.num_neurons_per_layer, activation_fn=tf.nn.sigmoid)
                bnorm_tan_sig = tf.layers.batch_normalization(tan_sig_layer)
                linear_output_layer = tf.contrib.layers.fully_connected(bnorm_tan_sig, 1,
                                                              activation_fn=None)

                error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output_placeholder, linear_output_layer))))
                train = tf.train.GradientDescentOptimizer(run_params.train_rate).minimize(error)
                tf.summary.tensor_summary("Tan Sigmoid", tan_sig_layer)
                tf.summary.tensor_summary("Linear Layer", linear_output_layer)
                tf.summary.scalar("MSE", error)
                summaries = tf.summary.merge_all()

        return NeuralNet(input_placeholder, output_placeholder, linear_output_layer, error, train, summaries, graph, scope)

