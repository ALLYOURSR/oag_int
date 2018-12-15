import tensorflow as tf

class NeuralNet:
    #Careful, the current implementation does not allow for multiple neural nets. Not trivial to do so in tf, will convert
    #as time permits
    def __init__(self, input_placeholder, output_placeholder, linear_output_layer, error_tensor, train_tensor, summaries):
        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        self.error_tensor = error_tensor
        self.train_tensor = train_tensor
        self.summaries = summaries
        self.linear_output_layer = linear_output_layer

    def get_value(self, indata):
        """"Runs the neural net on indata, an [n, v] shaped array of input variables, where num columns v = number of variables.
         Returns an array of length n"""
        with tf.Session() as sess:
            return sess.run(self.linear_output_layer, feed_dict={self.input_placeholder: indata})
