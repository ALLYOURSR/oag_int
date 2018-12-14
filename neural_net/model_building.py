import tensorflow as tf

def train_net(data_array, num_neurons, batch_size, num_training_runs):
    #The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
    #  nonlinear function approximator, of the form [input]->[tan-sigmoid hidden layer] -> [linear weighted sum layer] -> output
    #data_array is a 2d array where each row corresponds to a given well and each column corresponds to a given data header
    #   the final column in data_array must be the values of the variable to predict, cum_365_prod for this exercise

    inp = tf.placeholder(dtype=tf.float32, shape=[None, data_array.shape[1]]) #input shape is an arbitrary length vector to allow batch training
    correct_output = tf.placeholder(dtype=tf.float32)
    tan_sig_layer = tf.contrib.layers.fully_connected(inp, activation_fn=tf.nn.sigmoid)

    weights = tf.get_variable("lin_layer_weights", [num_neurons], initializer=tf.initializers.he_normal)
    biases = tf.get_variable("lin_layer_biases", [num_neurons], initializer=tf.initializers.he_normal)
    lin_layer = tf.add(tf.multiply(weights, tan_sig_layer), biases) #w*a + b, simple linear activation weighted sum

    error = tf.reduce_mean(tf.square(lin_layer - correct_output))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error)  # training step

    with tf.Session() as sess:
        i0 = 0
        i1 = batch_size
        tf.global_variables_initializer().run() #initialize variables
        for i in range(num_training_runs): #training episodes
            in_arr = data_array[i0:i1,:data_array.shape[0]-1] #grab all but the last column
            out_arr = data_array[i0:i1, -1] #grab last column

            sess.run(train, feed_dict={inp: in_arr, correct_output: out_arr})
            MSE = sess.run(error, feed_dict={inp: in_arr, correct_output: out_arr})

            if i % 50 == 0:
                print("{0}.) Training MSE: {1}".format(i,MSE))

            i0 = (i0 + batch_size) % data_array.shape[0]#Wrap index if past bounds
            i1 = (i1 + batch_size) % data_array.shape[0]#Wrap index if past bounds
