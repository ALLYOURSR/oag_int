import tensorflow as tf

def train_net(input_placeholder, output_tensor):
    """Normally, I would prefer to explicitly pass objects for training, like the graph in tensorflow
    tensorflow, however, keeps track of a default_graph object automatically, which is fine so long as only one
    default graph is needed"""
    error = tf.reduce_mean(tf.square(Y - y_model))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error)  # training step

    with tf.Session() as sess:
        tf.global_variables_initializer().run()  # initialize all tensorflow variables
        errors = [0, 1]  # error list to check for convergence.
        for i in range(500):  # training episodes
            for x, y in zip(trX[::i + 1], trY[::i + 1]):  # +1 for speed-up towards end of training, not necessary
                sess.run(train, feed_dict={X: x, Y: y})
                MSE = sess.run(error, feed_dict={X: x, Y: y})
            if i % 2 == 0:
                errors[0] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i, MSE)
                else:
                errors[1] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i, MSE)
                if np.isclose(errors[0], errors[1]):
                    print("Done Training!")  # regression has converged
                break