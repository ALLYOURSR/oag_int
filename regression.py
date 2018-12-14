
import tensorflow as tf

def build_net(num_inputs, num_layers):
    #The following net is based on a simple, dual layer neural network designed as a general purpose multivariable
    #  nonlinear function approximator, of the form [input]->[tan-sigmoid layer] -> [linear layer] -> output

    input = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs]) #input shape is an arbitrary length vector to allow batch training
    tan_sig_layer = tf.contrib.layers.fully_connected(input, activation_fn=tf.nn.sigmoid)
    lin_layer = tf.contrib.layers.fully_connected(tan_sig_layer, biases_initializer=tf.initializers.he_normal)

    error = tf.reduce_mean(tf.square(Y - y_model))  # cost function, mse
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error)  # training step

    with tf.Session() as sess:
        tf.global_variables_initializer().run() #initialize all tensorflow variables
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







def nonlinear_regression():
    """Perform nonlinear regression training."""

    trX, trY = nonlinear_data() #imported from a local CSV file

    X = tf.placeholder(dtype="float") #placeholder for feeding x values
    Y = tf.placeholder(dtype="float") #placeholder for feeding y values

    #all variables initialized to 0.1
    A1 = tf.Variable(0.1, name="weights") #first parameter
    A2 = tf.Variable(0.1, name="weights") #second parameter
    x0 = tf.Variable(0.1, name="weights") #third parameter
    dx = tf.Variable(0.1, name="weights") #fourth parameter
    y_model = A2 + tf.divide(A1-A2,1+tf.exp((X-x0)/dx)) #Boltzmann Sigmoid
    error = tf.reduce_mean(tf.square(Y - y_model)) #cost function, mse
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error) #training step

    with tf.Session() as sess:
        tf.global_variables_initializer().run() #initialize all tensorflow variables
        errors = [0, 1] # error list to check for convergence.
        for i in range(500): #training episodes
            for x, y in zip(trX[::i+1], trY[::i+1]): #+1 for speed-up towards end of training, not necessary
                sess.run(train, feed_dict={X: x, Y: y})
                MSE = sess.run(error, feed_dict={X: x, Y: y})
            if i % 2 == 0:
                errors[0] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i,MSE)
            else:
                errors[1] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i,MSE)
            if np.isclose(errors[0], errors[1]):
                print("Done Training!") #regression has converged
                break

    #return finalized parameters
    A1 = sess.run(A1) #final first parameter
    A2 = sess.run(A2) #final second parameter
    x0 = sess.run(x0) #final third parameter
    dx = sess.run(dx) #final fourth parameter
    error = sess.run(error, feed_dict={X: x, Y: y}) #final error
    return A1, A2, x0, dx, error

def linear_regression():
    """Perform linear regression training."""

    trX, trY = linear_data() #defines our pseudo-linear toy data

    X = tf.placeholder(dtype="float") #placeholder for feeding x values
    Y = tf.placeholder(dtype="float") #placeholder for feeding y values

    #initialize weight and bias variables at 0.0.
    #these will change during training in order to minimize the error
    #produced by our 'cost' function
    m = tf.Variable(0.0, name="weights") #weight variable (~ slope)
    b = tf.Variable(0.0, name="bias") #bias variable (~ intercept)
    y_model = tf.multiply(m, X) + b #y = m * X + b equation of a line
    error = tf.reduce_mean(tf.square(Y - y_model)) #cost function, calculates the mean-square-error
    train = tf.train.GradientDescentOptimizer(0.01).minimize(error) #training step, gradient descent is used to minimize the error between the model and the experimental data

    with tf.Session() as sess:
        tf.global_variables_initializer().run() #initialize all tensorflow variables
        errors = [0, 1] #error list to check for convergence.
        for i in range(100): #training episodes
            for x, y in zip(trX, trY): #for all x and y values
                sess.run(train, feed_dict={X: x, Y: y}) #feed data into model
                MSE = sess.run(error, feed_dict={X: x, Y: y}) #evaluate error
            if i % 2 == 0: #when i is even, set errors[0]
                errors[0] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i,MSE) #print error
            else: #when i is odd, set errors[1]
                errors[1] = sess.run(error, feed_dict={X: x, Y: y})
                print("{0}.) Training MSE: {1}".format(i,MSE) #print error
            if np.isclose(errors[0], errors[1]): #check if errors[0] == errors[1]
                print("Done Training!") #regression has converged
                break

    #return finalized parameters
    m = sess.run(m) #return fitted slope
    b = sess.run(b) #return fitted intercept
    error = sess.run(error, feed_dict={X: x, Y: y}) #return final error
    return m, b, error
