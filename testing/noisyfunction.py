import numpy as np

def get_test_data():
    #Generate a noisy, multivariable dataset to ensure that the neural net is capable of learning
    #function f = aw^4 + bx^3 + cy^2 + z

    num_rows = 50000
    a = 2
    b = .5
    c = 3

    noise_magnitude = .1

    dom = np.random.rand(num_rows,4) #Initiate with random values for each variable
    f = np.zeros([num_rows,1])
    perturbed = np.zeros([num_rows,1])


    for i in range(dom.shape[0]):
        w,x,y,z = dom[i,0], dom[i,1], dom[i,2], dom[i,3]

        #f[i] = w * (a * np.power(np.e, x) + b*np.power(y, 2) + c*z)
        f[i] = a * np.power(w, 4) + b * np.power(x,3) + c * np.power(y, 2) + z
        perturbed[i] = f[i] + np.random.normal(0, noise_magnitude)


    out_arr = np.concatenate([dom, perturbed], 1)

    return out_arr