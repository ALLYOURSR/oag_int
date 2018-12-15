import matplotlib.pyplot as plt
import math

def plot_data(data_array, headers=None):
    #Simple method for a quick glance at which variables may influence cum_prod_365
    #Headers should be in the same order as data_array
    num_plots = data_array.shape[1]-1
    ncols = math.ceil(num_plots/2)

    f, ax = plt.subplots(nrows=2, ncols=ncols)
    ax = ax.flatten()
    for i in range(num_plots):
        ax[i].scatter(data_array[:,i], data_array[:,-1])
        if headers is not None:
            ax[i].set_title(headers[i].name)

    plt.show()