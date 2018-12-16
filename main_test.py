from config import *
from config import NeuralNetTypes
from testing import *
from neural_net import *
from visualization import plot_data
import numpy as np

#Instantiate config objects which specify how to parse data files
global_config = GlobalConfig()

run_params = RunParams()
run_params.write_to_file()#Save a record of the run for iteration later


noisy_arr, clean_arr = get_test_data()

neural_net_basic = build_net_basic(noisy_arr.shape[1]-1, run_params)
neural_net_bnorm = build_net_bnorm(noisy_arr.shape[1]-1, run_params)

train_net(noisy_arr, neural_net_basic, run_params)
train_net(noisy_arr, neural_net_bnorm, run_params)

results_basic = neural_net_basic.get_value(clean_arr[:,:-1])
results_bnorm = neural_net_bnorm.get_value(clean_arr[:,:-1])

residuals_basic = np.subtract(clean_arr[:,-1], results_basic.flatten())
residuals_bnorm = np.subtract(clean_arr[:,-1], results_bnorm.flatten())

print(np.mean(residuals_basic))
print(np.mean(residuals_bnorm))
