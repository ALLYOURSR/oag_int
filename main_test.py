from config import *
from config import NeuralNetTypes
from testing import *
from neural_net import *
from visualization import plot_data


#Instantiate config objects which specify how to parse data files
global_config = GlobalConfig()

run_params = RunParams()
run_params.write_to_file()#Save a record of the run for iteration later


in_arr = get_test_data()


if run_params.neural_net_type is NeuralNetTypes.Basic:
    neural_net = build_net_basic(in_arr, run_params)
elif run_params.neural_net_type is NeuralNetTypes.BatchNormalized:
    neural_net = build_net_bnorm(in_arr, run_params)
else:
    raise NotImplementedError("Neural net type {0} not implemented!".format(run_params.neural_net_type))

train_net(in_arr, neural_net, run_params)

