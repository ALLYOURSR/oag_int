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
    input_placeholder, output_placeholder, error, train, summaries = build_net_basic(in_arr, run_params.num_neurons)
elif run_params.neural_net_type is NeuralNetTypes.BatchNormalized:
    input_placeholder, output_placeholder, error, train, summaries = build_net_bnorm(in_arr, run_params.num_neurons)
else:
    raise NotImplementedError("Neural net type {0} not implemented!".format(run_params.neural_net_type))

train_net(in_arr, input_placeholder, output_placeholder, error, train, summaries, run_params.batch_size, run_params.num_training_steps)

