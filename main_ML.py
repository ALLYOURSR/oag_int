from config import *
from config import NeuralNetTypes
from objects import WellManager
from parsing import parse_file
from neural_net import *
from utils import plot_data
from neural_net import NeuralNetFactory

#Instantiate config objects which specify how to parse data files
global_config = GlobalConfig()
completion_config = CompletionConfig(global_config.completion_filename)
prod_by_operated_day_config = ProdByOperatedDayConfig(global_config.production_data_filename)
#well_index_config = WellIndexConfig(global_config.well_index_filename)

configs_to_parse = [completion_config,
                 prod_by_operated_day_config,
                 ]

well_manager = WellManager()#Instantiate WellManager which acts as a store for all well data
net_factory = NeuralNetFactory()


#Keep track of incomplete wells to prune after parsing all files
skipped_apis = set()

#Parse files
for c in configs_to_parse:
    skipped_apis.update(parse_file(c, well_manager, global_config.data_directory))

for s in skipped_apis:
    well_manager.remove_well(s)

print("{0} skipped, {1} remaining, {2} parsed".format(len(skipped_apis), len(well_manager.get_apis()), len(skipped_apis) + len(well_manager.get_apis())))



#Train and test multiple nets
all_runs = [RunParams("single", {"neural_net_type":NeuralNetTypes.Basic, "num_neurons_per_layer":2}),
            RunParams("batch_normalized", {"neural_net_type":NeuralNetTypes.BatchNormalized}),
            RunParams("dual_layer_4", {"neural_net_type": NeuralNetTypes.Dual_Layer_Basic, "num_neurons_per_layer": 8}),
            RunParams("dual_layer_8", {"neural_net_type": NeuralNetTypes.Dual_Layer_Basic, "num_neurons_per_layer": 4})
           ]

for r in all_runs:
    r.write_params_to_file()
    data_arr = well_manager.get_data_array(r.headers_to_evaluate)
    neural_net = net_factory.build_net(r.neural_net_type, data_arr.shape[1]-1, r)
    last_mse = train_net(data_arr, neural_net, r)
    r.write_results_to_file(last_mse)








