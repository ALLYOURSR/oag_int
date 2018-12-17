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


run_params = RunParams('default')
run_params.write_params_to_file()#Save a record of the run for iteration later

data_arr = well_manager.get_data_array(run_params.headers_to_evaluate)
#plot_data(in_arr, run_params.headers_to_evaluate)

neural_net = net_factory.build_net(run_params.neural_net_type, data_arr.shape[1]-1, run_params)

train_net(data_arr, neural_net, run_params)


