from config import *
from objects import WellManager
from parsing import parse_file
from neural_net import train_net


#Instantiate config objects which specify how to parse data files
global_config = GlobalConfig()
completion_config = CompletionConfig(global_config.completion_filename)
prod_by_operated_day_config = ProdByOperatedDayConfig(global_config.production_data_filename)
well_index_config = WellIndexConfig(global_config.well_index_filename)

configs_to_parse = [completion_config,
                 prod_by_operated_day_config,
                 well_index_config]

#Instantiate WellManager which acts as a store for all well data
well_manager = WellManager()

#Keep track of incomplete wells to prune after parsing all files
skipped_apis = set()

#Parse files
for c in configs_to_parse:
    skipped_apis.update(parse_file(c, well_manager, global_config.data_directory))

for s in skipped_apis:
    well_manager.remove_well(s)

print("{0} skipped, {1} remaining, {2} parsed".format(len(skipped_apis), len(well_manager._well_metadata), len(skipped_apis) + len(well_manager._well_metadata)))

run_params = RunParams()
run_params.write_to_file()


in_arr = well_manager.get_data_array(run_params.headers_to_evaluate)

train_net(in_arr, run_params.num_neurons, run_params.batch_size, run_params.num_training_steps)

