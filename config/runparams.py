from os.path import join
from .headertypes import HeaderTypes
from .NeuralNetTypes import NeuralNetTypes

class RunParams:
    def __init__(self, run_name, dict_params):
        default_vals = {
            'records_dir': "/home/x/Projects/oag_int/run_records/",
            'data_dir': "/home/x/Projects/oag_int/oag-tech-interview-data/",
            'run_name': run_name,
            'num_neurons_per_layer': 8,
            'batch_size': 200,
            'num_training_steps': 1000000,
            'neural_net_type': NeuralNetTypes.Basic,
            'train_rate': .1,
            'log_period': 100, #Steps between logging

            'headers_to_evaluate': [
                                    #HeaderTypes.foot_per_stage,
                                    #HeaderTypes.fluid_barrels,
                                    HeaderTypes.lateral_length,
                                    HeaderTypes.max_treat_pressure,
                                    HeaderTypes.proppant_pounds,
                                    #HeaderTypes.proppant_pounds_per_foot,
                                    HeaderTypes.stages,

                                    # Note: For now, make sure the last element is cum_365_prod,
                                    # as the training algorithm uses the last column from WellManager.get_data_array
                                    # to calculate error!
                                    HeaderTypes.cum_365_prod
                                  ]
        }
        for i in default_vals.items():
            setattr(self, i[0], default_vals.get(i[0], i[1]))

        for i in dict_params.items():
            if i[0] in default_vals:
                setattr(self, i[0], dict_params.get(i[0], i[1]))

    def write_params_to_file(self):
        outstr = ""
        #Quick and dirty log
        for attr, value in self.__dict__.items():
            outstr += "{0}:{1}\n".format(attr, value)

        filename = "{0}_params".format(self.run_name)
        f = open(join(self.records_dir, filename), "w")
        f.write(outstr)

    #TODO: Move this to a more intuitive location
    def write_results_to_file(self, mse):
        outstr = str(mse)

        filename = "{0}_results".format(self.run_name)
        f = open(join(self.records_dir, filename), "w")
        f.write(outstr)
