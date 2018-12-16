from os.path import join
from .headertypes import HeaderTypes
from .NeuralNetTypes import NeuralNetTypes

class RunParams:
    def __init__(self):
        self.records_dir = "~/Projects/oag-int/run_records/"
        self.data_dir = "~/Projects/oag-int/oag-tech-interview-data/"
        self.run_name = "basic"
        self.num_neurons_per_layer = 8
        self.batch_size = 100
        self.num_training_steps = 100000
        self.neural_net_type = NeuralNetTypes.Basic
        self.train_rate = .1
        self.log_period = 50 #Steps between logging

        self.headers_to_evaluate = [#HeaderTypes.foot_per_stage,
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

    def write_params_to_file(self):
        outstr = ""
        #Quick and dirty logs
        for attr, value in self.__dict__.items():
            outstr += "{0}:{1}\n".format(attr, value)

        filename = "{0}_params".format(self.run_name)
        f = open(join(self.records_dir, filename), "w")
        f.write(outstr)

    def write_to_file(self):
        """Writes the run parameters to a file so that runs can be reproduced and tweaked"""
        # TODO
        pass
