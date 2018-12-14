from .header_enum import HeaderTypes


class RunParams:
    def __init__(self):
        self.num_neurons = 15
        self.batch_size=1000
        self.num_training_steps = 10000

        self.headers_to_evaluate = {HeaderTypes.foot_per_stage,
                                    HeaderTypes.fluid_barrels,
                                    HeaderTypes.foot_per_stage,
                                    HeaderTypes.lateral_length,
                                    HeaderTypes.max_treat_pressure,
                                    HeaderTypes.proppant_pounds,
                                    HeaderTypes.proppant_pounds_per_foot,
                                    HeaderTypes.stages,

                                    # Note: For now, make sure the last element is cum_365_prod,
                                    # as the training algorithm uses the last column from WellManager.get_data_array
                                    # to calculate error!
                                    HeaderTypes.cum_365_prod
                                    }

    def write_to_file(self):
        """Writes the run parameters to a file so that runs can be reproduced and tweaked"""
        # TODO
        pass