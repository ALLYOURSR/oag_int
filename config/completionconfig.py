from .baseconfig import BaseConfig
from objects.headertypes import HeaderTypes

#Config file to keep track of which headers are relevant for each data file
class CompletionConfig(BaseConfig):
    def __init__(self, filename):
        super().__init__(filename)
        self.metadata_headers = {HeaderTypes.api}
        self.data_headers = {HeaderTypes.fluid_barrels,
                             HeaderTypes.stages,
                             HeaderTypes.foot_per_stage,
                             HeaderTypes.lateral_length,
                             HeaderTypes.max_treat_pressure,
                             HeaderTypes.proppant_pounds,
                             HeaderTypes.proppant_pounds_per_foot,
                             HeaderTypes.max_treat_pressure}