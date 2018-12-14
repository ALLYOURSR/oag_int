from .baseconfig import BaseConfig
from .header_enum import HeaderTypes

#Config file to keep track of which headers are relevant for each data file
class ProdByOperatedDayConfig(BaseConfig):
    def __init__(self, filename):
        super().__init__(filename)
        self.metadata_headers = {HeaderTypes.api}
        self.data_headers = {HeaderTypes.cum_365_prod}
