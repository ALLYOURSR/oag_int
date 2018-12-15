from .headertypes import HeaderTypes
from .baseconfig import BaseConfig

#Config file to keep track of which headers are relevant for each data file
class WellIndexConfig(BaseConfig):
    def __init__(self, filename):
        super().__init__(filename)
        self.metadata_headers = {HeaderTypes.api,
                        HeaderTypes.field_name}
