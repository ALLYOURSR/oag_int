from enums import TimeSeriesHeaderTypes
from .baseconfig import BaseConfig

class MonthlyProductionConfig(BaseConfig):
    def __init__(self, filename):
        super().__init__(filename)
        self.headers_to_read = [
            TimeSeriesHeaderTypes.date,
            TimeSeriesHeaderTypes.oil_barrels,
            TimeSeriesHeaderTypes.water_barrels,
            TimeSeriesHeaderTypes.gas_mcf,
    ]
