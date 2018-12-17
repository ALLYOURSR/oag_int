from config import *
from objects import WellManager, HeaderTypes, TimeSeriesHeaderTypes
from parsing import parse_file, parse_time_series
from datetime import datetime, timedelta

#Instantiate config objects which specify how to parse data files
global_config = GlobalConfig()
prod_by_operated_day_config = ProdByOperatedDayConfig(global_config.production_data_filename)
well_index_config = WellIndexConfig(global_config.well_index_filename)
well_index_config.metadata_headers = [HeaderTypes.spud_date]
monthly_production_config = MonthlyProductionConfig(global_config.monthly_production_filename)

well_manager = WellManager()#Instantiate WellManager which acts as a store for all well data


#Keep track of incomplete wells to prune after parsing all files
skipped_apis = set()

for s in skipped_apis:
    well_manager.remove_well(s)

#Parse time series data files
print("Parsing file {0}".format(monthly_production_config.filename))
parse_time_series(monthly_production_config, well_manager, global_config.data_directory)


#Prune wells without spud dates or time series data
apis_to_prune = []
for api in well_manager.get_apis():
    w = well_manager.get_well(api)
    if len(w._series_data.keys()) == 0: #TODO: remove private member access
        apis_to_prune.append(api)

print("Pruning {0} wells missing time series data".format(len(apis_to_prune)))
[well_manager.remove_well(api) for api in apis_to_prune]

#Sort all the time series, in case they're out of order after parsing
[well_manager.get_well(api).sort_time_series() for api in well_manager.get_apis()]

count = 0
for api in well_manager.get_apis():
    lll = len(well_manager.get_well(api)._series_data[TimeSeriesHeaderTypes.oil_barrels])
    if lll < 10:
        count += 1

print("{0} of {1}".format(count, len(well_manager.get_apis())))


for api in well_manager.get_apis():
    w = well_manager.get_well(api)
    #For this toy example, we'll calculate cumulative oil production over 365 days for each well
    start_time = timedelta(0)
    timespan = timedelta(365)
    val = w.get_cumulative_value(TimeSeriesHeaderTypes.oil_barrels, timespan)
    if val is not None:
        print("API: {0} cum_oil_365: {1}".format(w.api, val))
print("Done")
