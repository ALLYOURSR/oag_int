import numpy as np
from enums.headertypes import TimeSeriesHeaderTypes
from datetime import timedelta

class Well:
    def __init__(self, api):
        self.api = api
        self._series_data = dict()#[key is header, value is [n,2] array of (date, value) pairs]
        self._well_metadata = dict()  # key is header, value is an array
        self._well_data = dict()  # key is header, value is an array
        # Note: I've separated metadata and numerical data so that I can use the optimized np.array for training

    def add_or_append_series(self, header, dates, values):
        assert(len(dates) == len(values))
        if len(dates) == 0:
            return

        arr = np.zeros([len(dates), 2])
        arr[:,0] = dates
        arr[:,1] = values
        if header not in self._series_data:
            self._series_data[header] = arr
        else:
            self._series_data[header] = np.concatenate([self._series_data[header], arr], 0)


    def add_or_update_metadata(self, header_to_data:dict):
        self._well_metadata.update(header_to_data)

    def add_or_update_data(self, header_to_data:dict):
        self._well_data.update(header_to_data)

    def sort_time_series(self):
        for s in self._series_data.items():
            self._series_data[s[0]] = s[1][np.argsort(s[1][:, 0])]

    def get_cumulative_value(self, header:TimeSeriesHeaderTypes, timespan:timedelta):
        """start_time is time measured relative to earliest data point
        if start_time + timespan > series length, aggregates over full time series
        returns None if header does not exist or spud_date is not set"""

        if header not in self._series_data:
            return None

        data_series = self._series_data[header]

        #All times below are converted to seconds from unix epoch
        start_date = self._series_data[header][0, 0]

        start_index = np.argmax(data_series[:,0] > (start_date + timespan.total_seconds()))#Get index of first element after start_time
        stop_index = np.argmax(data_series[:,0] > start_date + timespan.total_seconds() + timespan.total_seconds())

        if stop_index == 0:#start_time + timespan > last data date
            stop_index = len(data_series[0])

        return np.sum(data_series[start_index:stop_index, 1])
