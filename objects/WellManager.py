import numpy as np
from enum import Enum

class WellManager:
    def __init__(self):
        self._well_metadata = dict()#key is int(api), value is an array
        self._well_data = dict()#key is int(api), value is an array
        #Note: I've separated metadata and numerical data so that I can use the optimized np.array for training

    def get_data(self, api):
        return self._well_data[api] or None

    def get_metadata(self, api):
        return self._well_metadata[api] or None

    def remove_well(self, api):
        if api in self._well_metadata:
            del self._well_metadata[api]
        if api in self._well_data:
            del self._well_data[api]


    def insert_or_update_metadata(self, api, header_to_data:dict):
        if api not in self._well_metadata:
            self._well_metadata[api] = dict()
        self._well_metadata[api].update(header_to_data)

    def insert_or_update_data(self, api, header_to_data:dict):
        if api not in self._well_data:
            self._well_data[api] = dict()
        self._well_data[api].update(header_to_data)

    def get_data_array(self, headers_to_write, specified_apis=None):
        """creates numpy array with specified, ordered iterateable collection headers_to_write for tensorflow input
        if well_apis is not None, only the specified apis will be selected; else all wells are used
        if include_api, first column in returned array corresponds to API
        """
        num_cols = len(headers_to_write)
        num_rows = len(specified_apis) if specified_apis is not None else len(self._well_data)
        out_arr = np.zeros([num_rows, num_cols])



        if specified_apis is None:
            specified_apis=self._well_data.keys()


        row = 0
        for w in self.specified_apis:
            column = 0
            for h in headers_to_write:
                out_arr[row, column] = self._well_data[w][h]
                column += 1
            row += 1

        return out_arr