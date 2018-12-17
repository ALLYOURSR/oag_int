import numpy as np
from enum import Enum
from objects import Well

class WellManager:
    def __init__(self):
        self._wells = dict()#Key is api, value is Well object

    def get_well(self, api):
        return self._wells[api] if api in self._wells else None

    def get_or_create_well(self, api):
        if api in self._wells:
            w = self._wells[api]
        else:
            w = Well(api)
            self._wells[api] = w

        return w

    def get_apis(self):
        return self._wells.keys()

    def remove_well(self, api):
        if api in self._wells:
            del self._wells[api]

    def add_or_append_series(self, api, header, dates, values):
        if api not in self._wells:
            self._wells[api] = Well(api)
        self._wells[api].add_or_append_series(header, dates, values)

    def insert_or_update_metadata(self, api, header_to_data:dict):
        if api not in self._wells:
            self._wells[api] = Well(api)
        self._wells[api].add_or_update_metadata(header_to_data)

    def insert_or_update_data(self, api, header_to_data:dict):
        if api not in self._wells:
            self._wells[api] = Well(api)
        self._wells[api].add_or_update_data(header_to_data)

    def get_data_array(self, headers_to_write, specified_apis=None):
        """creates numpy array with specified, ordered iterateable collection headers_to_write for tensorflow input
        if well_apis is not None, only the specified apis will be selected; else all wells are used
        if include_api, first column in returned array corresponds to API
        """
        num_cols = len(headers_to_write)
        num_rows = len(specified_apis) if specified_apis is not None else len(self._wells)
        out_arr = np.zeros([num_rows, num_cols])
        num_skipped = 0


        if specified_apis is None:
            specified_apis=self._wells.keys()


        row = 0
        for api in specified_apis:
            w = self._wells[api]
            column = 0
            skip = False
            for h in headers_to_write:
                #Verify that the well has data from all the specified headers
                if h not in w._well_data:
                    skip = True
                    num_skipped += 1

            if not skip:
                for h in headers_to_write:
                    out_arr[row, column] = w._well_data[h]
                    column += 1
                row += 1

        #Truncate empty rows
        out_arr = out_arr[:out_arr.shape[0]-num_skipped]

        return out_arr