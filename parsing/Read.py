from objects import WellManager, Well
from os.path import join
from objects import HeaderTypes
from objects import TimeSeriesHeaderTypes as Tsht
from config import BaseConfig
from datetime import datetime
from utils import update_progress

def _get_column_indices(header_line, headers):
    hl_split = header_line.lower().replace("\"", "").split(',')
    header_to_column = dict()

    for h in headers:
        header_to_column[h] = hl_split.index(h.value)


    return header_to_column

def _get_temporary_lists(headers):
    all_arrays = dict()
    for h in headers:
        all_arrays[h] = []#Using a list because np arrays are contiguous and adding even one element copies the entire array

    return all_arrays

def _write_to_well(well_manager:WellManager, api, lists):
    w = well_manager.get_or_create_well(api)
    for i in lists.items():
        w.add_or_append_series(i[0], lists[Tsht.date], i[1])#Currently storing the dates redundantly for each series, this may be optimizable with some loss of generality

def parse_time_series(config:BaseConfig, well_manager:WellManager, data_directory:str):
    filepath = join(data_directory, config.filename)
    # Parse header
    # Keeping this generic, in case column orders change in subsequent data files
    infile = open(filepath)
    header_line = infile.readline()

    try:
        header_to_column = _get_column_indices(header_line, config.headers_to_read + [Tsht.api])
    except KeyError:
        raise LookupError("Header value specified in config was not found in data file!")


    current_api = -1
    temp_lists = _get_temporary_lists(config.headers_to_read)

    all_lines = infile.readlines()
    total_lines = len(all_lines)
    parsed_lines = 0

    # Parse data
    for l in all_lines:
        sp = l.replace("\"", "").split(',')
        api = sp[header_to_column[Tsht.api]]
        if api != current_api:#Write in large batches to minimize resizing of underlying numpy array
            _write_to_well(well_manager, api, temp_lists)

            temp_lists = _get_temporary_lists(config.headers_to_read)
            current_api = api
        try:
            date = datetime.strptime(sp[header_to_column[Tsht.date]], '%Y-%m-%d %H:%M:%S').timestamp() #Convert to seconds since linux epoch. Ignoring timezones
            oil_barrels = float(sp[header_to_column[Tsht.oil_barrels]])
            water_barrels = float(sp[header_to_column[Tsht.water_barrels]])
            gas_mcf = float(sp[header_to_column[Tsht.gas_mcf]])
        except ValueError:
            continue

        temp_lists[Tsht.water_barrels].append(water_barrels)
        temp_lists[Tsht.date].append(date)
        temp_lists[Tsht.oil_barrels].append(oil_barrels)
        temp_lists[Tsht.gas_mcf].append(gas_mcf)

        parsed_lines += 1
        if parsed_lines%100 == 0:
            update_progress(parsed_lines/total_lines)

    _write_to_well(well_manager, api, temp_lists)


def parse_file(config:BaseConfig, well_manager:WellManager, data_directory:str):
    """Parses the given file, filling well_manager with well data
    Does not write data if any data or metadata columns are null
    returns: set of skipped apis"""
    incomplete_well_apis=set()
    total_parse_count = 0

    filepath = join(data_directory, config.filename)

    #Parse header
    #Keeping this generic, in case column orders change in subsequent data files
    infile = open(filepath)
    header_line = infile.readline()

    all_headers = set()
    all_headers.add(config.api_header)
    all_headers.update(config.metadata_headers)
    all_headers.update(config.data_headers)

    try:
        header_to_column = _get_column_indices(header_line, all_headers)
    except KeyError:
        raise LookupError("Header value specified in config was not found in data file!")

    #Parse data
    for l in infile.readlines():
        sp = l.split('\",\"')

        # Remove starting and trailing quotes which are parsing artifacts
        sp[0].replace("\"", "")
        sp[-1].replace("\"", "")


        api = sp[header_to_column[HeaderTypes.api]]

        parsed_meta_vals = dict()
        total_parse_count += 1
        skip = False
        for h in config.metadata_headers:
            parsed_meta_vals[h] = sp[header_to_column[h]]

        parsed_data_vals = dict()
        for h in config.data_headers:
            try:
                parsed_data_vals[h] = float(sp[header_to_column[h]])
            except ValueError:
                incomplete_well_apis.add(api)
                skip = True
                break

        if not skip:
            well_manager.insert_or_update_metadata(api, parsed_meta_vals)
            well_manager.insert_or_update_data(api, parsed_data_vals)

    print("Skipped {0} of {1} for file {2}".format(len(incomplete_well_apis), total_parse_count, filepath))

    return incomplete_well_apis