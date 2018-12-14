from objects import WellManager, Well
from os.path import join
from config import HeaderTypes, BaseConfig


def _get_column_indices(header_line, headers):
    hl_split = header_line.lower().replace("\"", "").split(',')
    header_to_column = dict()

    for h in headers:
        header_to_column[h] = hl_split.index(h.value)


    return header_to_column

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
        sp = l.replace("\"", "").split(',')
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