from enums.headertypes import HeaderTypes

#TODO: probably remove this
class BaseConfig:
    def __init__(self, filename):
        self.filename = filename
        self.data_headers = {}#Data headers must be numerical columns
        self.metadata_headers = {}#Metadata headers are read and stored as strings
        self.api_header = HeaderTypes.api#API must be an int parseable string