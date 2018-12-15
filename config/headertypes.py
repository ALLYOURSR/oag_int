from enum import Enum

#Single, convenient location for modifying header strings to search for, in case they change in other data sets
#Also limits potential for typo induced errors by defining header title strings in one place
class HeaderTypes(Enum):
    api = "api"
    fluid_barrels = "fluid_bbl"
    foot_per_stage = "ft_per_stage"
    lateral_length = "lateral_length"
    max_treat_pressure = "max_treat_press"
    field_name = "field_name"
    cum_365_prod = "cum_oil_365"
    proppant_pounds = "propp_lbs"
    proppant_pounds_per_foot = "propp_lbs_per_ft"
    stages = "stages"
    top = "top"