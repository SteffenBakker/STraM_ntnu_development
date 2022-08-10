"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

SCALING_FACTOR = 10000 #for computational purposes

# Data settings (create_sets_class)

NEW_FUEL_FACTOR = 0.5
RISK_FREE_RATE = 0.02 
MAX_TRUCK_CAP = 30
INTERNATIONAL_CAP = 240000000 #this should be the total transport in tonnes or so ...  Going to be removed!!
FACTOR_INT_CAP_ROAD = 0.4
FACTOR_INT_CAP_RAIL = 0.05 
FACTOR_INT_CAP_SEA = 0.75
FLEET_RR = 0.2

#Model settings

EMISSION_VIOLATION_PENALTY = 500
POSITIVE_PART_PENALTY = 100  #TO DO: test the value of this parameter