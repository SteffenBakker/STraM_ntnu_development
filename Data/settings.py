"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

#SCALING_FACTOR = 10000   THE OLD FACTOR, more intuitive with 10^6
SCALING_FACTOR = 100000 #for computational purposes, from TONNES -> MEGATONNES    and also then NOK-> MEGANOK
#gurobi's tolerance is 10^-6
#(minimum demand is now 3.3*10^3 tonnes, so should be OK)
# for transport costs, the minimum is 6.7 kroners/ton -> 6.7MKR/MTONNES



# Data settings (create_sets_class)

RISK_FREE_RATE = 0.02 
MAX_TRUCK_CAP = 30
FLEET_RR = 0.2  # TO DO: define different fleet renewal rates (for mode fuel combinations)

#it seems like the capacity on international transport is not being used. CHECK if it was used
INTERNATIONAL_CAP = 240000000 #this should be the total transport in tonnes or so ...  Going to be removed!!
FACTOR_INT_CAP_ROAD = 0.4
FACTOR_INT_CAP_RAIL = 0.05 
FACTOR_INT_CAP_SEA = 0.75


#Model settings

EMISSION_VIOLATION_PENALTY = 1*10**5  # THIS WAS 500    #CO2 Cap is at most 4E7, while obj function is now around 7E12, so I believe this penalty should be higher! Maybe 1-1.5E5
MAX_TRANSPORT_AMOUNT_PENALTY = 100  #TO DO: test the value of this parameter. 
