"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

#SCALING_FACTOR = 10000   THE OLD FACTOR, more intuitive with 10^6, However, 10**5 is a bit faster (10%)
SCALING_FACTOR = 10**5 #for computational purposes, from TONNES -> MEGATONNES    and also then NOK-> MEGANOK
#gurobi's tolerance is 10^-6
#(minimum demand is now 3.3*10^3 tonnes, so should be OK)
# for transport costs, the minimum is 6.7 kroners/ton -> 6.7MKR/MTONNES

# Data settings (create_sets_class)

RISK_FREE_RATE = 0.038 # social discount rate, ref Ruben (Old:  https://tradingeconomics.com/norway/government-bond-yield  -> 3.2%)
MAX_TRUCK_CAP = 30
EMPTY_VEHICLE_FACTOR = 0.6

#Model settings

EMISSION_VIOLATION_PENALTY = 1*10**5  # TO DO: depend on scaling factor
#THIS WAS 500    #CO2 Cap is at most 4E7, while obj function is now around 7E12, so I believe this penalty should be higher! Maybe 1-1.5E5
MAX_TRANSPORT_AMOUNT_PENALTY = 1  #TO DO: test the value of this parameter. (0.001 was too low)
MIPGAP = 0.005 # fraction, multiply with 100 to get percentage (0.5%)