"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

SCALING_FACTOR = 10**5 #for computational purposes, from TONNES -> MEGATONNES    and also then NOK-> MEGANOK
#THE OLD FACTOR was 10**4, more intuitive with 10^6, However, 10**5 is a bit faster (10%)
SCALING_FACTOR_MONETARY = SCALING_FACTOR*10**2
SCALING_FACTOR_WEIGHT = SCALING_FACTOR
SCALING_FACTOR_EMISSIONS = SCALING_FACTOR*10**2


#gurobi's tolerance is 10^-6
#(minimum demand is now 3.3*10^3 tonnes, so should be OK)
# for transport costs, the minimum is 6.7 kroners/ton -> 6.7MKR/MTONNES

# Data settings (create_sets_class)

RISK_FREE_RATE = 0.038 # social discount rate, ref Ruben (Old:  https://tradingeconomics.com/norway/government-bond-yield  -> 3.2%)
MAX_TRUCK_CAP = 30 #to do: check this value
EMPTY_VEHICLE_FACTOR = 0.6

INTERPOLATE_DEMAND_DATA_2040 = True

#Model settings

NO_INVESTMENTS = False
MIPGAP = 0.005 # fraction, multiply with 100 to get percentage (0.5%)

EMISSION_VIOLATION_PENALTY = 10**11/SCALING_FACTOR_EMISSIONS  
MAX_TRANSPORT_AMOUNT_PENALTY = 10**7/SCALING_FACTOR_WEIGHT  

#the model is quite sensitive to the initial conditions (and quickly becomes infeasible when choosing a too high rail share for example)
GROWTH_ON_ROAD = 1.4 #the transport amount on road can only grow with 40%
INIT_MODE_SPLIT_LOWER = {"Sea":0.65,"Road":0.20,"Rail":0.015}  #rail = 0.03, road = 0.25
INIT_MODE_SPLIT_UPPER = {"Sea":0.75,"Road":0.30,"Rail":0.025}  #rail = 0.03, road = 0.25