"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

SCALING_FACTOR = 10**5 #for computational purposes, from TONNES -> MEGATONNES    and also then NOK-> MEGANOK
#THE OLD FACTOR was 10**4, more intuitive with 10^6, However, 10**5 is a bit faster (10%)


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

EMISSION_VIOLATION_PENALTY = SCALING_FACTOR*10**(-1)  # /10**3 -> then we violate the targets quite heavily!
#THIS WAS 500    #CO2 Cap is at most 4E7, while obj function is now around 7E12, so I believe this penalty should be higher! Maybe 1-1.5E5
MAX_TRANSPORT_AMOUNT_PENALTY = SCALING_FACTOR*10**0   

#the model is quite sensitive to the initial conditions (and quickly becomes infeasible when choosing a too high rail share for example)
GROWTH_ON_ROAD = 1.4 #the transport amount on road can only grow with 40%
INIT_MODE_SPLIT_LOWER = {"Sea":0.65,"Road":0.20,"Rail":0.015}  #rail = 0.03, road = 0.25
INIT_MODE_SPLIT_UPPER = {"Sea":0.75,"Road":0.30,"Rail":0.025}  #rail = 0.03, road = 0.25