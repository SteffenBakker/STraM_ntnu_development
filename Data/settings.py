"""
SETTINGS  / HARDCODED PARAMETERS
"""

# General settings

SCALING_FACTOR = 10**5 #for computational purposes, from TONNES -> MEGATONNES    and also then NOK-> MEGANOK
#THE OLD FACTOR was 10**4, more intuitive with 10^6, However, 10**5 is a bit faster (10%)
SCALING_FACTOR_MONETARY = 10**9    #7
SCALING_FACTOR_WEIGHT = 10**6    #5
SCALING_FACTOR_EMISSIONS = 10**8 #if smaller, then the CO2_FEE disappears, needs to be small as emissions are in GRAM CO2

#gurobi's tolerance is 10^-6
#(minimum demand is now 3.3*10^3 tonnes, so should be OK)
# for transport costs, the minimum is 6.7 kroners/ton -> 6.7MKR/MTONNES

# Data settings (create_sets_class)

RISK_FREE_RATE = 0.038 # social discount rate, ref Ruben (Old:  https://tradingeconomics.com/norway/government-bond-yield  -> 3.2%)
CO2_PRICE_FACTOR = {2022:1,
                    2026:1,
                    2030:1,
                    2040:1,  #1.5,
                    2050:1}
INCLUDE_DRY_BULK = True
TWO_MODE_PATHS = True
NUM_DIGITS_PRECISION = 5 #for the rounding in data generation

MAX_TRUCK_CAP = 30 #to do: check this value
EMPTY_VEHICLE_FACTOR = 0.6

INTERPOLATE_DEMAND_DATA_2040 = False #there

#Model settings

NO_INVESTMENTS = False
MIPGAP = 0.005 # fraction, multiply with 100 to get percentage (0.5%)

#the model is quite sensitive to the initial conditions (and quickly becomes infeasible when choosing a too high rail share for example)
GROWTH_ON_ROAD = 2 #the transport amount on road can only grow with 100%
RHO_STAR = 0.975  #implied rho of around 20% over 5 years
