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

# Data settings (ConstructData.py)

EXCHANGE_RATE_EURO_TO_NOK = 10
RISK_FREE_RATE = 0.038 # social discount rate, ref Ruben (Old:  https://tradingeconomics.com/norway/government-bond-yield  -> 3.2%)
CO2_PRICE_FACTOR = {2022:1,
                    2026:1,
                    2030:1,
                    2040:1,  #1.5,
                    2050:1}
NO_DRY_BULK = False
NO_LIQUID_BULK = True

#path generation
NUM_MODE_PATHS = 2  #hvor mange modes kan bli brukt på en path? 
# single mode paths can lead to infeasibilities in the model (the flow/demand constraint). Some demand requests are then not feasible.

NUM_DIGITS_PRECISION = 5 #for the rounding in data generation

MAX_TRUCK_CAP = 30 #tonnes
AVG_TRUCK_PAYLOAD = 15 #tonnes
EMPTY_VEHICLE_FACTOR = 0.6




#Model settings
NO_INVESTMENTS = False
MIPGAP = 0.002 # fraction, multiply with 100 to get percentage (0.5%)

#discount rate
RHO_STAR = 0.975  #implied rho of around 20% over 5 years


############# Model formulation Parameters ################

ABSOLUTE_DEVIATION = 0.001
RELATIVE_DEVIATION = 0.001
FEAS_RELAX = 0


#### COLOR MAP ###

def rgb_constructor(r,g,b):
    rgb_code = (r,g,b)
    return tuple([x / 255 for x in rgb_code]) #needs to return a number between zero and one

#https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
color_map_stram = {
    "ocean": rgb_constructor(189, 224, 254),
    "land": rgb_constructor(229, 229, 229),
    "Electricity": "mediumseagreen",#rgb_constructor(197, 20, 20),
    "Battery": rgb_constructor(49,132,109), #"mediumseagreen",#
    "Catenary": rgb_constructor(10,91,113), #"darkgreen",
    "Hydrogen": rgb_constructor(2,61,100), #"royalblue",,
    "Ammonia": rgb_constructor(253,174,40),
    "Methanol": rgb_constructor(140,142,75),
    "Diesel": rgb_constructor(95,95,95),
    "MGO": rgb_constructor(95,95,95),
    "HFO": rgb_constructor(51,51,51),
    "Road": rgb_constructor(207, 65, 84), 
    "Sea": rgb_constructor(47, 85, 151), 
    "Rail": rgb_constructor(55, 0, 30), 
    "Emission": rgb_constructor(172,0,0),
    "Emission (Empty Trips)":    rgb_constructor(230,0,0),  
    "RailTrack": rgb_constructor(38,38,38),
    "Terminal":rgb_constructor(130,130,130),
    "RailElectr.": rgb_constructor(10,91,113),
    "Charging": rgb_constructor(49,132,109),
    "H2_Filling": rgb_constructor(2,61,100),
    "LCOT": rgb_constructor(38,38,38),
    "LCOT (Empty Trips)":    rgb_constructor(130,130,130),
    "Time value":               rgb_constructor(88,137,83),
    "Transfer":                 rgb_constructor(255,192,0),
    "CO2_Penalty":              "darkred",
}