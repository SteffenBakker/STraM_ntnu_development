#check

import pandas as pd
from Data.ConstructData import TransportSets
from Data.settings import *

sheet_name_scenarios = 'four_scenarios'
base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.

#NUM DECIMALS

def count_decimals(number):
    n = str(number)[::-1].find('.')
    return n

pd.Series(list(map(count_decimals, list(base_data.C_TRANSP_COST.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_CO2.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_TRANSFER.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_EDGE_INV.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_NODE_INV.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_EDGE_UPG.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_CHARGE.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.D_DEMAND.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.E_EMISSIONS.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.CO2_fee.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.C_CO2.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.Q_EDGE_INV.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.Q_NODE_BASE.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.Q_NODE_INV.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.Q_EDGE_BASE.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.BIG_M_UPG.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.AVG_DISTANCE.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.R_TECH_READINESS_MATURITY.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.Q_SHARE_INIT_MAX.values())))).describe()
pd.Series(list(map(count_decimals, list(base_data.BIG_M_UPG.values())))).describe()


#SIZE


pd.Series(base_data.C_TRANSP_COST.values()).describe()
pd.Series(base_data.C_CO2.values()).describe()
pd.Series(base_data.C_TRANSFER.values()).describe()
pd.Series(base_data.C_EDGE_INV.values()).describe()
pd.Series(base_data.C_NODE_INV.values()).describe()
pd.Series(base_data.C_EDGE_UPG.values()).describe()
pd.Series(base_data.C_CHARGE.values()).describe()
pd.Series(base_data.D_DISCOUNT_RATE).describe()
pd.Series(base_data.D_DEMAND.values()).describe()
pd.Series(base_data.E_EMISSIONS.values()).describe()
pd.Series(base_data.CO2_fee.values()).describe()
pd.Series(base_data.C_CO2.values()).describe()
pd.Series(base_data.Q_EDGE_BASE.values()).describe()
pd.Series(base_data.Q_NODE_BASE.values()).describe()
pd.Series(base_data.Q_NODE_INV.values()).describe()
pd.Series(base_data.Q_EDGE_INV.values()).describe()
pd.Series(base_data.BIG_M_UPG.values()).describe()
pd.Series(base_data.AVG_DISTANCE.values()).describe()
pd.Series(base_data.R_TECH_READINESS_MATURITY.values()).describe()
pd.Series(base_data.Q_SHARE_INIT_MAX.values()).describe()
pd.Series(base_data.BIG_M_UPG.values()).describe()
base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR

base_data.tech_base_bass_model






# driver code
vals = [1.222, 2.33333, 3.000000001]
  
# Map method returns a map object
# so we cast it into list using list()

  
print(output_list)