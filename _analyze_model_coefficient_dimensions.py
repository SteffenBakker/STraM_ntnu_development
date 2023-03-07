# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

import os
#Remember to set the right workingdirectory. Otherwise errors with loading the classes
# os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')
#os.chdir("M:/Documents/GitHub/AIM_Norwegian_Freight_Model") #uncomment this for stand-alone testing of this fille

from Model import TranspModel, RiskInformation
from ExtractResults import OutputData
from Data.Create_Sets_Class import TransportSets
from Data.settings import *

import mpisppy.utils.sputils as sputils
from solver_and_scenario_settings import scenario_creator

#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

import pyomo.environ as pyo
import numpy as np
import pandas as pd
#from mpisppy.opt.ef import ExtensiveForm
#import mpisppy.scenario_tree as scenario_tree
import time
import sys
import pickle
import json #works across operating systems

import cProfile
import pstats



#################################################
#                   user input                  #
#################################################

profiling = False
distribution_on_cluster = False  #is the code to be run on the cluster using the distribution package?

analysis_type = 'SP' #, 'EEV' , 'SP'         expected value probem, expectation of EVP, stochastic program
sheet_name_scenarios = 'three_scenarios_new' #scenarios_base,three_scenarios_new, three_scenarios_with_maturity

# risk parameters
cvar_coeff = 0.2    # \lambda: coefficient for CVaR in mean-CVaR objective
cvar_alpha = 0.8    # \alpha:  indicates how far in the tail we care about risk
#TODO: test if this is working
    
#################################################
#                   main code                   #
#################################################


if __name__ == "__main__":
    

    #     --------- DATA  ---------   #
    
    instance_run = 'base'

        
    print("Reading data...", flush=True)
    start = time.time()
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios) #init_data is used to fix the mode-fuel mix in the first time period.
    print("Done reading data.", flush=True)
    print("Time used reading the base data:", time.time() - start)
    sys.stdout.flush()

    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?

    
    #     --------- MODEL  ---------   #
    
#in the objective:

    pd.DataFrame(base_data.D_DEMAND.values()).describe()  #0 - 2E6

    pd.DataFrame(base_data.C_TRANSP_COST.values()).describe()  #0 - 2E6
    pd.DataFrame(base_data.C_CO2.values()).describe() #0-2E2
    pd.DataFrame(base_data.C_TRANSFER.values()).describe() #0-4E2
    pd.DataFrame(base_data.C_EDGE_RAIL.values()).describe() #0- 1E6
    pd.DataFrame(base_data.C_NODE.values()).describe() # 3E5
    pd.DataFrame(base_data.C_UPG.values()).describe() #1.5E5
    pd.DataFrame(base_data.C_CHARGE.values()).describe() #160

    #pd.DataFrame(base_data.E_EMISSIONS.values()).describe() #0 - 5E4 #not relevant anymore
    
    pd.DataFrame(base_data.Q_EDGE_RAIL.values()).describe()   #7.5
    pd.DataFrame(base_data.Q_NODE_BASE.values()).describe()   #1000
    pd.DataFrame(base_data.Q_NODE.values()).describe() #300
    pd.DataFrame(base_data.Q_CHARGE_BASE.values()).describe()

    pd.DataFrame(base_data.BIG_M_UPG.values()).describe() #11

    pd.DataFrame(base_data.AVG_DISTANCE.values()).describe() #729

    pd.DataFrame(base_data.Q_SHARE_INIT_MAX.values()).describe()



    import math
    def magnitude_order(num):
        if num == 0:
            return 0

        absnum = abs(num)
        order = math.log10(absnum)
        res = math.floor(order)

        return res
    print('-----')
    for (name,dict) in [('demand',base_data.D_DEMAND),
                        ('transp_cost',base_data.C_TRANSP_COST),
                        ('co2cost',base_data.C_CO2),
                        ('transfcost',base_data.C_TRANSFER),
                        ('edge',base_data.C_EDGE_RAIL),
                        ('node',base_data.C_NODE),
                        ('upg',base_data.C_UPG),
                        ('charge',base_data.C_CHARGE),
                        ('q_edge',base_data.Q_EDGE_RAIL),
                        ('q_node_base',base_data.Q_NODE_BASE),
                        ('q_node_add',base_data.Q_NODE)
                        ]:
        val_list=list(dict.values())
        magnitude_list=list(map(magnitude_order, val_list))
        print(name)
        print('min: ', min(magnitude_list))
        print('max: ', max(magnitude_list))
        print('-----')


        
    #inspect_list = list([values for (i,j,p,t),values in base_data.D_DEMAND.items() if p!="Dry bulk"])
    inspect_list = list([values for (i,j,p,t),values in base_data.D_DEMAND.items() if p=="Dry bulk"])

    pd.Series((inspect_list)).describe()

    
    
    
    

    

    

    