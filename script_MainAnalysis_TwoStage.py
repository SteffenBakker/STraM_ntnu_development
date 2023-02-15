# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

import os
#Remember to set the right workingdirectory. Otherwise errors with loading the classes
# os.chdir('C://Users//steffejb//OneDrive - NTNU//Work//GitHub//AIM_Norwegian_Freight_Model//AIM_Norwegian_Freight_Model')
#os.chdir("M:/Documents/GitHub/AIM_Norwegian_Freight_Model") #uncomment this for stand-alone testing of this fille

from TranspModelClass import TranspModel, RiskInformation
from ExtractModelResults import OutputData
from Data.Create_Sets_Class import TransportSets
from Data.settings import *

#import mpisppy.utils.sputils as sputils
#from solver_and_scenario_settings import scenario_creator
#from mpisppy.opt.ph import PH


#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

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

from Utils2 import Logger

#################################################
#                   user input                  #
#################################################

analysis_type = 'SP' #, 'EEV' , 'SP'         expected value probem, expectation of EVP, stochastic program
wrm_strt = False  #use EEV as warm start for SP
sheet_name_scenarios = 'scenarios_base' #scenarios_base,three_scenarios_new, three_scenarios_with_maturity
time_periods = None  #[2022,2026,2030] or None for default up to 2050

# risk parameters
cvar_coeff = 0.2    # \lambda: coefficient for CVaR in mean-CVaR objective
cvar_alpha = 0.8    # \alpha:  indicates how far in the tail we care about risk
#TODO: test if this is working

NoBalancingTrips = False  #default at False

log_to_file = True

#################################################
#                   main code                   #
#################################################

run_identifier = analysis_type + '_' + sheet_name_scenarios
if wrm_strt:
    run_identifier = run_identifier +'_WrmStrt'
if NoBalancingTrips:
    run_identifier = run_identifier +'_NoBalancingTrips'

sys.stdout = Logger(run_identifier,log_to_file)


def solve_init_model(base_data,risk_info):
        
    #set the data to focus only on base year

    base_data.init_data = True
    base_data.T_TIME_PERIODS = base_data.T_TIME_PERIODS_INIT
    base_data.combined_sets()

    InitModel = TranspModel(data=base_data, risk_info=risk_info)
    InitModel.NoBalancingTrips = NoBalancingTrips
    InitModel.solve_base_year = True
    print('-----------------')
    print('constructing initialization model',flush=True)
    start = time.time()
    InitModel.construct_model()
    print("Time used constructing the model:", time.time() - start,flush=True)
    print('-----------------')

    print('solving initialization model',flush=True)
    start = time.time()
    InitModel.solve_model()
    print("Time used solving the model:", time.time() - start,flush=True)
    print('-----------------')
    sys.stdout.flush()

    #extract the important output
    x_flow_base_period_init = []
    t = base_data.T_TIME_PERIODS[0]
    for (i,j,m,r) in base_data.A_ARCS:
        a = (i,j,m,r)
        for f in base_data.FM_FUEL[m]:
            for p in base_data.P_PRODUCTS:
                for s in base_data.S_SCENARIOS:
                    weight = InitModel.model.x_flow[(a,f,p,t,s)].value
                    if weight > 0:
                        x_flow_base_period_init.append((a,f,p,t,s,weight))
    EMISSION_CAP_ABSOLUTE_BASE_YEAR = InitModel.model.total_emissions[base_data.T_TIME_PERIODS[0],base_data.S_SCENARIOS[0]].value  #same emissions across all scenarios!

    return x_flow_base_period_init, EMISSION_CAP_ABSOLUTE_BASE_YEAR

def construct_and_solve_SP(base_data,
                            risk_info, 
                            last_time_period=False,
                            time_periods = None):
    
    # ------ SOLVE INIT MODEL ----------#
    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)

    # ------ CHANGE DATA BACK TO STANDARD ----------#

    base_data.init_data = False
    if time_periods == None:
        base_data.update_time_periods(base_data.T_TIME_PERIODS_ALL)
    else:
        base_data.update_time_periods(time_periods)

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing SP model...")

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.NoBalancingTrips = NoBalancingTrips
    model_instance.last_time_period = last_time_period
    model_instance.construct_model()
    model_instance.fix_variables_first_time_period(x_flow_base_period_init)
    
    #if fix_first_stage:
    #    model_instance.fix_variables_first_stage(output_EV)

    print("Done constructing model.")
    print("Time used constructing the model:", time.time() - start)
    print("----------")
    sys.stdout.flush()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving model...",flush=True)
    start = time.time()
    model_instance.solve_model(FeasTol=10**(-2),num_focus=0) 
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", end="", flush=True)

    return model_instance,base_data

def construct_and_solve_EEV(base_data,risk_info):

    base_data.S_SCENARIOS = ['BBB']
    base_data.combined_sets()

        ############################
        ###  1: solve init model ###
        ############################
    
    #first solve the init model to initialize values
    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)

    #focus on all data this time
        
    base_data.init_data = False
    if time_periods == None:
        base_data.T_TIME_PERIODS = base_data.T_TIME_PERIODS_ALL    
    else:
        base_data.T_TIME_PERIODS = time_periods
    base_data.combined_sets()


        ############################
        ###  #2: solve EV        ###
        ############################
    
    
    # ------ CONSTRUCT MODEL ----------#

    print("Constructing EV model.....", end="", flush=True)

    start = time.time()
    model_instance_EV = TranspModel(data=base_data, risk_info=risk_info)
    model_instance_EV.NoBalancingTrips = NoBalancingTrips
    #constructing
    model_instance_EV.construct_model()
    #fixing variables
    model_instance_EV.fix_variables_first_time_period(x_flow_base_period_init)

    print("Done constructing EV model.",flush=True)
    print("Time used constructing the model:", time.time() - start,flush=True)
    print("----------", flush=True)


    #  ---------  SOLVE MODEL  ---------    #

    print("Solving EV model.....",end="",flush=True)
    start = time.time()
    model_instance_EV.solve_model()
    print("Done solving model.")
    print("Time used solving the model:", time.time() - start)
    print("----------",  flush=True)


        ############################
        ###  #3: solve EEV       ###
        ############################
    
    base_data.S_SCENARIOS = base_data.S_SCENARIOS_ALL
    base_data.combined_sets()

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing EEV model...",end='', flush=True)

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.NoBalancingTrips = NoBalancingTrips
    model_instance.construct_model()
    model_instance.fix_variables_first_stage(model_instance_EV.model)
    
    #if fix_first_stage:
    #    model_instance.fix_variables_first_stage(output_EV)

    print("Done constructing EEV model.",flush=True)
    print("Time used constructing the model:", time.time() - start,flush=True)
    print("----------",  flush=True)


    #  ---------  SOLVE MODEL  ---------    #

    print("Solving EEV model...",end='',flush=True)
    start = time.time()
    #options = option_settings_ef()
    model_instance.solve_model(FeasTol=10**(-3)) #to make sure that the warm start is feasible
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------",  flush=True)


    # --------- SAVE RESULTS ------------#

    #-----------------------------------

    if analysis_type == "SP":
        file_string = 'output_data_' + "EEV" + '_' + sheet_name_scenarios
        if NoBalancingTrips:
            file_string = file_string +'_NoBalancingTrips'
        
        output = OutputData(model_instance.model,base_data)

        with open(r"Data//" + file_string, 'wb') as output_file: 
            print("Dumping EEV output in pickle file.....", end="",flush=True)
            pickle.dump(output, output_file)
            print("done.",flush=True)
        
        sys.stdout.flush()


    return model_instance, base_data

def construct_and_solve_SP_warm_start(base_data,
                            risk_info, 
                            last_time_period=False,
                            time_periods = None):
    
    model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)

    model_instance.unfix_variables_first_stage()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving SP model with EEV warm start...",end='',flush=True)
    start = time.time()
    model_instance.solve_model(warmstart=True,FeasTol=10**(-2))
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", flush=True)

    return model_instance,base_data



def main(analysis_type):
    
    print('----------------------------')
    print('Doing the following analysis: ')
    print(analysis_type)
    print(sheet_name_scenarios)
    if wrm_strt:
        print('Using EEV warm start')
    print('----------------------------')
    sys.stdout.flush()
    #     --------- DATA  ---------   #
    
            
    print("Reading data...", flush=True)
    start = time.time()
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.
    print("Done reading data.", flush=True)
    print("Time used reading the base data:", time.time() - start,flush=True)
    sys.stdout.flush()

    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?
    base_data.risk_information = risk_info
    
    #     --------- MODEL  ---------   #
    # solve model
    if analysis_type == "SP":
        if wrm_strt:
            model_instance,base_data = construct_and_solve_SP_warm_start(base_data,risk_info,time_periods=time_periods)
        else:
            model_instance,base_data = construct_and_solve_SP(base_data,risk_info,time_periods=time_periods)

    elif analysis_type == "EEV":
        model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)
    
    #  --------- SAVE OUTPUT ---------    #

    print("Dumping data in pickle file...", end="")
    with open(r'Data//base_data_'+sheet_name_scenarios, 'wb') as data_file: 
        pickle.dump(base_data, data_file)
    print("done.")

    #-----------------------------------

    file_string = 'output_data_' + run_identifier   
    
    output = OutputData(model_instance.model,base_data)

    with open(r"Data//" + file_string, 'wb') as output_file: 
        print("Dumping output in pickle file.....", end="")
        pickle.dump(output, output_file)
        print("done.")
    
    sys.stdout.flush()


def main2():

    print('----------------------------')
    print('Only do init: ', analysis_type, sheet_name_scenarios)
    print('----------------------------')
    sys.stdout.flush()
    #     --------- DATA  ---------   #
     
    print("Reading data...", flush=True)
    start = time.time()
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.
    print("Done reading data.", flush=True)
    print("Time used reading the base data:", time.time() - start,flush=True)
    sys.stdout.flush()

    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?
    base_data.risk_information = risk_info

    print("solve init model", flush=True)
    # ------ SOLVE INIT MODEL ----------#
    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)
    print("finished solving init model", flush=True)


def last_time_period_run():
    
    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.
    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)
    base_data.init_data = False
    base_data.update_time_periods(base_data.T_TIME_PERIODS_ALL)
    base_data.last_time_period = True
    base_data.combined_sets()

    ef = construct_model_template_ef(base_data,risk_info,
                                fix_first_time_period=False, x_flow_base=None,
                                fix_first_stage=False,first_stage_variables=None,
                                scenario_names=base_data.scenario_information.scenario_names,
                                last_time_period=True)
    ef = solve_model_template_ef(ef)

    file_string = 'output_data_' + analysis_type + '_' + sheet_name_scenarios +'_last_period' 
    output = OutputData(ef,base_data,EV_problem=False)

    with open(r"Data//" + file_string, 'wb') as output_file: 
        print("Dumping output in pickle file...", end="")
        pickle.dump(output, output_file)
        print("done.")

if __name__ == "__main__":
    
    #for analysis_type in ['SP','EEV']:
    #    main(analysis_type=analysis_type)
    
    main(analysis_type=analysis_type)
    #main2()

    #last_time_period_run()

    


    # if profiling:
        #     profiler = cProfile.Profile()
        #     profiler.enable()
        


        
