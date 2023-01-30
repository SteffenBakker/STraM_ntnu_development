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
time_periods = None  #[2022,2026,2030] or None for default up to 2050

# risk parameters
cvar_coeff = 0.2    # \lambda: coefficient for CVaR in mean-CVaR objective
cvar_alpha = 0.8    # \alpha:  indicates how far in the tail we care about risk
#TODO: test if this is working

NoBalancingTrips = False  #default at False

#################################################
#                   main code                   #
#################################################


def solve_init_model(base_data,risk_info):
        
    #set the data to focus only on base year

    
    base_data.init_data = True
    base_data.T_TIME_PERIODS = base_data.T_TIME_PERIODS_INIT
    base_data.combined_sets()

    InitModel = TranspModel(data=base_data, risk_info=risk_info)
    InitModel.NoBalancingTrips = NoBalancingTrips
    InitModel.solve_base_year = True
    print('solving initialization model')
    InitModel.construct_model()
    result = InitModel.opt.solve(InitModel.model, tee=True, symbolic_solver_labels=True, keepfiles=False)  # , tee=True, symbolic_solver_labels=True, keepfiles=True)

    if result.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
        print('the model is infeasible')

    #extract the important output
    x_flow_base_period_init = []
    t = base_data.T_TIME_PERIODS[0]
    for (i,j,m,r) in base_data.A_ARCS:
        a = (i,j,m,r)
        for f in base_data.FM_FUEL[m]:
            for p in base_data.P_PRODUCTS:
                weight = InitModel.model.x_flow[(a,f,p,t)].value
                if weight > 0:
                    x_flow_base_period_init.append((a,f,p,t,weight))
    EMISSION_CAP_ABSOLUTE_BASE_YEAR = InitModel.model.total_emissions[base_data.T_TIME_PERIODS[0]].value

    return x_flow_base_period_init, EMISSION_CAP_ABSOLUTE_BASE_YEAR

def construct_model_template(base_data,risk_info, 
                            fix_first_time_period, x_flow_base,
                            fix_first_stage,first_stage_variables,
                            scenario_names,
                            last_time_period=False):

    print("Constructing model...", end="", flush=True)
    start = time.time()
    scenario_creator_kwargs = {'base_data':base_data, 
                                'fix_first_time_period':fix_first_time_period,'x_flow_base_period_init':x_flow_base,
                                'fix_first_stage':fix_first_stage, 'first_stage_variables':first_stage_variables,
                                "risk_info":risk_info,
                                'NoBalancingTrips':NoBalancingTrips,
                                'last_time_period':last_time_period}

    ef = sputils.create_EF(
        scenario_names,  #scenario_names
        scenario_creator,
        scenario_creator_kwargs = scenario_creator_kwargs,
        nonant_for_fixed_vars = True #  MAYBE FALSE FOR VSS? (bool--optional) – If True, enforces non-anticipativity constraints for all variables, including those which have been fixed. Default is True.
    ) 
    print("done.", flush=True)
    print("Time used constructing the model:", time.time() - start, flush=True)

    return ef

def solve_model_template(ef):

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving model...")
    start = time.time()
    #options = option_settings_ef()
    solver = pyo.SolverFactory('gurobi')  #options["solvername"]
    solver.options['MIPGap']= MIPGAP # 'TimeLimit':600 (seconds)
    solver.solve(ef, tee= True)  #logfile= r'Data/Instance_results_write_to_here/Instance'+instance_run+'/logfile'+instance_run+'.log',
    print("Done solving model.")
    print("Time used solving the model:", time.time() - start)

    return ef

def solve_SP(base_data,risk_info, time_periods = None):

    #first solve the init model to initialize values

    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)

    base_data.init_data = False
    if time_periods == None:
        base_data.update_time_periods(base_data.T_TIME_PERIODS_ALL)
    else:
        base_data.update_time_periods(time_periods)

    ef = construct_model_template(base_data,risk_info,
                                fix_first_time_period=True, x_flow_base=x_flow_base_period_init,
                                fix_first_stage=False,first_stage_variables=None,
                                scenario_names=base_data.scenario_information.scenario_names,
                                last_time_period=False)
    ef = solve_model_template(ef)
    
    return ef, base_data

def solve_EEV(base_data,risk_info,time_periods=None):

    #first solve the init model to initialize values

    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)

    #focus on all data this time
        
    base_data.init_data = False
    
    if time_periods == None:
        base_data.update_time_periods(base_data.T_TIME_PERIODS_ALL)
    else:
        base_data.update_time_periods(time_periods)

    ####################################
    ####   Solve EV problem      #
    ####################################

    print('')
    print('SOLVING EV')
    print('')
    ef = construct_model_template(base_data,risk_info,
                                fix_first_time_period=True, x_flow_base=x_flow_base_period_init,
                                fix_first_stage=False,first_stage_variables=None,
                                scenario_names=['BBB'])
    ef = solve_model_template(ef)

    output_EV = OutputData(ef,base_data,EV_problem=True)
    

    # NEED TO SAVE SOME OUTPUT!!! THAT IS X_FLOW AND INVESTMENT DECISIONS!!!

    ####################################
    ####   Solve EEV problem      #
    ####################################

    #  --------- CONSTRUCT MODEL ---------     #
    print('')
    print('SOLVING EEV')
    print('')
    ef = construct_model_template(base_data,risk_info,
                                    fix_first_time_period=False, x_flow_base=x_flow_base_period_init,
                                    fix_first_stage=True,first_stage_variables=output_EV,
                                    scenario_names=base_data.scenario_information.scenario_names)
    ef = solve_model_template(ef)

    return ef, base_data

def main(analysis_type):
    
    print('----------------------------')
    print('Doing the following analysis: ', analysis_type)
    print('----------------------------')

    #     --------- DATA  ---------   #
    
            
    print("Reading data...", flush=True)
    start = time.time()
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.
    print("Done reading data.", flush=True)
    print("Time used reading the base data:", time.time() - start)
    sys.stdout.flush()

    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?
    
    
    #     --------- MODEL  ---------   #
    
    # solve model
    if analysis_type == "SP":
        ef, base_data = solve_SP(base_data,risk_info,time_periods=time_periods)
    elif analysis_type == "EEV":
        ef, base_data = solve_EEV(base_data,risk_info,time_periods=time_periods)
    
    #  --------- SAVE OUTPUT ---------    #

    file_string = 'output_data_' + analysis_type + '_' + sheet_name_scenarios 
    if NoBalancingTrips:
        file_string = file_string +'_NoBalancingTrips'
    output = OutputData(ef,base_data,EV_problem=False)

    with open(r"Data//" + file_string, 'wb') as output_file: 
        print("Dumping output in pickle file...", end="")
        pickle.dump(output, output_file)
        print("done.")

    print("Dumping data in pickle file...", end="")
    
    with open(r'Data//base_data_'+sheet_name_scenarios, 'wb') as data_file: 
        pickle.dump(base_data, data_file)
    
    print("done.")
    sys.stdout.flush()

def last_time_period_run():
    
    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios, init_data=False) #init_data is used to fix the mode-fuel mix in the first time period.
    x_flow_base_period_init, base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR = solve_init_model(base_data,risk_info)
    base_data.init_data = False
    base_data.update_time_periods(base_data.T_TIME_PERIODS_ALL)
    base_data.last_time_period = True
    base_data.combined_sets()

    ef = construct_model_template(base_data,risk_info,
                                fix_first_time_period=False, x_flow_base=None,
                                fix_first_stage=False,first_stage_variables=None,
                                scenario_names=base_data.scenario_information.scenario_names,
                                last_time_period=True)
    ef = solve_model_template(ef)

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

    #last_time_period_run()

    









    # if profiling:
        #     profiler = cProfile.Profile()
        #     profiler.enable()
        


        
