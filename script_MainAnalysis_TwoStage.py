# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

import os
#Remember to set the right workingdirectory. Otherwise errors with loading the classes
# os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')

from TranspModelClass import TranspModel
from ExtractModelResults import OutputData
from Data.Create_Sets_Class import TransportSets
from Data.settings import *
from solver_and_scenario_settings import scenario_creator, scenario_denouement, option_settings_ef
#from postprocessing import extract_output_ef,extract_aggregated_output_ef, extract_output_ph,plot_figures

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
import mpisppy.scenario_tree as scenario_tree
import time
import sys
import pickle

import cProfile
import pstats


#################################################
#                   user input                  #
#################################################

profiling = False
distribution_on_cluster = False  #is the code to be run on the cluster using the distribution package?
instance_run = 'base'     #change instance_run to choose which instance you want to run

read_data_from_scratch = True #Use cached data? Exctracting data is a bit slow in debug mode
extract_data_postprocessing = True #postprocessing is quite slow. No need to do when testing the model. 

analysis_type = 'SP' # 'EEV' , 'SP'         expected value probem, expectation of EVP, stochastic program
sheet_name_scenarios = 'three_scenarios' #scenarios_base, three_scenarios
    

#################################################
#                   main code                   #
#################################################


if __name__ == "__main__":
    

    #     --------- DATA  ---------   #
    

    EV_problem = False
    if analysis_type == 'EV':
        EV_problem = True
    EEV_problem = False
    if analysis_type == 'EEV':
        EEV_problem = True

    if not os.path.exists(r'Data/Instance_results_write_to_here/Instance'+instance_run):
        os.makedirs(r'Data/Instance_results_write_to_here/Instance'+instance_run)
        
    if read_data_from_scratch:
        start = time.time()
        base_data = TransportSets(sheet_name_scenarios) 
        with open(r'Data\base_data', 'wb') as data_file: 
            pickle.dump(base_data, data_file)
        print("Time used reading the base data:", time.time() - start)
    else:
        with open(r'Data\base_data', 'rb') as data_file:
            base_data = pickle.load(data_file)


    #   --------- SCENARIOS ---------  #


    scenario_names = base_data.scenario_information.scenario_names    
    if analysis_type == 'EV':
        scenario_names = ['MMM'] 


    #  --------- CONSTRUCT MODEL ---------     #


    start = time.time()
    scenario_creator_kwargs = {'base_data':base_data, 'fix_first_stage':EEV_problem}
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs = scenario_creator_kwargs,
        nonant_for_fixed_vars = True #  MAYBE FALSE FOR VSS? (bool--optional) – If True, enforces non-anticipativity constraints for all variables, including those which have been fixed. Default is True.
    ) 
    print("Time used constructing the model:", time.time() - start)


    #  ---------  SOLVE MODEL  ---------    #


    start = time.time()
    #options = option_settings_ef()
    solver = pyo.SolverFactory('gurobi')  #options["solvername"]
    solver.options['MIPGap']= MIPGAP # 'TimeLimit':600 (seconds)
    results = solver.solve(ef,logfile= r'Data/Instance_results_write_to_here/Instance'+instance_run+'/logfile'+instance_run+'.log', tee= True)
    print("Time used solving the model:", time.time() - start)


    #  --------- SAVE OUTPUT ---------    #


    file_string = 'output_data_' + analysis_type
    if extract_data_postprocessing:        
        output = OutputData(ef,base_data,instance_run,EV_problem)
        with open(r'Data\\' + file_string, 'wb') as output_file: 
            pickle.dump(output, output_file)





    # if profiling:
    #     profiler = cProfile.Profile()
    #     profiler.enable()
