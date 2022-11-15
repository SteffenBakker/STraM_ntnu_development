# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

import os
#Remember to set the right workingdirectory. Otherwise errors with loading the classes
# os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')

from TranspModelClass import TranspModel
from PostProcessClass import OutputData
from Data.Create_Sets_Class import TransportSets
from solver_and_scenario_settings import scenario_creator, scenario_denouement, option_settings
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


distribution_on_cluster = False  #is the code to be run on the cluster using the distribution package?
read_data_from_scratch = True #Use cached data? Exctracting data is a bit slow in debug mode
extract_data_postprocessing = True #postprocessing is quite slow. No need to do when testing the model. 
instance_run = 'base'     #change instance_run to choose which instance you want to run

profiling = False

#################################################
#                   main code                   #
#################################################


if __name__ == "__main__":
    
    
    if not os.path.exists(r'Data/Instance_results_write_to_here/Instance'+instance_run):
        os.makedirs(r'Data/Instance_results_write_to_here/Instance'+instance_run)
        
    if read_data_from_scratch:
        base_data = TransportSets() #needs to be initialized with some scenario.
        with open(r'Data\base_data', 'wb') as data_file: 
            pickle.dump(base_data, data_file)
    else:
        with open(r'Data\base_data', 'rb') as data_file:
            base_data = pickle.load(data_file)
 
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    #Model   (consider removing the base_model, not used)
    #base_model = TranspModel(data=base_data)
    #base_model.construct_model()

    #Scenarios
    scenario_names = base_data.scenario_information.scenario_names


    #Solve model 
    scenario_creator_kwargs = {'base_data':base_data}
    options = option_settings()
    solver = pyo.SolverFactory(options["solvername"])
    
    start = time.time()
    
    ef = sputils.create_EF(
        base_data.scenario_information.scenario_names,
        scenario_creator,
        scenario_creator_kwargs = scenario_creator_kwargs
    )
    results = solver.solve(ef, logfile= r'Data/Instance_results_write_to_here/Instance'+instance_run+'/logfile'+instance_run+'.log', tee= True)
    
    print('EF objective value:', pyo.value(ef.EF_Obj))
    stop = time.time()
    print("The time of the run:", stop - start)

    #scenarios = sputils.ef_scenarios(ef)
    if extract_data_postprocessing:        
        output = OutputData(ef,base_data,instance_run)
        with open(r'Data\output_data', 'wb') as output_file: 
            pickle.dump(output, output_file)
        
        #output.emission_results(base_data)
        #plot_figures(base_data,dataset_x_flow,scenarios,instance_run,solution_method)
        #output.cost_and_investment_table(base_data)
