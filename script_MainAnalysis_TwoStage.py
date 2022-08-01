# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

#Remember to set the right workingdirectory. Otherwise errors with loading the classes
import os
#os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')

from TranspModelClass import TranspModel
from Data.Create_Sets_Class import TransportSets
from solver_and_scenario_settings import scenario_creator, scenario_denouement, option_settings, get_all_scenario_names
from postprocessing import extract_output_ef, extract_output_ph,plot_figures

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.opt.ph import PH
import mpisppy.scenario_tree as scenario_tree
import time
import sys

import cProfile
import pstats

start = time.time()

#################################################
#                   user input                  #
#################################################


distribution_on_cluster = False  #is the code to be run on the cluster using the distribution package?
extract_data = False #this is quite slow (the postprocessing with all the write to file)
set_instance_manually = True
instance = '2'     #change instance_run to choose which instance you want to run
manual_instance = {'Manual':{'sol_met':'ef',
                                    'scen_struct':6,
                                    'co2_price':1,
                                    'costs':'avg',
                                    'probs':'equal',
                                    'emission_reduction':75}}
profiling = True

#################################################
#                   main code                   #
#################################################


if __name__ == "__main__":

    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()    
    
    if distribution_on_cluster:
        instance_run = sys.argv[1]
    else:
        instance_run = instance # or sys.argv[1] if master.bash is run
    
    # READ DATA
    
    if set_instance_manually:
        instance_run = 'Manual'
        instance_dict = manual_instance
    else:
        instance_data = pd.read_excel(r'Data/instances.xlsx', sheet_name='Sheet1')
        instances = [str(i) for i in list(instance_data['instance'])]
        instance_dict = {instances[i]:instance_data.set_index('instance').to_dict('records')[i] for i in range(len(instances))}
        
        print("Running and logging instance ", instance_run)
        print(instance_dict[instance_run])
    
    if not os.path.exists(r'Data/Instance_results_write_to_here/Instance'+instance_run):
        os.makedirs(r'Data/Instance_results_write_to_here/Instance'+instance_run)
    
    solution_method = instance_dict[instance_run]['sol_met']
    scenario_structure = instance_dict[instance_run]['scen_struct']
    CO2_price = instance_dict[instance_run]['co2_price']
    fuel_cost = instance_dict[instance_run]['costs']
    probabilities = instance_dict[instance_run]['probs']
    emission_reduction = instance_dict[instance_run]['emission_reduction']
    
    base_data = TransportSets('HHH',CO2_price,fuel_cost, emission_reduction) #needs to be initialized with some scenario I guess.
    all_scenario_names = get_all_scenario_names(scenario_structure,base_data)    
    base_model = TranspModel(instance=instance_run,base_data=base_data, one_time_period=False, scenario = all_scenario_names[0],
                          carbon_scenario = CO2_price, fuel_costs=fuel_cost, emission_reduction=emission_reduction)
    base_model.construct_model()
    options = option_settings()
    

    #SOLVE MODEL AND EXTRACT OUTPUT
    scenario_creator_kwargs = {'probabilities':probabilities, 'base_model':base_model, 'base_data':base_data}
    if solution_method == "ef":
        solver = pyo.SolverFactory(options["solvername"])

        ef = sputils.create_EF(
                all_scenario_names,
                scenario_creator,
                scenario_creator_kwargs = scenario_creator_kwargs
            )

        results = solver.solve(ef, logfile= r'Data/Instance_results_write_to_here/Instance'+instance_run+'/logfile'+instance_run+'.log', tee= True)
        print('EF objective value:', pyo.value(ef.EF_Obj))
        stop = time.time()
        print("The time of the run:", stop - start)

        if extract_data:        
            dataset = extract_output_ef(ef,base_data,instance_run)  #this one is quite slow!
        scenarios = sputils.ef_scenarios(ef)

    if solution_method == "ph":

        ############### HUB STUFF
        options["xhat_specific_options"] = {"xhat_solver_options":
                                              options["iterk_solver_options"],
                                              "xhat_scenario_dict": \
                                              {"ROOT": "Scen1",
                                               "ROOT_0": "Scen1",
                                               "ROOT_1": "Scen4",
                                               "ROOT_2": "Scen7"},
                                              "csvname": "specific.csv"}
        
        ph = PH(
            options,
            all_scenario_names,
            scenario_creator, scenario_denouement,
            scenario_creator_kwargs = scenario_creator_kwargs
            )
        ph.ph_main()
        print(ph)

        dataset = extract_output_ph(ph,base_data,instance_run)
        scenarios = ph.local_subproblems

    if extract_data:
        plot_figures(base_data,dataset,scenarios,instance_run,solution_method)
        
    if profiling:
        profiler.disable()
        profiler.dump_stats('aim_model.stats')
        stats = pstats.Stats('aim_model.stats')
        stats.sort_stats('tottime').print_stats(20)
        #it seems like it is the PATH-ARC RULE that makes the code slow (many/?most? constraints are defined here!)
        #AIM_Norwegian_Freight_Model\TranspModelClass.py:152(<genexpr>)
        stats.sort_stats('cumtime').print_stats(20)
        #this is done in the scenario_creator