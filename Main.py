# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""
import os
from pyomo.common.tempfiles import TempfileManager
basepath = os.getcwd().replace(os.sep, '/')
TempfileManager.tempdir = basepath+"/temp/pyomo"
#Also try to force this to None and see what happens

#Remember to set the right workingdirectory. Otherwise errors with loading the classes
# os.chdir('C://Users//steffejb//OneDrive - NTNU//Work//GitHub//AIM_Norwegian_Freight_Model//AIM_Norwegian_Freight_Model')
#os.chdir("M:/Documents/GitHub/AIM_Norwegian_Freight_Model") #uncomment this for stand-alone testing of this fille

from Model import TranspModel, RiskInformation
from ExtractResults import OutputData
from Data.Create_Sets_Class import TransportSets
from Data.settings import *
from Data.interpolate import interpolate
from VisualizeResults import visualize_results

from pyomo.environ import *
import time
import sys
import pickle

import cProfile
import pstats

from Utils import Logger

#################################################
#                   user input                  #
#################################################


analysis = "standard"  # ["standard","only_generate_data", "risk", "single_time_period","carbon_price_sensitivity","run_all"]
scenario_tree = "AllScen" #AllScen,4Scen, 9Scen
analysis_type = "SP" #,  'EEV' , 'SP'         expected value probem, expectation of EVP, stochastic program
wrm_strt = False  #use EEV as warm start for SP

# risk parameters
cvar_coeff = 0.3    # \lambda: coefficient for CVaR in mean-CVaR objective
cvar_alpha = 0.8    # \alpha:  indicates how far in the tail we care about risk
# TODO: test if this is working

log_to_file = False
time_periods = None  #[2022,2026,2030] or None for default up to 2050






#################################################
#                   main code                   #
#################################################


def construct_and_solve_SP(base_data,
                            risk_info, 
                            single_time_period=False,
                            time_periods = None,
                            NoBalancingTrips = None):

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing SP model...")

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.NoBalancingTrips = NoBalancingTrips
    model_instance.single_time_period = single_time_period
    model_instance.construct_model()

    print("Done constructing model.")
    print("Time used constructing the model:", time.time() - start)
    print("----------")
    sys.stdout.flush()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving model...",flush=True)
    start = time.time()
    model_instance.solve_model(#FeasTol=10**(-2),
                               #num_focus=1,
                               #Method=-1,  #i have observed that it gets solved with both primal simplex, dual simplex and barrier. So, standard concurrent option is good
                               ) 
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", end="", flush=True)

    return model_instance,base_data

def construct_and_solve_EEV(base_data,risk_info):


        ############################
        ###  #2: solve EV        ###
        ############################
    
    base_data.S_SCENARIOS = ['BBB']
    base_data.combined_sets()

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing EV model.....", end="", flush=True)

    start = time.time()
    model_instance_EV = TranspModel(data=base_data, risk_info=risk_info)
    #constructing
    model_instance_EV.construct_model()

    print("Done constructing EV model.",flush=True)
    print("Time used constructing the model:", time.time() - start,flush=True)
    print("----------", flush=True)


    #  ---------  SOLVE MODEL  ---------    #

    print("Solving EV model.....",end="",flush=True)
    start = time.time()
    model_instance_EV.solve_model(FeasTol=(10**(-6)), #need high precision, otherwise it becomes infeasible in EEV
                                  num_focus= 0,  # 0 is automatic, 1 is fast low precision
                                  ) #
    print("Done solving model.")
    print("Time used solving the model:", time.time() - start)
    print("----------",  flush=True)

    # --------- SAVE EV RESULTS -----------

    file_string = "EV_" + scenario_tree
    
    output = OutputData(model_instance_EV.model,base_data)

    with open(r"Data//output//" + file_string+'.pickle', 'wb') as output_file: 
        print("Dumping EV output in pickle file.....", end="",flush=True)
        pickle.dump(output, output_file)
        print("done.",flush=True)
    

        ############################
        ###  #3: solve EEV       ###
        ############################
    
    base_data.S_SCENARIOS = base_data.S_SCENARIOS_ALL
    base_data.combined_sets()

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing EEV model...",end='', flush=True)

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.construct_model()
    model_instance.fix_variables_first_stage(model_instance_EV.model)
    
    print("Done constructing EEV model.",flush=True)
    print("Time used constructing the model:", time.time() - start,flush=True)
    print("----------",  flush=True)


    #  ---------  SOLVE MODEL  ---------    #

    print("Solving EEV model...",end='',flush=True)
    start = time.time()
    #options = option_settings_ef()
    model_instance.solve_model(#FeasTol=10**(-2),
                               #num_focus=1, #put high to make sure that the warm start is feasible
                               #Method = -1,
                               ) 
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------",  flush=True)


    # --------- SAVE EEV RESULTS -----------

    file_string = "EEV_" + scenario_tree
    
    output = OutputData(model_instance.model,base_data)

    with open(r"Data//output//" + file_string+'.pickle', 'wb') as output_file: 
        print("Dumping EEV output in pickle file.....", end="",flush=True)
        pickle.dump(output, output_file)
        print("done.",flush=True)
    
    sys.stdout.flush()


    return model_instance, base_data

def construct_and_solve_SP_warm_start(base_data,
                            risk_info, 
                            ):
    
    model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)

    model_instance.unfix_variables_first_stage()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving SP model with EEV warm start...",end='',flush=True)
    start = time.time()
    model_instance.solve_model(warmstart=True,FeasTol=10**(-2),num_focus=1)
    #  https://support.gurobi.com/hc/en-us/community/posts/4406728832145-Crossover-Takes-Long
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", flush=True)

    return model_instance,base_data

def generate_base_data():
    
    if scenario_tree == 'AllScen':
        sheet_name_scenarios = 'scenarios_base' 
    elif scenario_tree == '4Scen':
        sheet_name_scenarios = 'three_scenarios_new' 
    elif scenario_tree == '9Scen':
        sheet_name_scenarios = 'nine_scenarios' 
    
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios) 
    time_periods = [2023, 2028, 2034, 2040, 2050]   # new time periods
    num_first_stage_periods = 2                                 # how many of the periods above are in first stage
    base_data = interpolate(base_data, time_periods, num_first_stage_periods)
    
    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?
    base_data.risk_information = risk_info

    base_data.S_SCENARIOS = base_data.S_SCENARIOS_ALL
    
    if time_periods == None:
        base_data.T_TIME_PERIODS = base_data.T_TIME_PERIODS_ALL
    else:
        base_data.T_TIME_PERIODS = time_periods 

    base_data.combined_sets()

    with open(r'Data//base_data//'+scenario_tree+'.pickle', 'wb') as data_file: 
        pickle.dump(base_data, data_file)
    
    return base_data

def main(analysis_type,
         risk_aversion=None,
         cvar_coeff=cvar_coeff,
         cvar_alpha=cvar_alpha,
         single_time_period=None,
         NoBalancingTrips=False,
         co2_factor = 1,
         ):
    
    #     --------- Setup  ---------   #

    if scenario_tree == 'AllScen':
        sheet_name_scenarios = 'scenarios_base' 
    elif scenario_tree == '4Scen':
        sheet_name_scenarios = 'three_scenarios_new' 
    elif scenario_tree == '9Scen':
        sheet_name_scenarios = 'nine_scenarios' 
        
    run_identifier = analysis_type + '_' + scenario_tree

    run_identifier2 = run_identifier
    if wrm_strt:
        run_identifier2 = run_identifier2 +'_WrmStrt'

    sys.stdout = Logger(run_identifier2,log_to_file)


    print('----------------------------')
    print('Doing the following analysis: ')
    print(analysis_type)
    print(scenario_tree)
    if risk_aversion is not None:
        print(risk_aversion)
    if co2_factor!=1:
        print("CO2 factor: ", co2_factor )
    if wrm_strt:
        print('Using EEV warm start')
    print('----------------------------')
    sys.stdout.flush()
    
    #     --------- DATA  ---------   #
    
            
    print("Reading data...", flush=True)
    start = time.time()
    base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios,co2_factor=co2_factor) 
    # define new timeline
    time_periods = [2023, 2028, 2034, 2040, 2050]   # new time periods
    num_first_stage_periods = 2                                 # how many of the periods above are in first stage
    base_data = interpolate(base_data, time_periods, num_first_stage_periods)

    print("Done reading data.", flush=True)
    print("Time used reading the base data:", time.time() - start,flush=True)
    sys.stdout.flush()

    if risk_aversion=="averse":
        cvar_alpha = (1-1/len(base_data.S_SCENARIOS))
    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    #add to the base_data class?
    base_data.risk_information = risk_info
    
    if single_time_period is not None:
        base_data.single_time_period = single_time_period
        base_data.combined_sets()

    #     --------- MODEL  ---------   #
    # solve model
    if analysis_type == "SP":
        if wrm_strt:
            model_instance,base_data = construct_and_solve_SP_warm_start(base_data,risk_info,time_periods=time_periods)
        else:
            model_instance,base_data = construct_and_solve_SP(base_data,risk_info,time_periods=time_periods)
    elif analysis_type == "EEV":
        model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)
    else:
        Exception('analysis type feil = '+analysis_type)
    #  --------- SAVE OUTPUT ---------    #
    scenario_tree2 = scenario_tree
    if single_time_period is not None:
        scenario_tree2 = scenario_tree2 + "_single_time_period_"+str(single_time_period)
    if co2_factor!=1:
        scenario_tree2 = scenario_tree2 + "_co2_factor_" + str(co2_factor)
    print("Dumping data in pickle file...", end="")
    with open(r'Data//base_data//'+scenario_tree2+'.pickle', 'wb') as data_file: 
        pickle.dump(base_data, data_file)
    print("done.")

    #-----------------------------------
    
    output = OutputData(model_instance.model,base_data)

    run_identifier2 = run_identifier
    if NoBalancingTrips:
        run_identifier2 = run_identifier2 +'_NoBalancingTrips'
    if risk_aversion is not None:
        run_identifier2 = run_identifier2 + '_' + risk_aversion
    if single_time_period is not None:
        run_identifier2 = run_identifier2 + "_single_time_period_"+str(single_time_period)
    if co2_factor!=1:
        run_identifier2 = run_identifier2 + "_co2_factor_" + str(co2_factor)
    with open(r"Data//output//" + run_identifier2+'.pickle', 'wb') as output_file: 
        print("Dumping output in pickle file.....", end="")
        pickle.dump(output, output_file)
        print("done.")
    
    sys.stdout.flush()

    #  --------- VISUALIZE RESULTS ---------    #

    visualize_results(analysis_type,scenario_tree,
                        noBalancingTrips=NoBalancingTrips,
                        single_time_period=single_time_period,
                        risk_aversion=risk_aversion,
                        scen_analysis_carbon = False,
                        carbon_factor = co2_factor
                      )

def risk_analysis():
    for risk_avers in ["neutral","averse"]:
            if risk_avers == "neutral":
                cvar_coeff=0 #not being used
            elif risk_avers == "averse":
                cvar_coeff=0.99
                #cvar_alpha = 1/N
            main(analysis_type="SP",cvar_coeff=cvar_coeff, risk_aversion=risk_avers)

if __name__ == "__main__":
    
    
    if analysis == "only_generate_data":
        base_data = generate_base_data()
    elif analysis == "risk":
        risk_analysis()
    elif analysis == "standard":
        main(analysis_type=analysis_type)
    elif analysis == "single_time_period":
        main(analysis_type=analysis_type,single_time_period=2034)
        main(analysis_type=analysis_type,single_time_period=2050)
    elif analysis == "no_balancing_trips":
        main(analysis_type=analysis_type,NoBalancingTrips=True)
    elif analysis == "carbon_price_sensitivity":
        for carbon_factor in [0,2]:
            main(analysis_type="SP",co2_factor=carbon_factor)
    elif analysis=="run_all":
        #main(analysis_type="EEV")
        #main(analysis_type="SP")
        main(analysis_type=analysis_type,single_time_period=2034)
        main(analysis_type=analysis_type,single_time_period=2050)
        risk_analysis()
        for carbon_factor in [0,2]:
            main(analysis_type="SP",co2_factor=carbon_factor)

    # if profiling:
        #     profiler = cProfile.Profile()
        #     profiler.enable()
        


        
