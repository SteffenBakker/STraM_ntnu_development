# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:24:04 2022

@author: steffejb
"""

import mpisppy.utils.sputils as sputils
from TranspModelClass import TranspModel
import mpisppy.scenario_tree as scenario_tree


def scenario_creator(scenario_name, **kwargs):
    
    CO2_price = kwargs.get('CO2_price')
    fuel_cost = kwargs.get('fuel_cost')
    emission_reduction = kwargs.get('emission_reduction')
    probabilities = kwargs.get('probabilities')
    
    TranspM = TranspModel(instance='TestInstance', one_time_period=False, scenario = scenario_name,
                          carbon_scenario = CO2_price, fuel_costs=fuel_cost, emission_reduction=emission_reduction)

    model = TranspM.construct_model()
    
    sputils.attach_root_node(model, sum(model.StageCosts[t] for t in [2020, 2025]),
                                        [model.x_flow[:,:,:,:,:,:,2020], model.z_inv_cap[:,:,:,:,2020],
                                         model.z_inv_upg[:,:,:,:,:,2020], model.z_inv_node[:,:,:,2020],
                                        model.charge_link[:, :, :, :, :, 2020],model.h_flow[:,:,2020],
                                         model.emission_violation[2020],
                                         model.x_flow[:,:,:,:,:,:,2025], model.z_inv_cap[:,:,:,:,2025],
                                         model.z_inv_upg[:,:,:,:,:,2025], model.z_inv_node[:,:,:,2025],
                                         model.charge_link[:,:,:,:,:,2025],model.h_flow[:,:,2025],
                                         model.emission_violation[2025]])

    ###### set scenario probabilties if they are not assumed equal######

    if probabilities == "low_test":
        if scenario_name == "HHH" or scenario_name == "MMM":
            model._mpisppy_probability = 0.15
        else:
            model._mpisppy_probability = 0.70

    if probabilities == "no_extremes":
        model._mpisppy_probability = 1/6

    if probabilities == "no_extremes_high":
        if scenario_name == "LLH" or scenario_name == "HLL" or scenario_name == "LHL":
            model._mpisppy_probability = 0.1
        else:
            model._mpisppy_probability = 0.7/3

    if probabilities == "low_extremes":
        if scenario_name == "HHH" or scenario_name == "LLL":
            model._mpisppy_probability = 0.02
        else:
            model._mpisppy_probability = 0.16

    if probabilities == "low":
        if scenario_name == "HHL" or scenario_name == "HLH" or scenario_name == "LHH":
            model._mpisppy_probability = 0.09
        if scenario_name == "HHH":
            model._mpisppy_probability = 0.03
        if scenario_name == "LHL" or scenario_name == "HLL" or scenario_name == "LLH":
            model._mpisppy_probability = 0.20
        if scenario_name == "LLL":
            model._mpisppy_probability = 0.10

    if probabilities == "high":
        if scenario_name == "LHL" or scenario_name == "LLH" or scenario_name == "HLL":
            model._mpisppy_probability = 0.09
        if scenario_name == "LLL":
            model._mpisppy_probability = 0.03
        if scenario_name == "HHL" or scenario_name == "HLH" or scenario_name == "LHH":
            model._mpisppy_probability = 0.20
        if scenario_name == "HHH":
            model._mpisppy_probability = 0.10

    return model


def scenario_denouement(rank, scenario_name, scenario):
    pass

def option_settings():
    options = {}
    options["asynchronousPH"] = False
    options["solvername"] = "gurobi"
    options["PHIterLimit"] = 2
    options["defaultPHrho"] = 1
    options["convthresh"] = 0.0001
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = True
    options["display_progress"] = True
    options["iter0_solver_options"] = None
    options["iterk_solver_options"] = None
    
    return options

def get_all_scenario_names(scenario_structure):
    all_scenario_names = list()
    if scenario_structure == 1:
        all_scenario_names = ['HHH', 'LLL', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH']
    elif scenario_structure == 2:
        all_scenario_names = ["HHH", "MMM", "LLL"]
    elif scenario_structure == 3:
        all_scenario_names = data.all_scenarios
    elif scenario_structure == 21:
        all_scenario_names = ['HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH']
    elif scenario_structure == 4: #deterministic equivalent to scen_struct 1/8scen
        all_scenario_names = ['AVG1','AVG11']
    elif scenario_structure == 5: #deterministic equivalent to scen_struct 2/3scen
        all_scenario_names = ['AVG2', 'AVG22']
    elif scenario_structure == 6: #deterministic equivalent to scen_struct 3
        all_scenario_names = ['AVG3', 'AVG33']
        
    return all_scenario_names