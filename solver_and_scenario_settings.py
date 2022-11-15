# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:24:04 2022

@author: steffejb
"""

import mpisppy.utils.sputils as sputils
from TranspModelClass import TranspModel
import mpisppy.scenario_tree as scenario_tree
import copy


def scenario_creator(scenario_name, **kwargs):
    
    base_data = kwargs.get('base_data')
    
    base_data.update_scenario_dependent_parameters(scenario_name)
    
    #deepcopy is slower than repetitively constructing the models.
    model_instance = TranspModel(data=base_data)
    model_instance.construct_model()
    model = model_instance.model
    
    first_stage = [2020, 2025]
    sputils.attach_root_node(model, sum(model.StageCosts[t] for t in first_stage),
                                         [model.x_flow[:,:,:,:,:,:,t] for t in first_stage]+
                                         [model.b_flow[:,:,:,:,:,:,t] for t in first_stage]+ 
                                         [model.h_flow[:,:,t] for t in first_stage]+
                                         [model.h_flow_balancing[:,:,t] for t in first_stage]+
                                         [model.q_transp_amount[:,:,t] for t in first_stage]+
                                         [model.q_max_transp_amount[:,:,t] for t in first_stage]+
                                         [model.y_charge[:, :, :, :, :, t] for t in first_stage]+
                                         [model.epsilon_edge[:,:,:,:,t] for t in first_stage]+
                                         [model.upsilon_upg[:,:,:,:,:,t] for t in first_stage]+ 
                                         [model.nu_node[:,:,:,t] for t in first_stage]+
                                         [model.z_emission[t] for t in first_stage] + 
                                         [model.total_emissions[t] for t in first_stage] #TO DO: check what happens if this is not included (AIM had this)
                                         )

    ###### set scenario probabilties if they are not assumed equal######

    model._mpisppy_probability = 0.5 #prob(scenario_nr) #TO DO

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
    
    #options["NumericFocus"] = 3  #  https://www.gurobi.com/documentation/9.5/refman/numericfocus.html
    
    return options

