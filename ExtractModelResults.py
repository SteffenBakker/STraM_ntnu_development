# -*- coding: utf-8 -*-
'''
Created on Thu Sep 22 13:19:20 2022

@author: steffejb
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo

from Data.settings import *

class OutputData():
    #ef,base_data,instance_run,EV_problem
    def __init__(self,ef,base_data,instance_run,EV_problem):# or (self)
        
        self.instance_run = instance_run
        
        self.scenarios = []
        if EV_problem:
            self.scenarios = ['MMM']
        else:
            for scen in sputils.ef_scenarios(ef):
                self.scenarios.append(scen[0])
        self.ob_function_value = pyo.value(ef.EF_Obj)
        
        self.all_variables,self.costs,self.x_flow,self.b_flow,self.h_path,self.y_charging,self.nu_node,self.epsilon_edge,self.upsilon_upgrade, \
            self.z_emission_violation,self.total_emissions,self.q_transp_amount,self.q_max_transp_amount = extract_model_results(base_data,ef,EV_problem)

        
def extract_model_results(base_data,ef,EV_problem):  #currently only for extensive form
    
    scenario_names_and_models = []
    if EV_problem:
        scenario_names_and_models.append(('MMM',ef))
    else:
        for scen in sputils.ef_scenarios(ef):
            scenario_names_and_models.append((scen[0],scen[1]))
    scenario_names = [snm[0] for snm in scenario_names_and_models]


    x_flow =               pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','product','time_period','weight', 'scenario'])
    b_flow =               pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','vehicle_type','time_period','weight', 'scenario'])
    h_path =               pd.DataFrame(columns = ['variable','path','product','time_period','weight', 'scenario'])
    y_charging =           pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','time_period','weight','scenario'])
    nu_node =               pd.DataFrame(columns = ['variable','from', 'terminal_type','mode', 'time_period', 'weight', 'scenario'])
    epsilon_edge =               pd.DataFrame(columns = ['variable','from','to','mode','route','time_period','weight','scenario'])
    upsilon_upgrade =            pd.DataFrame(columns = ['variable','from', 'to', 'mode', 'route','fuel', 'time_period', 'weight', 'scenario'])
    z_emission_violation = pd.DataFrame(columns = ['variable','time_period','weight','scenario'])
    total_emissions =      pd.DataFrame(columns = ['variable','time_period','weight','scenario'])
    q_transp_amount = pd.DataFrame(columns = ['variable','mode','fuel','time_period','weight','scenario'])
    q_max_transp_amount = pd.DataFrame(columns = ['variable','mode','fuel','weight','scenario'])
    
    
    
    vars = ["TranspOpexCost","TranspCO2Cost","TranspOpexCostB","TranspCO2CostB","TransfCost","EdgeCost","NodeCost","UpgCost","ChargeCost"]
    costs = {var:{(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in scenario_names} for var in vars}
    costs['MaxTranspPenaltyCost'] = {scen:0 for scen in scenario_names}

    for (scen_name,scen_model) in scenario_names_and_models:
        # (scen_name,scen_model)=scenario_names_and_models[0]
        modell = scen_model

        variable = 'x_flow'
        for (i,j,m,r) in base_data.A_ARCS:
            a = (i,j,m,r)
            for f in base_data.FM_FUEL[m]:
                for t in base_data.T_TIME_PERIODS:
                    for p in base_data.P_PRODUCTS:
                        weight = modell.x_flow[(a,f,p,t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,i,j,m,r,f,p,t,weight, scen_name], index=x_flow.columns)
                            x_flow = pd.concat([x_flow, a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'b_flow'
        for (i,j,m,r) in base_data.A_ARCS:
            a = (i,j,m,r)
            for f in base_data.FM_FUEL[m]:
                for t in base_data.T_TIME_PERIODS:
                    for v in base_data.VEHICLE_TYPES_M[m]:
                        weight = modell.b_flow[(a,f,v,t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,i,j,m,r,f,v,t,weight, scen_name], index=b_flow.columns)
                            b_flow = pd.concat([b_flow, a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'h_path'
        for kk in base_data.K_PATHS:
            #k = K_PATH_DICT[kk]
            for t in base_data.T_TIME_PERIODS:
                for p in base_data.P_PRODUCTS:
                    weight = modell.h_path[(kk, p, t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,kk, p, t, weight, scen_name], index=h_path.columns)
                        h_path = pd.concat([h_path,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'epsilon_edge'
        for t in base_data.T_TIME_PERIODS:
            for i,j,m,r in base_data.E_EDGES_RAIL:
                e = (i,j,m,r)
                weight = modell.epsilon_edge[(e, t)].value
                if weight > 0:
                    a_series = pd.Series([variable,i,j,m,r, t, weight, scen_name], index=epsilon_edge.columns)
                    epsilon_edge = pd.concat([epsilon_edge,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'upsilon_upg'
        for t in base_data.T_TIME_PERIODS:
            for (e,f) in base_data.U_UPGRADE:
                (i,j,m,r) = e
                weight = modell.upsilon_upg[(i,j,m,r,f,t)].value
                if weight > 0:
                    a_series = pd.Series([variable,i,j,m,r, f,t, weight, scen_name],index=upsilon_upgrade.columns)
                    upsilon_upgrade = pd.concat([upsilon_upgrade,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'nu_node'
        for t in base_data.T_TIME_PERIODS:
            for (i, m) in base_data.NM_LIST_CAP:
                for c in base_data.TERMINAL_TYPE[m]:
                    weight = modell.nu_node[(i, c, m, t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i, c, m, t, weight, scen_name],index=nu_node.columns)
                        nu_node = pd.concat([nu_node,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'y_charging'
        for t in base_data.T_TIME_PERIODS:
            for (e,f) in base_data.EF_CHARGING:
                (i,j,m,r) = e
                weight = modell.y_charge[(i,j,m,r,f,t)].value
                if weight > 0:
                    a_series = pd.Series([variable,i,j,m,r,f,t, weight, scen_name],index=y_charging.columns)
                    y_charging = pd.concat([y_charging,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'z_emission'
        for t in base_data.T_TIME_PERIODS:
            weight = modell.z_emission[t].value
            a_series = pd.Series([variable,t, weight, scen_name],index=z_emission_violation.columns)
            z_emission_violation = pd.concat([z_emission_violation,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'total_emissions'
        for t in base_data.T_TIME_PERIODS:
            weight5 = modell.total_emissions[t].value
            a_series2 = pd.Series([variable,t, weight5, scen_name],index=total_emissions.columns)
            total_emissions = pd.concat([total_emissions,a_series2.to_frame().T],axis=0, ignore_index=True)    
        variable = 'q_transp_amount'
        for m in base_data.M_MODES:
            for f in base_data.FM_FUEL[m]:
                for t in base_data.T_TIME_PERIODS:
                    weight = modell.q_transp_amount[(m, f, t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,m, f, t, weight, scen_name], index=q_transp_amount.columns)
                        q_transp_amount = pd.concat([q_transp_amount,a_series.to_frame().T],axis=0, ignore_index=True)
        variable = 'q_max_transp_amount'
        for m in base_data.M_MODES:
            for f in base_data.FM_FUEL[m]:
                weight = modell.q_max_transp_amount[(m, f)].value
                if weight > 0:
                    a_series = pd.Series([variable,m, f, weight, scen_name], index=q_max_transp_amount.columns)
                    q_max_transp_amount = pd.concat([q_max_transp_amount,a_series.to_frame().T],axis=0, ignore_index=True)
        
        all_variables = pd.concat([x_flow,b_flow,h_path,y_charging,nu_node,epsilon_edge,upsilon_upgrade,
                    z_emission_violation,total_emissions,q_transp_amount,q_max_transp_amount],ignore_index=True)

        for t in base_data.T_TIME_PERIODS:
            for var in vars:
                costs[var][(t,scen_name)] = getattr(modell,str(var))[t].value
        costs['MaxTranspPenaltyCost'][scen_name] = modell.MaxTranspPenaltyCost.value

        return (all_variables, costs, x_flow,b_flow,h_path,y_charging,nu_node,epsilon_edge,upsilon_upgrade,z_emission_violation,total_emissions,q_transp_amount,q_max_transp_amount)

            
            
    

