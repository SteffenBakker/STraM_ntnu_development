# -*- coding: utf-8 -*-
'''
Created on Thu Sep 22 13:19:20 2022

@author: steffejb
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import math

from Data.settings import *

class OutputData():

    def __init__(self,modell,base_data,EV_problem=False):# or (self)
        
        self.all_variables = None
        self.costs = None
        self.x_flow = None
        self.b_flow=None
        self.h_path=None
        self.y_charging=None
        self.nu_node=None
        self.epsilon_edge=None
        self.upsilon_upgrade= None
        self.q_transp_amount=None
        self.q_max_transp_amount=None
        self.FirstStageCosts = None
        
        self.ob_function_value = pyo.value(modell.objective_function)
                
        self.extract_model_results(base_data,modell, EV_problem)

    def extract_model_results(self,base_data,modell, EV_problem):  #currently only for extensive form
        
        scenario_names = base_data.S_SCENARIOS

        x_flow =               pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','product','time_period','weight', 'scenario'])
        b_flow =               pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','vehicle_type','time_period','weight', 'scenario'])
        h_path =               pd.DataFrame(columns = ['variable','path','product','time_period','weight', 'scenario'])
        y_charging =           pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','time_period','weight','scenario'])
        nu_node =               pd.DataFrame(columns = ['variable','from','mode', 'time_period', 'weight', 'scenario'])
        epsilon_edge =               pd.DataFrame(columns = ['variable','from','to','mode','route','time_period','weight','scenario'])
        upsilon_upgrade =            pd.DataFrame(columns = ['variable','from', 'to', 'mode', 'route','fuel', 'time_period', 'weight', 'scenario'])
        q_transp_amount = pd.DataFrame(columns = ['variable','mode','fuel','time_period','weight','scenario'])
        q_max_transp_amount = pd.DataFrame(columns = ['variable','mode','fuel','weight','scenario'])
        
        vars = ["TranspOpexCost","TranspCO2Cost","CO2_PENALTY","TranspOpexCostB","TranspCO2CostB","TranspTimeCost","TransfCost","EdgeCost","NodeCost","UpgCost","ChargeCost", "FillingCost"] #the same structure, easy to read
        costs = {var:{(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in scenario_names} for var in vars}
        costs['MaxTranspPenaltyCost'] = {scen:0 for scen in scenario_names}

        self.SecondStageCosts = {scen_name:None for scen_name in scenario_names}
        self.CvarPosPart = {scen_name:None for scen_name in scenario_names}
        self.MaxTranspPenaltyCost = {scen_name:None for scen_name in scenario_names}
    
        
            #############
            # Variables #
            #############

        for scen_name in scenario_names:

            variable = 'x_flow'
            for (i,j,m,r) in base_data.A_ARCS:
                a = (i,j,m,r)
                for f in base_data.FM_FUEL[m]:
                    for t in base_data.T_TIME_PERIODS:
                        for p in base_data.P_PRODUCTS:
                            weight = modell.x_flow[(a,f,p,t,scen_name)].value
                            if weight is not None: 
                                if weight != 0:
                                    a_series = pd.Series([variable,i,j,m,r,f,p,t,weight, scen_name], index=x_flow.columns)
                                    x_flow = pd.concat([x_flow, a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'b_flow'
            for (i,j,m,r) in base_data.A_ARCS:
                a = (i,j,m,r)
                for f in base_data.FM_FUEL[m]:
                    for t in base_data.T_TIME_PERIODS:
                        for v in base_data.VEHICLE_TYPES_M[m]:
                            weight = modell.b_flow[(a,f,v,t,scen_name)].value
                            if weight is not None: 
                                if weight != 0:
                                    a_series = pd.Series([variable,i,j,m,r,f,v,t,weight, scen_name], index=b_flow.columns)
                                    b_flow = pd.concat([b_flow, a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'h_path'
            for kk in base_data.K_PATHS:
                #k = K_PATH_DICT[kk]
                for t in base_data.T_TIME_PERIODS:
                    for p in base_data.P_PRODUCTS:
                        weight = modell.h_path[(kk, p, t,scen_name)].value
                        if weight is not None: 
                            if weight > 0:
                                a_series = pd.Series([variable,kk, p, t, weight, scen_name], index=h_path.columns)
                                h_path = pd.concat([h_path,a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'epsilon_edge'
            for (i,j,m,r,t) in base_data.ET_INV:
                    e = (i,j,m,r)
                    weight = modell.epsilon_edge[(e, t,scen_name)].value
                    if weight is None:
                        weight = 0    
                    a_series = pd.Series([variable,i,j,m,r, t, weight, scen_name], index=epsilon_edge.columns)
                    epsilon_edge = pd.concat([epsilon_edge,a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'upsilon_upg'
            for (e,f,t) in base_data.UT_UPG:
                    (i,j,m,r) = e
                    weight = modell.upsilon_upg[(i,j,m,r,f,t,scen_name)].value
                    if math.isnan(weight):
                        raise Exception('cannot be na')
                    if weight is None:
                        weight = 0
                    a_series = pd.Series([variable,i,j,m,r, f,t, weight, scen_name],index=upsilon_upgrade.columns)
                    upsilon_upgrade = pd.concat([upsilon_upgrade,a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'nu_node'
            for (i,m,t) in base_data.NM_CAP_INCR_T:
                weight = modell.nu_node[(i, m, t,scen_name)].value
                if weight is None:
                    weight = 0
                a_series = pd.Series([variable,i, m, t, weight, scen_name],index=nu_node.columns)
                nu_node = pd.concat([nu_node,a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'y_charging'
            for (e,f,t) in base_data.EFT_CHARGE:
                (i,j,m,r) = e
                weight = modell.y_charge[(i,j,m,r,f,t,scen_name)].value
                if weight is None:
                    weight = 0
                a_series = pd.Series([variable,i,j,m,r,f,t, weight, scen_name],index=y_charging.columns)
                y_charging = pd.concat([y_charging,a_series.to_frame().T],axis=0, ignore_index=True)
            variable = 'q_transp_amount'
            for m in base_data.M_MODES:
                for f in base_data.FM_FUEL[m]:
                    for t in base_data.T_TIME_PERIODS:
                        weight = modell.q_transp_amount[(m, f, t,scen_name)].value
                        if weight is not None:
                            if weight > 0:
                                a_series = pd.Series([variable,m, f, t, weight, scen_name], index=q_transp_amount.columns)
                                q_transp_amount = pd.concat([q_transp_amount,a_series.to_frame().T],axis=0, ignore_index=True)

            
            #########
            # COSTS #
            #########

            for t in base_data.T_TIME_PERIODS:
                for var in vars:
                    weight = getattr(modell,str(var))[(t,scen_name)].value
                    if var in ["ChargeCost", "FillingCost"]:
                        weight = weight*sum(base_data.D_DISCOUNT_RATE**n for n in base_data.Y_YEARS[t])
                        #print(var,t,weight)
                    costs[var][(t,scen_name)] = weight

            all_variables = pd.concat([x_flow,b_flow,h_path,y_charging,nu_node,epsilon_edge,upsilon_upgrade,
                        q_transp_amount,q_max_transp_amount],ignore_index=True)


            #SECOND STAGE COSTS
            self.SecondStageCosts[scen_name] = modell.SecondStageCosts[scen_name].value
            self.CvarPosPart[scen_name] = modell.CvarPosPart[scen_name].value
            
        #########
        # ----- #
        #########

        self.all_variables = all_variables
        self.costs = costs
        self.x_flow = x_flow
        self.b_flow=b_flow
        self.h_path=h_path
        self.y_charging=y_charging
        self.nu_node=nu_node
        self.epsilon_edge=epsilon_edge
        self.upsilon_upgrade= upsilon_upgrade

        self.q_transp_amount=q_transp_amount
        self.q_max_transp_amount=q_max_transp_amount
        
        #just take from the last run as it is a first stage decision
        self.FirstStageCosts = modell.FirstStageCosts[base_data.S_SCENARIOS[0]].value 
        self.CvarAux = modell.CvarAux.value
        


        
if False: #alternative for pickle


    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return obj.to_json(orient='records') #
            return json.JSONEncoder.default(self, obj)

    with open(r'Data\\output_vars.json', 'w') as fp:
        json.dump({ '0':output.all_variables,
                    '1':output.x_flow,
                    '2':output.b_flow,
                    '3':output.h_path,
                    '4':output.y_charging,
                    '5':output.nu_node,
                    '6':output.epsilon_edge,
                    '7':output.upsilon_upgrade,
                    '8':output.z_emission_violation,
                    '9':output.total_emissions,
                    '10':output.q_transp_amount,
                    '11':output.q_max_transp_amount
                    },fp, cls=JSONEncoder)


    costs = output.costs
    for key in costs.keys():
        costs[key] = list(costs[key].items())
    
    with open(r'Data\\output_costs.json', "w") as outfile:
        json.dump(costs, outfile)


    #------------------------------------------
    #then we can start reading here 

    costs = json.load(open(r'Data\\output_costs.json'))
    json_file_output_vars = json.load(open(r'Data\\output_vars.json'))
    class Output():
        def __init__(self,costs,json_file_output_vars):
            
            for key in costs.keys():
                if isinstance(costs[key][0][0],list):
                    costs[key] = {tuple(x[0]):x[1] for x in costs[key]}
                else:
                    costs[key] = {x[0]:x[1] for x in costs[key]}
            self.costs = costs

            self.all_variables =            pd.read_json(json_file_output_vars['0'], orient='records')
            self.x_flow =                   pd.read_json(json_file_output_vars['1'], orient='records')
            self.b_flow=                    pd.read_json(json_file_output_vars['2'], orient='records')
            self.h_path=                    pd.read_json(json_file_output_vars['3'], orient='records')
            self.y_charging=                pd.read_json(json_file_output_vars['4'], orient='records')
            self.nu_node=                   pd.read_json(json_file_output_vars['5'], orient='records')
            self.epsilon_edge=              pd.read_json(json_file_output_vars['6'], orient='records')
            self.upsilon_upgrade=           pd.read_json(json_file_output_vars['7'], orient='records')
            self.z_emission_violation=      pd.read_json(json_file_output_vars['8'], orient='records')
            self.total_emissions=           pd.read_json(json_file_output_vars['9'], orient='records')
            self.q_transp_amount=           pd.read_json(json_file_output_vars['10'], orient='records')
            self.q_max_transp_amount=       pd.read_json(json_file_output_vars['11'], orient='records')
    output = Output(costs,json_file_output_vars)

