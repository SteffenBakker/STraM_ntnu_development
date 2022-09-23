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

from Data.settings import *

class OutputData():
    
    def __init__(self,ef,base_data,instance_run):# or (self)
        
        #self.base_data = base_data
        self.instance_run = instance_run
        self.ef = ef
        
        self.x_flow = None
        self.y_charging =  None
        self.z_emission_violation= None
        self.total_emissions = None
        self.w_node = None
        self.v_edge = None
        self.u_upgrade= None
        self.ppqq = None
        self.ppqq_sum = None
        
        self.scenarios = []
        for scen in sputils.ef_scenarios(self.ef):
            self.scenarios.append(scen[0])
        
        self.extract_model_results(base_data)
        self.cost_and_investment_table(base_data)
        
    def extract_model_results(self,base_data):  #currently only for extensive form
        
        self.x_flow =               pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','product','time_period','weight', 'scenario'])
        self.h_path =               pd.DataFrame(columns = ['variable','path','product','time_period','weight', 'scenario'])
        self.y_charging =           pd.DataFrame(columns = ['variable','from','to','mode','route','fuel','time_period','weight','scenario'])
        self.w_node =               pd.DataFrame(columns = ['variable','from', 'terminal_type','mode', 'time_period', 'weight', 'scenario'])
        self.v_edge =               pd.DataFrame(columns = ['variable','from','to','mode','route','time_period','weight','scenario'])
        self.u_upgrade =            pd.DataFrame(columns = ['variable','from', 'to', 'mode', 'route','fuel', 'time_period', 'weight', 'scenario'])
        self.z_emission_violation = pd.DataFrame(columns = ['variable','time_period','weight','scenario'])
        self.total_emissions =      pd.DataFrame(columns = ['variable','time_period','weight','scenario'])
        self.ppqq =                 pd.DataFrame(columns = ['variable','mode' ,'fuel','time_period','weight','scenario']) 
        self.ppqq_sum =             pd.DataFrame(columns = ['variable','mode' ,'time_period','weight','scenario']) 
        self.positive_part_deviations_fuel =  pd.DataFrame(columns = ['variable','mode' ,'fuel','time_period','weight','scenario']) 
        self.positive_part_deviations_sum = pd.DataFrame(columns = ['variable','mode' ,'time_period','weight','scenario']) 
        
        for scen in sputils.ef_scenarios(self.ef):
            modell = scen[1]
            variable = 'x_flow'
            for (i,j,m,r) in base_data.A_ARCS:
                a = (i,j,m,r)
                for f in base_data.FM_FUEL[m]:
                    for t in base_data.T_TIME_PERIODS:
                        for p in base_data.P_PRODUCTS:
                            weight = modell.x_flow[(a,f,p,t)].value
                            if weight > 0:
                                a_series = pd.Series([variable,i,j,m,r,f,p,t,weight, scen[0]], index=self.x_flow.columns)
                                self.x_flow = self.x_flow.append(a_series, ignore_index=True)
            variable = 'h_path'
            for kk in base_data.K_PATHS:
                #k = self.K_PATH_DICT[kk]
                for t in base_data.T_TIME_PERIODS:
                    for p in base_data.P_PRODUCTS:
                        weight = modell.h_flow[(kk, p, t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,kk, p, t, weight, scen[0]], index=self.h_path.columns)
                            self.h_path = self.h_path.append(a_series, ignore_index=True)
            variable = 'v_edge'
            for t in base_data.T_TIME_PERIODS:
                for i,j,m,r in base_data.E_EDGES_RAIL:
                    e = (i,j,m,r)
                    weight = modell.v_edge[(e, t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r, t, weight, scen[0]], index=self.v_edge.columns)
                        self.v_edge = self.v_edge.append(a_series, ignore_index=True)
            variable = 'u_upg'
            for t in base_data.T_TIME_PERIODS:
                for (e,f) in base_data.U_UPGRADE:
                    (i,j,m,r) = e
                    weight = modell.u_upg[(i,j,m,r,f,t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r, f,t, weight, scen[0]],index=self.u_upgrade.columns)
                        self.u_upgrade = self.u_upgrade.append(a_series, ignore_index=True)
            variable = 'w_node'
            for t in base_data.T_TIME_PERIODS:
                for (i, m) in base_data.NM_LIST_CAP:
                    for c in base_data.TERMINAL_TYPE[m]:
                        weight = modell.w_node[(i, c, m, t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,i, c, m, t, weight, scen[0]],index=self.w_node.columns)
                            self.w_node = self.w_node.append(a_series, ignore_index=True)
            variable = 'y_charging'
            for t in base_data.T_TIME_PERIODS:
                for (e,f) in base_data.EF_CHARGING:
                    (i,j,m,r) = e
                    weight = modell.y_charge[(i,j,m,r,f,t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r,f,t, weight, scen[0]],index=self.y_charging.columns)
                        self.y_charging = self.y_charging.append(a_series, ignore_index=True)
            variable = 'z_emission'
            for t in base_data.T_TIME_PERIODS:
                weight = modell.z_emission[t].value
                a_series = pd.Series([variable,t, weight, scen[0]],index=self.z_emission_violation.columns)
                self.z_emission_violation = self.z_emission_violation.append(a_series, ignore_index=True)
            variable = 'total_emissions'
            for t in base_data.T_TIME_PERIODS:
                weight5 = modell.total_emissions[t].value
                a_series2 = pd.Series([variable,t, weight5, scen[0]],index=self.total_emissions.columns)
                self.total_emissions = self.total_emissions.append(a_series2, ignore_index=True)    
            variable = 'ppqq'
            for (m,f,t) in base_data.MFT_MIN0:
                weight = modell.ppqq[(m,f,t)].value
                #if weight > 0:
                a_series = pd.Series([variable,m,f,t, weight, scen[0]],index=self.ppqq.columns)
                self.ppqq = self.ppqq.append(a_series, ignore_index=True)
            variable = 'ppqq_sum'
            for (m,t) in base_data.MT_MIN0:
                weight = modell.ppqq_sum[(m,t)].value
                #if weight > 0:
                a_series = pd.Series([variable,m,t, weight, scen[0]],index=self.ppqq_sum.columns)
                self.ppqq_sum = self.ppqq_sum.append(a_series, ignore_index=True)
            
            for (m,f,t) in base_data.MFT_MIN0:
                weight = modell.ppqq[(m,f,t)].value - modell.q_transp_amount[m,f,t].value - modell.q_transp_amount[m,f,base_data.T_MIN1[t]].value
                a_series = pd.Series([variable,m,f,t, weight, scen[0]],index=self.positive_part_deviations_fuel.columns)
                self.positive_part_deviations_fuel = self.positive_part_deviations_fuel.append(a_series, ignore_index=True)
            
            for (m,t) in base_data.MT_MIN0:
                weight = modell.ppqq_sum[(m,t)].value - (
                    sum(modell.q_transp_amount[m,f,t].value - modell.q_transp_amount[m,f,base_data.T_MIN1[t]].value for f in base_data.FM_FUEL[m]))
                a_series = pd.Series([variable,m,t, weight, scen[0]],index=self.positive_part_deviations_sum.columns)
                self.positive_part_deviations_sum = self.positive_part_deviations_sum.append(a_series, ignore_index=True)
        
            self.all_variables = pd.concat([self.x_flow,self.h_path,self.y_charging,self.w_node,self.v_edge,self.u_upgrade,
                      self.z_emission_violation,self.total_emissions,self.ppqq,self.ppqq_sum],ignore_index=True)
            
            
        
    def cost_and_investment_table(self,base_data):
        

        #FIRST ON YEARLY LEVEL
        
        transport_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        transfer_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        edge_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        node_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        upgrade_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios} 
        charging_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        emission_violation_penalty = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        positive_part_penalty_fuel = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        positive_part_penalty_sum = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in self.scenarios}
        
        self.all_variables['cost_contribution'] = 0
        
        for index, row in self.all_variables.iterrows():
            variable = row['variable']
            i = row['from']
            j = row['to']
            m = row['mode']
            c = row['terminal_type']
            r = row['route']
            f = row['fuel']
            p = row['product']
            kk = row['path']
            value = row['weight']
            t = row['time_period']
            s = row['scenario']
            e = (row['from'],row['to'],row['mode'],row['route'])
            
            cost_contribution = 0
            if variable == 'x_flow':
                cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*(base_data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+
                                                                      base_data.C_CO2[(i,j,m,r,f,p,t)])*value for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
                transport_costs[(t,s)] += cost_contribution
            elif variable == 'h_path':
                cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*base_data.C_TRANSFER[(kk,p)]*value for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
                transfer_costs[(t,s)] += cost_contribution
            elif variable == 'v_edge':
                cost_contribution = base_data.C_EDGE_RAIL[e]*value
                edge_costs[(t,s)] += cost_contribution
            elif variable == 'u_upg':
                cost_contribution = base_data.C_UPG[(e,f)]*value
                upgrade_costs[(t,s)] += cost_contribution
            elif variable == 'w_node':
                cost_contribution = base_data.C_NODE[(i,c,m)]*value
                node_costs[(t,s)] += cost_contribution
            elif variable == 'y_charging':
                cost_contribution = base_data.C_CHARGE[(e,f)]*value
                charging_costs[(t,s)] += cost_contribution
            elif variable == 'z_emission':
                cost_contribution = EMISSION_VIOLATION_PENALTY*value
                emission_violation_penalty[(t,s)] += cost_contribution
            elif variable == 'ppqq':
                cost_contribution = POSITIVE_PART_PENALTY_FUEL*value
                positive_part_penalty_fuel[(t,s)] +=  cost_contribution 
            elif variable == 'ppqq_sum':
                cost_contribution = POSITIVE_PART_PENALTY_SUM*value 
                positive_part_penalty_sum[(t,s)] +=  cost_contribution
            self.all_variables.at[index,'cost_contribution'] = cost_contribution
        
        #%columns_of_interest = self.all_variables.loc[:,('variable','time_period','scenario','cost_contribution')]
        self.aggregated_values =  self.all_variables.groupby(['variable', 'time_period', 'scenario']).agg({'cost_contribution':'sum', 'weight':'sum'})
        #https://stackoverflow.com/questions/46431243/pandas-dataframe-groupby-how-to-get-sum-of-multiple-columns
            
    
    def print_some_insights(self):
        print('The average deviations (in absolute terms) in the positive part approximation are: ')
        print(self.positive_part_deviations_fuel['weight'].mean(), ' for q_fuel AND')
        print(self.positive_part_deviations_sum['weight'].mean(), ' for q_sum ')
        print('Consider setting a lower/higher penalty for computational purposes ')
        
        output.positive_part_deviations_fuel
    
    
        #THEN DISCOUNTED ON 2022 LEVEL
        
        #delta = base_data.D_DISCOUNT_RATE**base_data.Y_YEARS[t][0]
        # TO DO
       

        


#EMISSIONS
#print('--------- Total emissions -----------')
    #print(e[0],t, 'Total emissions: ',modell.total_emissions[t].value,', emission violation: ',
    #    modell.z_emission[t].value,', violation/emission_cap: ', 1-(modell.total_emissions[t].value/(base_data.CO2_CAP[2020])))
#print('Number of variables: ',modell.nvariables())
#print('Number of constraints: ',modell.nconstraints())

        



#PLAY AROUND

#output.h_path['weight'].describe()   #mean 3.2
#output.x_flow['weight'].describe()  # mean 3911
 
        
