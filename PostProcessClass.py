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
    
    def __init__(self,ef,data,instance_run):# or (self)
        
        self.data = data
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
        
        self.extract_model_results()
        
    def extract_model_results(self):  #currently only for extensive form
        
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
        
        for scen in sputils.ef_scenarios(self.ef):
            modell = scen[1]
            variable = 'x_flow'
            for (i,j,m,r) in self.data.A_ARCS:
                a = (i,j,m,r)
                for f in self.data.FM_FUEL[m]:
                    for t in self.data.T_TIME_PERIODS:
                        for p in self.data.P_PRODUCTS:
                            weight = modell.x_flow[(a,f,p,t)].value*self.data.AVG_DISTANCE[a]
                            if weight > 0:
                                a_series = pd.Series([variable,i,j,m,r,f,p,t,weight, scen[0]], index=self.x_flow.columns)
                                self.x_flow = self.x_flow.append(a_series, ignore_index=True)
            variable = 'h_path'
            for kk in self.data.K_PATHS:
                #k = self.K_PATH_DICT[kk]
                for t in self.data.T_TIME_PERIODS:
                    for p in self.data.P_PRODUCTS:
                        weight = modell.h_flow[(kk, p, t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,kk, p, t, weight, scen[0]], index=self.h_path.columns)
                            self.h_path = self.h_path.append(a_series, ignore_index=True)
            variable = 'v_edge'
            for t in self.data.T_TIME_PERIODS:
                for i,j,m,r in self.data.E_EDGES_RAIL:
                    e = (i,j,m,r)
                    weight = modell.v_edge[(e, t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r, t, weight, scen[0]], index=self.v_edge.columns)
                        self.v_edge = self.v_edge.append(a_series, ignore_index=True)
            variable = 'u_upg'
            for t in self.data.T_TIME_PERIODS:
                for (e,f) in self.data.U_UPGRADE:
                    (i,j,m,r) = e
                    weight = modell.u_upg[(i,j,m,r,f,t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r, f,t, weight, scen[0]],index=self.u_upgrade.columns)
                        self.u_upgrade = self.u_upgrade.append(a_series, ignore_index=True)
            variable = 'w_node'
            for t in self.data.T_TIME_PERIODS:
                for (i, m) in self.data.NM_LIST_CAP:
                    for c in self.data.TERMINAL_TYPE[m]:
                        weight = modell.w_node[(i, c, m, t)].value
                        if weight > 0:
                            a_series = pd.Series([variable,i, c, m, t, weight, scen[0]],index=self.w_node.columns)
                            self.w_node = self.w_node.append(a_series, ignore_index=True)
            variable = 'y_charging'
            for t in self.data.T_TIME_PERIODS:
                for (e,f) in self.data.EF_CHARGING:
                    (i,j,m,r) = e
                    weight = modell.y_charge[(i,j,m,r,f,t)].value
                    if weight > 0:
                        a_series = pd.Series([variable,i,j,m,r,f,t, weight, scen[0]],index=self.y_charging.columns)
                        self.y_charging = self.y_charging.append(a_series, ignore_index=True)
            variable = 'z_emission'
            for t in self.data.T_TIME_PERIODS:
                weight = modell.z_emission[t].value
                a_series = pd.Series([variable,t, weight, scen[0]],index=self.z_emission_violation.columns)
                self.z_emission_violation = self.z_emission_violation.append(a_series, ignore_index=True)
            variable = 'total_emissions'
            for t in self.data.T_TIME_PERIODS:
                weight5 = modell.total_emissions[t].value
                a_series2 = pd.Series([variable,t, weight5, scen[0]],index=self.total_emissions.columns)
                self.total_emissions = self.total_emissions.append(a_series2, ignore_index=True)    
            variable = 'ppqq'
            for (m,f,t) in self.data.MFT_MIN0:
                weight = modell.ppqq[(m,f,t)].value
                if weight > 0:
                    a_series = pd.Series([variable,m,f,t, weight, scen[0]],index=self.ppqq.columns)
                    self.ppqq = self.ppqq.append(a_series, ignore_index=True)
            variable = 'ppqq_sum'
            for (m,t) in self.data.MT_MIN0:
                weight = modell.ppqq_sum[(m,t)].value
                if weight > 0:
                    a_series = pd.Series([variable,m,t, weight, scen[0]],index=self.ppqq_sum.columns)
                    self.ppqq_sum = self.ppqq_sum.append(a_series, ignore_index=True)
            
        
            self.all_variables = pd.concat([self.x_flow,self.h_path,self.y_charging,self.w_node,self.v_edge,self.u_upgrade,
                      self.z_emission_violation,self.total_emissions,self.ppqq,self.ppqq_sum],ignore_index=True)
            
            
        
    def cost_and_investment_table(self):
        

        #FIRST ON YEARLY LEVEL
        
        transport_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        transfer_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        edge_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        node_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        upgrade_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios} 
        charging_costs = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        emission_violation_penalty = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        positive_part_penalty_fuel = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        positive_part_penalty_sum = {(t,scen):0 for t in self.data.T_TIME_PERIODS for scen in self.scenarios}
        
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
                cost_contribution = (self.data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+self.data.C_CO2[(i,j,m,r,f,p,t)])*value
                transport_costs[(t,s)] += cost_contribution
            elif variable == 'h_path':
                cost_contribution = self.data.C_TRANSFER[(kk,p)]*value
                transfer_costs[(t,s)] += cost_contribution
            elif variable == 'v_edge':
                cost_contribution = self.data.C_EDGE_RAIL[e]*value
                edge_costs[(t,s)] += cost_contribution
            elif variable == 'u_upg':
                cost_contribution = self.data.C_UPG[(e,f)]*value
                upgrade_costs[(t,s)] += cost_contribution
            elif variable == 'w_node':
                cost_contribution = self.data.C_NODE[(i,c,m)]*value
                node_costs[(t,s)] += cost_contribution
            elif variable == 'y_charging':
                cost_contribution = self.data.C_CHARGE[(e,f)]*value
                charging_costs[(t,s)] += cost_contribution
            elif variable == 'z_emission':
                cost_contribution = EMISSION_VIOLATION_PENALTY*value
                emission_violation_penalty[(t,s)] += cost_contribution
            elif variable == 'ppqq':
                cost_contribution = POSITIVE_PART_PENALTY*value
                positive_part_penalty_fuel[(t,s)] +=  cost_contribution 
            elif variable == 'ppqq_sum':
                cost_contribution = POSITIVE_PART_PENALTY*value 
                positive_part_penalty_sum[(t,s)] +=  cost_contribution
            self.all_variables.at[index,'cost_contribution'] = cost_contribution
        
        #%columns_of_interest = self.all_variables.loc[:,('variable','time_period','scenario','cost_contribution')]
        self.aggregated_values =  self.all_variables.groupby(['variable', 'time_period', 'scenario']).agg({'cost_contribution':'sum'})
        #https://stackoverflow.com/questions/46431243/pandas-dataframe-groupby-how-to-get-sum-of-multiple-columns
            
    

        
        
    
    
        #THEN DISCOUNTED ON 2022 LEVEL
        
        #delta = self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0]
        # TO DO
       

        


#EMISSIONS
#print('--------- Total emissions -----------')
    #print(e[0],t, 'Total emissions: ',modell.total_emissions[t].value,', emission violation: ',
    #    modell.z_emission[t].value,', violation/emission_cap: ', 1-(modell.total_emissions[t].value/(self.data.CO2_CAP[2020])))
#print('Number of variables: ',modell.nvariables())
#print('Number of constraints: ',modell.nconstraints())

        
        
        
