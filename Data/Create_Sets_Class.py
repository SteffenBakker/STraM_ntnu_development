# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:31:04 2021

@author: ingvi
Adapted by Ruben
"""

"Example data"

import os
#os.chdir('M:/Documents/GitHub/AIM_Norwegian_Freight_Model') #uncomment this for stand-alone testing of this fille
#os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')


from Data.settings import *
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt

from math import cos, asin, sqrt, pi
import networkx as nx
from itertools import islice
pd.set_option("display.max_rows", None, "display.max_columns", None)

from Data.BassDiffusion import BassDiffusion 

# from FreightTransportModel.Utils import plot_all_graphs  #FreightTransportModel.

#Class containing information about all scenarios
#Information in this class can be used to activate a scenario in a TransportSets object, meaning that the corresponding parameter values are changed
class ScenarioInformation():
    def __init__(self, prefix,sheet_name='scenarios_base'): 

        self.prefix = prefix #get prefix from calling object
        self.scenario_file = "scenarios.xlsx" #potentially make this an input parameter to choose a scenario set
               
        #read and process fuel group data
        fuel_group_data = pd.read_excel(self.prefix+self.scenario_file, sheet_name='fuel_groups')

        self.fuel_group_names = [] #list of fuel group names
        self.fuel_groups = {} #dict from fuel group names to (m,f) combinations
        for index, row in fuel_group_data.iterrows():
            if row["Fuel group"] not in self.fuel_group_names:
                self.fuel_group_names = self.fuel_group_names + [row["Fuel group"]]    #add fuel group name to list
                self.fuel_groups[row["Fuel group"]] = [(row["Mode"], row["Fuel"])] #create new entry in dict
            else:
                self.fuel_groups[row["Fuel group"]] = self.fuel_groups[row["Fuel group"]] + [(row["Mode"], row["Fuel"])]  #append entry in dict
        self.mf_to_fg = {} #translate (m,f) to fuel group name
        for fg in self.fuel_groups:
            for (m,f) in self.fuel_groups[fg]:
                self.mf_to_fg[(m,f)] = fg
            

        #read and process scenario data
        scenario_data = pd.read_excel(self.prefix+self.scenario_file, sheet_name=sheet_name)
        self.num_scenarios = len(scenario_data)
        self.scenario_names = ["scen_" + str(i).zfill(len(str(self.num_scenarios))) for i in range(self.num_scenarios)] #initialize as scen_00, scen_01, scen_02, etc.
        self.probabilities = [1.0/self.num_scenarios] * self.num_scenarios #initialize as equal probabilities
        

        self.fg_cost_factor = [{}] * self.num_scenarios
        self.fg_maturity_path_name = [{}] * self.num_scenarios
        for index, row in scenario_data.iterrows():
            if "Name" in scenario_data:
                self.scenario_names[index] = row["Name"] #update scenario names if available
            if "Probability" in scenario_data:
                self.probabilities[index] = row["Probability"] #update probabilities if available
            for fg in self.fuel_group_names:
                new_cost_entry = {fg : row[f"Cost_{fg}"]} #new entry for the dictionary fg_cost_factor[index]
                self.fg_cost_factor[index] = dict(self.fg_cost_factor[index], **new_cost_entry) #add new entry to existing dict (trick from internet)
                new_maturity_entry = {fg : row[f"Maturity_{fg}"]}
                self.fg_maturity_path_name[index] = dict(self.fg_maturity_path_name[index], **new_maturity_entry)

        #make dicts for scenario name to nr and vice versa
        self.scen_name_to_nr = {}
        self.scen_nr_to_name = {}
        for i in range(len(self.scenario_names)):
            self.scen_name_to_nr[self.scenario_names[i]] = i
            self.scen_nr_to_name[i] = self.scenario_names[i]
        
        self.mode_fuel_cost_factor = [] #list of dictionaries (per scenario) from (m,f) pair to transport cost factor (relative to base cost)
        for s in range(self.num_scenarios):
            self.mode_fuel_cost_factor.append({})
            for fg in self.fuel_group_names:
                for mf in self.fuel_groups[fg]:
                    self.mode_fuel_cost_factor[s][mf] = self.fg_cost_factor[s][fg]

        

test_scenario_information = ScenarioInformation('Data/')


#Class containing all relevant data
#Note: also contains the scenario information (in self.scenario_information)
#One scenario can be activated (indicated by self.active_scenario_nr) by the procedure update_scenario_dependent_parameters(self,scenario_nr)
#Activating a scenario means that all relevant parameters are changed to their scenario values
class TransportSets():

    def __init__(self,init_data=False,sheet_name_scenarios='scenarios_base'):# or (self)
        self.run_file = "main"  # "sets" or "main"
        self.prefix = '' 
        if self.run_file == "main":
            self.prefix = r'Data/'
        elif self.run_file =="sets":
            self.prefix = '' 
        
        self.init_data = init_data #set T_TIME_PERIODS to first period
        self.last_time_period = False #only solve last time period -> remove all operational constraints for the other periods

        #read/construct scenario information
        self.active_scenario_name = "benchmark" #no scenario has been activated; all data is from benchmark setting
        self.scenario_information = ScenarioInformation(self.prefix,sheet_name_scenarios) #TODO: check performance of this
        self.scenario_information_EV = ScenarioInformation(self.prefix,'EV_scenario') 

        self.risk_information = None

        #read/construct data                
        self.construct_pyomo_data()
        if init_data:
            self.T_TIME_PERIODS = self.T_TIME_PERIODS_INIT
        self.combined_sets()

    def construct_pyomo_data(self):

        self.pwc_aggr = pd.read_csv(self.prefix+r'demand.csv')
        self.city_coords = pd.read_csv(self.prefix+r'zonal_aggregation.csv', sep=';')


        print("Reading and constructing data")

        self.scaling_factor = SCALING_FACTOR #10E-5
        self.precision_digits = 6

        self.S_SCENARIOS_ALL = self.scenario_information.scenario_names
        self.S_SCENARIOS = self.S_SCENARIOS_ALL

        self.M_MODES = ["Road", "Rail", "Sea"]


        # -----------------------
        # ------- Network--------
        # -----------------------

        sea_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Sea')
        road_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Road')
        rail_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Rail')

        distances_dict = {}

        for index, row in sea_distance.iterrows():
            distances_dict[(row["Fra"],row["Til"],"Sea",int(row["Route"]))] = row["Km - sjø"]
        for index, row in road_distance.iterrows():
            distances_dict[(row["Fra"],row["Til"],"Road",int(row["Route"]))] = row["Km - road"]
        for index, row in rail_distance.iterrows():
            distances_dict[(row["Fra"],row["Til"],"Rail",int(row["Route"]))] = row["Km - rail"]

        self.Utlandet = ['Nord-Sverige','Sør-Sverige','Europa','Verden']
        self.N_NODES = set()
        self.E_EDGES = []
        self.E_EDGES_RAIL = []
        
        self.A_ARCS = []
        
        self.AVG_DISTANCE = {}
        for (i,j,m,r),value in distances_dict.items():
            a1 = (i,j,m,r)
            a2 = (j,i,m,r)
            self.N_NODES.add(i)
            self.N_NODES.add(j)
            self.E_EDGES.append(a1)
            self.A_ARCS.append(a1)
            self.A_ARCS.append(a2)
            self.AVG_DISTANCE[a1] = value
            self.AVG_DISTANCE[a2] = value
            if (i in self.Utlandet) or (j in self.Utlandet):
                self.AVG_DISTANCE[a1] = value/2
                self.AVG_DISTANCE[a2] = value/2
                # half the distance for international transport -> split the emissions and costs
            if m == 'Rail':
                if (i not in self.Utlandet) or (j not in self.Utlandet):
                    self.E_EDGES_RAIL.append(a1)
        self.N_NODES = list(self.N_NODES)
        self.N_NODES_NORWAY = set(self.N_NODES) - set(self.Utlandet)

        self.AE_ARCS = {e:[] for e in self.E_EDGES}
        self.AM_ARCS = {m:[] for m in self.M_MODES}
        for (i,j,m,r) in self.E_EDGES:
            a1 = (i,j,m,r)
            a2 = (j,i,m,r)
            self.AE_ARCS[a1].append(a1)
            self.AE_ARCS[a1].append(a2)
            self.AM_ARCS[m].append(a1)
            self.AM_ARCS[m].append(a2)
        
        
        self.SEA_NODES = self.N_NODES.copy()
        self.SEA_NODES.remove("Hamar")
        
        self.SEA_NODES_NORWAY = self.N_NODES_NORWAY.copy()
        self.SEA_NODES_NORWAY.remove("Kontinentalsokkelen")
        self.SEA_NODES_NORWAY.remove("Hamar")
        
        self.ROAD_NODES = self.N_NODES.copy()
        self.ROAD_NODES.remove("Kontinentalsokkelen")
        self.ROAD_NODES.remove("Verden")
        
        self.RAIL_NODES = self.N_NODES.copy()
        self.RAIL_NODES.remove("Kontinentalsokkelen")
        self.RAIL_NODES.remove("Europa")
        self.RAIL_NODES.remove("Verden")

        self.NM_NODES = {m:None for m in self.M_MODES}
        self.NM_NODES["Road"] = self.ROAD_NODES
        self.NM_NODES["Sea"] = self.SEA_NODES
        self.NM_NODES["Rail"] = self.RAIL_NODES

        self.M_MODES_CAP = ["Rail", "Sea"]
        
        self.RAIL_NODES_NORWAY = self.N_NODES_NORWAY.copy()
        self.RAIL_NODES_NORWAY.remove("Kontinentalsokkelen")
        
        self.N_NODES_CAP_NORWAY = {"Rail": self.RAIL_NODES_NORWAY,
                            "Sea": self.SEA_NODES_NORWAY}


        #####################################
        ## Mode-Fuel stuff
        #####################################

        self.F_FUEL = ["Diesel", "Ammonia", "Hydrogen", "Battery electric", "Electric train (CL)", "LNG", "MGO",
                       'Biogas', 'Biodiesel', 'Biodiesel (HVO)', 'Battery train', "HFO"]

        self.FM_FUEL = {"Road": ["Diesel", "Hydrogen", "Battery electric", 'Biodiesel', 'Biogas'],
                        "Rail": ["Diesel", "Hydrogen", "Battery train", "Electric train (CL)", 'Biodiesel'],  #Hybrid: non-existing
                        "Sea": ["LNG", "MGO", "Hydrogen", "Ammonia", 'Biodiesel (HVO)', 'Biogas', "HFO"]} 

        self.NEW_MF_LIST = [("Road", "Hydrogen"), ("Road", "Battery electric"), ("Rail", "Hydrogen"),
                            ("Rail", "Battery train"), ("Sea", "Hydrogen"), ("Sea", "Ammonia"), ('Road', 'Biodiesel'),
                            ('Road', 'Biogas'), ('Rail', 'Biodiesel'), ('Sea', 'Biodiesel (HVO)'), ('Sea', 'Biogas')]

        self.NEW_F_LIST = set([e[1] for e in self.NEW_MF_LIST])

        self.NM_LIST_CAP = [(node, mode) for mode in self.M_MODES_CAP for node in self.N_NODES_CAP_NORWAY[mode]]
        

        # -----------------------
        # ------- Timing --------
        # -----------------------

        #NOTE: A BUNCH OF HARDCODING IN THE TIME-RELATED SETS BELOW
        #self.T_TIME_PERIODS = [2020, 2025, 2030, 2040, 2050] #(OLD)
        self.T_TIME_PERIODS = [2022, 2026, 2030, 2040, 2050] 
        self.T_MIN1 = {self.T_TIME_PERIODS[tt]:self.T_TIME_PERIODS[tt-1] for tt in range(1,len(self.T_TIME_PERIODS))} 
        self.T_TIME_FIRST_STAGE_BASE = [2022, 2026] 
        self.T_TIME_SECOND_STAGE_BASE = [2030, 2040, 2050] 
        
        #we have to switch between solving only first time period, and all time periods. (to initialize the transport shares and emissions)
        self.T_TIME_PERIODS_ALL = self.T_TIME_PERIODS
        self.T_TIME_PERIODS_INIT = [self.T_TIME_PERIODS[0]]


                
        
        
        
        # -----------------------
        # ------- Other--------
        # -----------------------

        self.P_PRODUCTS = ['Dry bulk', 'Fish', 'General cargo', 'Industrial goods', 'Other thermo',
                           'Timber', 'Wet bulk']
        
        self.TERMINAL_TYPE = {"Rail": ["Combination", "Timber"], "Sea": ["All"]}
        
        self.PT = {"Combination": ['Dry bulk', 'Fish', 'General cargo', 'Industrial goods', 'Other thermo','Wet bulk'],
                   "Timber": ['Timber'],
                   "All": self.P_PRODUCTS}

        
            

        ####################################
        ### ORIGIN, DESTINATION AND DEMAND #
        ####################################


        self.OD_PAIRS = []
        

        #Start with filtering of demand data. What to include and what not.
        D_DEMAND_ALL = {}  #5330 entries, after cutting off the small elements, only 4105 entries

        #then read the pwc_aggr data
        
        #WE ONLY TAKE DEMAND BETWEEN COUNTIES! SO, THIS IS OUR DEFINITION OF LONG-DISTANCE TRANSPORT
        for index, row in self.pwc_aggr.iterrows():
            from_node = row['from_fylke_zone']
            to_node = row['to_fylke_zone']
            product = row['commodity_aggr']
            if from_node !=  to_node and from_node in self.N_NODES and to_node in self.N_NODES:
                #if from_node not in ['Europa', 'Verden','Kontinentalsokkelen']: #TO DO, change back: temporary solution
                D_DEMAND_ALL[(from_node, to_node,product ,int(row['year']))] = round(float(row['amount_tons']),0)
                
        demands = pd.Series(D_DEMAND_ALL.values())         
        
        # DO THIS ANALYSIS AS TON/KM?, opposed to as in TONNES?

        # demands.describe()   #huge spread -> remove the very small stuff. Even demand of 5E-1
        # demands.plot.hist(by=None, bins=1000)
        # demands.hist(cumulative=True, density=1, bins=100)
        
        total_base_demand=round(demands.sum(),0)  #'1.339356e+09' TONNES
        cutoff = demands.quantile(0.25) #HARDCODED, requires some analyses
        demands2 = demands[demands > cutoff]  #and (demands < demands.quantile(0.9)
        reduced_base_demand = round(demands2.sum(),0)  #'1.338888e+09' TONNES
        print('percentage demand removed: ',(total_base_demand-reduced_base_demand)/total_base_demand*100,'%')  #NOT EVEN 0.1% removed!!
        
        # demands2.plot.hist(by=None, bins=1000)
        # demands2.hist(cumulative=True, density=1, bins=100)
        
        # print("{:e}".format(round(cutoff,0)))    # '3.306000e+03'
        # print("{:e}".format(round(demands.max(),0)))           # '2.739037e+07'
    
        D_DEMAND_CUT = {key:value for key,value in D_DEMAND_ALL.items() if value > cutoff}  #(o,d,p,t)
        
        self.OD_PAIRS = {p: [] for p in self.P_PRODUCTS}
        for (o,d,p,t), value in D_DEMAND_CUT.items():
            if ((o,d) not in self.OD_PAIRS[p]):
                self.OD_PAIRS[p].append((o,d))
        self.ODP = []
        self.OD_PAIRS_ALL = set()
        for p in self.P_PRODUCTS:
            for (o, d) in self.OD_PAIRS[p]:
                self.OD_PAIRS_ALL.add((o, d))
                self.ODP.append((o, d, p))
        self.OD_PAIRS_ALL = list(self.OD_PAIRS_ALL)
        

        #demand
        self.D_DEMAND = {(o,d,p,t):0 for t in self.T_TIME_PERIODS for (o,d,p) in self.ODP}        
        for (o,d,p,t), value in D_DEMAND_CUT.items():
            self.D_DEMAND[(o,d,p,t)] = round(value/self.scaling_factor,self.precision_digits)
        
        if INTERPOLATE_DEMAND_DATA_2040:
            for (o,d,p,t), value in self.D_DEMAND.items(): 
                if t == 2040:
                    v30 = self.D_DEMAND[(o,d,p,2030)]
                    v40 = self.D_DEMAND[(o,d,p,2040)]
                    v50 = self.D_DEMAND[(o,d,p,2050)]
                    if  (v30 <= v40 <=v50) or (v30 >= v40 >=v50):
                        pass
                    else:
                        self.D_DEMAND[(o,d,p,2040)] = float(np.mean([v30,v50]))
        
        self.D_DEMAND_AGGR = {t:0 for t in self.T_TIME_PERIODS}
        for (o,d,p,t),value in self.D_DEMAND.items():
            self.D_DEMAND_AGGR[t] += value


        # To do_ What was happening here?
        # self.A_LINKS = {l: [] for l in self.A_ARCS}
        # for (i,j,m,r) in self.A_ARCS:
        #     for f in self.FM_FUEL[m]:
        #         if not (f == "HFO" and i in self.N_NODES_NORWAY and j in self.N_NODES_NORWAY):
        #             self.A_LINKS[(i,j,m,r)].append((i,j,m,f,r))

        # ------------------------
        # ----LOAD ALL PATHS------
        # ------------------------


        self.K_PATHS = []
        all_generated_paths = pd.read_csv(self.prefix+r'generated_paths_Ruben_2_modes.csv', converters={'paths': eval})
        self.K_PATH_DICT = {i:None for i in range(len(all_generated_paths))}
        for index, row in all_generated_paths.iterrows():
            elem = tuple(row['paths']) 
            self.K_PATHS.append(index)
            self.K_PATH_DICT[index]=elem
        
        self.OD_PATHS = {od: [] for od in self.OD_PAIRS_ALL}
        for od in self.OD_PAIRS_ALL:
            for k in self.K_PATHS:
                path = self.K_PATH_DICT[k]
                if od[0] == path[0][0] and od[-1] == path[-1][1]:
                    self.OD_PATHS[od].append(k)

        

        #multi-mode paths and unimodal paths
        self.MULTI_MODE_PATHS = []
        for kk in self.K_PATHS:
            k = self.K_PATH_DICT[kk]
            if len(k) > 1:
                for i in range(len(k)-1):
                    if k[i][2] != k[i+1][2]:
                        self.MULTI_MODE_PATHS.append(kk)
        self.UNI_MODAL_PATHS = list(set(self.K_PATHS)-set(self.MULTI_MODE_PATHS))

        #Paths with transfer in node i to/from mode m
        self.TRANSFER_PATHS = {(i,m) : [] for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m]}
        
        for m in self.M_MODES_CAP:
            for i in self.N_NODES_CAP_NORWAY[m]:
                for kk in self.MULTI_MODE_PATHS:
                    k = self.K_PATH_DICT[kk]
                    for j in range(len(k)-1):
                        if (k[j][1] == i) and (k[j][2] == m or k[j+1][2] == m) and (k[j][2] != k[j+1][2]):
                            self.TRANSFER_PATHS[(i,m)].append(kk)

        #Origin and destination paths
        self.ORIGIN_PATHS = {(i,m): [] for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m]}
        self.DESTINATION_PATHS = {(i,m): [] for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m]}
        
        for m in self.M_MODES_CAP:
            for i in self.N_NODES_CAP_NORWAY[m]:
                for kk in self.K_PATHS:
                    k = self.K_PATH_DICT[kk]
                    if (k[0][0] == i) and (k[0][2] == m):
                        self.ORIGIN_PATHS[(i,m)].append(kk)
                    if (k[-1][1] == i) and (k[-1][2] == m):
                        self.DESTINATION_PATHS[(i,m)].append(kk)
        
        
        self.KA_PATHS = {a:[] for a in self.A_ARCS}
        for kk in self.K_PATHS:
            k = self.K_PATH_DICT[kk]
            for (i,j,m,r) in k:
                a = (i,j,m,r)
                self.KA_PATHS[a].append(kk)

        self.KA_PATHS_UNIMODAL = {a:[] for a in self.A_ARCS}
        for kk in self.UNI_MODAL_PATHS:
            k = self.K_PATH_DICT[kk]
            for (i,j,m,r) in k:
                a = (i,j,m,r)
                self.KA_PATHS_UNIMODAL[a].append(kk)
        
        #Vehicle types
        self.prod_to_vehicle_type = pd.read_excel(self.prefix+r'transport_costs_emissions_raw.xlsx', sheet_name='prod_to_vehicle')
        self.VEHICLE_TYPE_MP = {}
        self.VEHICLE_TYPES_M = {m:[] for m in self.M_MODES}
        for (m,p,v) in zip(self.prod_to_vehicle_type['Mode'], self.prod_to_vehicle_type['Product group'], self.prod_to_vehicle_type['Vehicle type']):
            self.VEHICLE_TYPE_MP[(m,p)] = v
            self.VEHICLE_TYPES_M[m].append(v)
        for m in self.M_MODES:
            self.VEHICLE_TYPES_M[m] = list(set(self.VEHICLE_TYPES_M[m]))
        self.V_VEHICLE_TYPES = list(set(self.VEHICLE_TYPE_MP.values()))
        
        self.PV_PRODUCTS = {v:[] for v in self.V_VEHICLE_TYPES}
        for (m,p),v in self.VEHICLE_TYPE_MP.items():
            self.PV_PRODUCTS[v].append(p)

        self.ANM_ARCS_IN = {(n,m):[] for n in self.N_NODES for m in self.M_MODES}
        self.ANM_ARCS_OUT = {(n,m):[] for n in self.N_NODES for m in self.M_MODES}
        for (i,j,m,r) in self.A_ARCS:
            a = (i,j,m,r)
            self.ANM_ARCS_IN[(j,m)].append(a)
            self.ANM_ARCS_OUT[(i,m)].append(a)



        #------------------------
        "Parameters"
        #-----------------------
        

        self.cost_data = pd.read_excel(self.prefix+r'transport_costs_emissions.xlsx', sheet_name='costs_emissions')
        self.emission_data = pd.read_excel(self.prefix+r'emission_cap.xlsx', sheet_name='emission_cap')
        
        self.EMISSION_CAP_RELATIVE = dict(zip(self.emission_data['Year'], self.emission_data['Percentage']))
        #self.EMISSION_CAP_RELATIVE = {year:round(cap/self.scaling_factor,0) for year,cap in self.EMISSION_CAP_RELATIVE.items()}   #this was max 4*10^13, now 4*10^7
        self.EMISSION_CAP_ABSOLUTE_BASE_YEAR = None
        
        transfer_data = pd.read_excel(self.prefix+r'transport_costs_emissions_raw.xlsx', sheet_name='transfer_costs')
        transfer_data.columns = ['Product', 'Transfer type', 'Transfer cost']
        
        
        self.PATH_TYPES = ["sea-rail", "sea-road", "rail-road"]
        # self.MULTI_MODE_PATHS_DICT = {q: [] for q in self.PATH_TYPES}
        self.C_MULTI_MODE_PATH = {(q,p): 0  for q in self.PATH_TYPES for p in self.P_PRODUCTS}
        for p in self.P_PRODUCTS:
            for q in self.PATH_TYPES:
                data_index = transfer_data.loc[(transfer_data['Product'] == p) & (transfer_data['Transfer type'] == q)]
                self.C_MULTI_MODE_PATH[q,p] = round(data_index.iloc[0]['Transfer cost'],1)  #10E6NOK/10E6TONNES
        
        
        mode_to_transfer = {('Sea','Rail'):'sea-rail',
                            ('Sea','Road'):'sea-road',
                            ('Rail','Road'):'rail-road',
                            ('Rail','Sea'):'sea-rail',
                            ('Road','Sea'):'sea-road',
                            ('Road','Rail'):'rail-road'}            
          
        self.C_TRANSFER = {(k,p):0 for k in self.K_PATHS for p in self.P_PRODUCTS}   #UNIT: NOK/T     MANY ELEMENTS WILL BE ZERO!! (NO TRANSFERS)
        
        for kk in self.MULTI_MODE_PATHS:
            k = self.K_PATH_DICT[kk]
            for p in self.P_PRODUCTS:
                cost = 0
                num_transfers = len(k)-1
                for n in range(num_transfers):
                    mode_from = k[n][2]
                    mode_to = k[n+1][2]
                    if mode_from != mode_to: 
                        cost += self.C_MULTI_MODE_PATH[(mode_to_transfer[(mode_from,mode_to)],p)]
                self.C_TRANSFER[(kk,p)] = round(cost,1)
            
        CO2_fee_data = pd.read_excel(self.prefix+r'transport_costs_emissions_raw.xlsx', sheet_name='CO2_fee')    
        self.CO2_fee = {t: 10000000 for t in self.T_TIME_PERIODS}   #UNIT: nok/gCO2
        for index, row in CO2_fee_data.iterrows():
            self.CO2_fee[row["Year"]] = row["CO2 fee base scenario (nok/gCO2)"]
            

        #base level transport costs (in average scenario)
        self.C_TRANSP_COST_BASE = {(i,j,m,r, f, p, t): 1000000 for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] 
                              for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}   #UNIT: NOK/T
        #scenario-dependent transport cost (computed using base cost)
        self.C_TRANSP_COST_NORMALIZED = {(m,f, p, t): 1000000 for m in self.M_MODES for f in self.FM_FUEL[m] 
                              for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}   #UNIT: NOK/Tkm
        self.E_EMISSIONS_NORMALIZED = {(m,f,p,t): 1000000 for m in self.M_MODES for f in self.FM_FUEL[m] 
                            for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}      #UNIT:  gCO2/T
        self.C_TRANSP_COST = {(i,j,m,r, f, p, t,s): 1000000 for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] 
                              for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS for s in self.S_SCENARIOS}   #UNIT: NOK/T
        self.E_EMISSIONS = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] 
                            for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}      #UNIT:  gCO2/T
        self.C_CO2 = {(i,j,m,r,f,p,t): 1000000 for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] 
                      for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}   #UNIT: nok/T

        for index, row in self.cost_data.iterrows():
            for (i,j,m,r) in self.A_ARCS:
                a = (i, j, m, r)
                if m == row["Mode"]:
                    f = row["Fuel"]
                    if f in self.FM_FUEL[m]: #get rid of the hybrid!!
                        p = row['Product group']
                        y = row['Year']
                        self.C_TRANSP_COST_NORMALIZED[(m,f,p,y)] = row['Cost (NOK/Tkm)']
                        self.E_EMISSIONS_NORMALIZED[(m,f,p,y)] = row['Emissions (gCO2/Tkm)']
                        #compute base cost
                        self.C_TRANSP_COST_BASE[(i, j, m, r, f, p, y)] = round((self.AVG_DISTANCE[a] * self.C_TRANSP_COST_NORMALIZED[(m,f,p,y)]), 2) 
                        #^: MINIMUM 6.7, , median = 114.8, 90%quantile = 2562.9,  max 9.6*10^7!!!
                        self.E_EMISSIONS[(i, j, m, r, f, p, y)] = round(self.AVG_DISTANCE[a] * row['Emissions (gCO2/Tkm)'], 1)
                        #CO2 costs per tonne:
                        self.C_CO2[(i, j, m, r, f, p, y)] =  round(self.E_EMISSIONS[(i, j, m, r, f, p, y)] * self.CO2_fee[row["Year"]], 1)

        for (i, j, m, r) in self.A_ARCS:
                for f in self.FM_FUEL[m]:
                    for p in self.P_PRODUCTS:
                        for y in self.T_TIME_PERIODS:
                            for s in self.S_SCENARIOS:
                                if y in self.T_TIME_FIRST_STAGE_BASE: #only update second-stage costs!
                                    self.C_TRANSP_COST[(i, j, m, r, f, p, y,s)] = self.C_TRANSP_COST_BASE[(i, j, m, r, f, p, y)] * 1 
                                elif y in self.T_TIME_SECOND_STAGE_BASE:
                                #transport cost = base transport cost * cost factor for fuel group associated with (m,f) for current active scenario:
                                    self.C_TRANSP_COST[(i, j, m, r, f, p, y,s)] = self.C_TRANSP_COST_BASE[(i, j, m, r, f, p, y)] * self.scenario_information.mode_fuel_cost_factor[self.scenario_information.scen_name_to_nr[s]][(m,f)] 


        
        #find the "cheapest" product group per vehicle type. 
        self.cheapest_product_per_vehicle = {(m,f,t,v):None for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS for v in self.VEHICLE_TYPES_M[m]}
        for m in self.M_MODES: 
            for f in self.FM_FUEL[m]: 
                for t in self.T_TIME_PERIODS:
                    for v in self.VEHICLE_TYPES_M[m]:
                        cheapest_product = None
                        lowest_cost = 200000000
                        for p in self.PV_PRODUCTS[v]:
                            if self.C_TRANSP_COST_NORMALIZED[(m,f,p,t)] < lowest_cost:
                                lowest_cost = self.C_TRANSP_COST_NORMALIZED[(m,f,p,t)]
                                cheapest_product = p
                        self.cheapest_product_per_vehicle[(m,f,t,v)] = cheapest_product

        #################
        #  INVESTMENTS  #
        #################
        
        rail_cap_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Cap rail')
        inv_rail_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Invest rail')
        inv_sea_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Invest sea')

        self.E_EDGES_UPG = []
        for index, row in inv_rail_data.iterrows():
            if pd.isnull(row["From"]):
                pass
            else:
                (i,j,m,r) = (row["From"],row["To"],row["Mode"],int(row["Route"]))
                edge = (i,j,m,r)
                if (i,j,m,r) not in self.E_EDGES_RAIL:
                    edge = (j,i,m,r)
                self.E_EDGES_UPG.append(edge)   
        
        self.U_UPGRADE=[] #only one option, upgrade to electrify by means of electric train
        for e in self.E_EDGES_UPG:
            self.U_UPGRADE.append((e,'Electric train (CL)')) #removed 'Battery electric' train as an option.

        BIG_COST_NUMBER = 10**12/self.scaling_factor # nord-norge banen = 113 milliarder = 113*10**9
        self.C_EDGE_RAIL = {e: BIG_COST_NUMBER for e in self.E_EDGES_RAIL}  #NOK  -> MNOK
        self.Q_EDGE_RAIL = {e: 0 for e in self.E_EDGES_RAIL}   # TONNES ->MTONNES
        self.Q_EDGE_BASE_RAIL = {e: 100000 for e in self.E_EDGES_RAIL}   # TONNES
        self.LEAD_TIME_EDGE_RAIL = {e: 50 for e in self.E_EDGES_RAIL} #years

        self.C_NODE = {(i,c,m) : BIG_COST_NUMBER for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]}  # NOK
        self.Q_NODE_BASE = {(i,c,m): 100000 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]} #endret    # TONNES
        self.Q_NODE = {(i,c,m): 100000 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]} #lagt til 03.05    # TONNES
        self.LEAD_TIME_NODE = {(i,c,m): 50 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]} #years

        self.C_UPG = {(e,f) : BIG_COST_NUMBER for (e,f) in self.U_UPGRADE}  #NOK
        self.BIG_M_UPG = {e: [] for e in self.E_EDGES_UPG}        # TONNES
        self.LEAD_TIME_UPGRADE = {(e,f): 50 for (e,f) in self.U_UPGRADE} #years

        #how many times can you invest?
        #self.INV_NODE = {(i,m,b): 4 for (i,m,b) in self.NMB_CAP}
        #self.INV_LINK = {(l): 1 for l in self.E_EDGES_RAIL}

        for index, row in inv_rail_data.iterrows():
            ii = row["From"] 
            jj = row["To"] 
            mm = row["Mode"] 
            rr = row["Route"]   
            for ((i,j,m,r),f) in self.U_UPGRADE:
                #if f == "Electric train (CL)":
                if (ii,jj,mm,rr)==(i,j,m,r) or (jj,ii,mm,rr)==(i,j,m,r):
                    if (ii,jj,mm,rr)==(i,j,m,r):
                        e = (i,j,m,r)
                    elif (jj,ii,mm,rr)==(i,j,m,r):
                        e = (i,j,m,r)
                    self.C_UPG[(e,f)] = round(row['Elektrifisering (NOK)']/self.scaling_factor,2)
                    self.LEAD_TIME_UPGRADE[(e,f)] = row['Leadtime']
                    #TO DO: allow for partially electrified rail. Now we only take fully electrified. 

                #if i == row["From"] and j == row["To"] and m == row["Mode"] and r == row["Route"] and u == 'Partially electrified rail':
                #    self.C_INV_UPG[(l,u)] = row['Delelektrifisering (NOK)']/self.scaling_factor

        for m in self.M_MODES_CAP:
            for i in self.N_NODES_CAP_NORWAY[m]:
                for c in self.TERMINAL_TYPE[m]:
                    if m == "Rail" and c=="Combination":
                        cap_data = rail_cap_data.loc[(rail_cap_data['Fylke'] == i)]
                        self.Q_NODE_BASE[i,c,m] = cap_data.iloc[0]['Kapasitet combi 2014 (tonn)']/self.scaling_factor   #MTONNES
                        cap_exp_data = inv_rail_data.loc[(inv_rail_data['Fylke'] == i)]
                        self.Q_NODE[i,c,m] = cap_exp_data.iloc[0]['Økning i kapasitet (combi)']/self.scaling_factor   #MTONNES
                        self.C_NODE[i,c,m] = cap_exp_data.iloc[0]['Kostnad (combi)']/self.scaling_factor   #MNOK
                        self.LEAD_TIME_NODE[i,c,m] = cap_exp_data.iloc[0]['LeadtimeCombi']
                    if m == "Rail" and c=="Timber":
                        cap_data = rail_cap_data.loc[(rail_cap_data['Fylke'] == i)]
                        self.Q_NODE_BASE[i,c,m] = cap_data.iloc[0]['Kapasitet tømmer (tonn)']/self.scaling_factor   #MTONNES
                        cap_exp_data = inv_rail_data.loc[(inv_rail_data['Fylke'] == i)]
                        self.Q_NODE[i,c,m] = cap_exp_data.iloc[0]['Økning av kapasitet (tømmer)']/self.scaling_factor   #MTONNES
                        self.C_NODE[i,c,m] = cap_exp_data.iloc[0]['Kostnad (tømmer)']/self.scaling_factor  #MNOK
                        self.LEAD_TIME_NODE[i,c,m] = cap_exp_data.iloc[0]['LeadtimeTimber']
                    if m == "Sea":
                        cap_data = inv_sea_data.loc[(inv_sea_data['Fylke'] == i)]
                        self.Q_NODE_BASE[i,c,m] = cap_data.iloc[0]['Kapasitet']/self.scaling_factor  #MTONNES
                        self.Q_NODE[i,c,m] = cap_data.iloc[0]['Kapasitetsøkning']/self.scaling_factor  #MTONNES
                        self.C_NODE[i,c,m] = cap_data.iloc[0]['Kostnad']/self.scaling_factor  #MNOK
                        self.LEAD_TIME_NODE[i,c,m] = cap_data.iloc[0]['Ledetid']

        #this is bad programming -> To do: update
        for (i, j, m, r) in self.E_EDGES_RAIL:
            a1 = (i, j, m, r)
            a2 = (j, i, m, r)
            capacity_data1 = rail_cap_data.loc[(rail_cap_data['Fra'] == i) & (rail_cap_data['Til'] == j) & (rail_cap_data['Rute'] == r)]
            capacity_data2 = rail_cap_data.loc[(rail_cap_data['Fra'] == j) & (rail_cap_data['Til'] == i) & (rail_cap_data['Rute'] == r)]
            capacity_exp_data1 = inv_rail_data.loc[(inv_rail_data['Fra'] == i) & (inv_rail_data['Til'] == j) & (inv_rail_data['Rute'] == r)]
            capacity_exp_data2 = inv_rail_data.loc[(inv_rail_data['Fra'] == j) & (inv_rail_data['Til'] == i) & (inv_rail_data['Rute'] == r)]
            if len(capacity_data1) > 0:
                capacity_data = capacity_data1
            elif len(capacity_data2) > 0:
                capacity_data = capacity_data2
            if len(capacity_exp_data1) > 0:
                capacity_exp_data = capacity_exp_data1
            elif len(capacity_exp_data2) > 0:
                capacity_exp_data = capacity_exp_data2
            
            self.Q_EDGE_BASE_RAIL[a1] = capacity_data.iloc[0]['Maks kapasitet']/self.scaling_factor
            self.Q_EDGE_RAIL[a1] = capacity_exp_data.iloc[0]['Kapasitetsøkning']/self.scaling_factor
            self.C_EDGE_RAIL[a1] = round(capacity_exp_data.iloc[0]['Kostnad']/self.scaling_factor,2) #
            self.LEAD_TIME_EDGE_RAIL[a1] = capacity_exp_data.iloc[0]['Ledetid'] #

        for l in self.E_EDGES_UPG:
            self.BIG_M_UPG[l] =  self.Q_EDGE_BASE_RAIL[l] + self.Q_EDGE_RAIL[l]#*self.INV_LINK[l] 

        "Discount rate"
        self.risk_free_interest_rate = RISK_FREE_RATE # 2%
        self.D_DISCOUNT_RATE = 1 / (1 + self.risk_free_interest_rate)
        

        # --------------------------
        # --------------------------
        # CHARGING EDGES CONSTRAINT
        # --------------------------
        # --------------------------
        
        charging_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Invest road')
        
        self.CHARGING_TECH = []
        for index,row in charging_data.iterrows():
            #print((row["Mode"],row["Fuel"]))
            self.CHARGING_TECH.append((row["Mode"],row["Fuel"]))
        # all arcs (one per arc pair ij/ji) with mode Road and fuels Battery or Hydrogen
        self.EF_CHARGING = []
        for (i,j,m,r) in self.E_EDGES:
            e =(i,j,m,r)
            utlandet = ["Europa", "Sør-Sverige", "Nord-Sverige"]
            if i not in utlandet or j not in utlandet: #and or 'or'
                for (m, f) in self.CHARGING_TECH:
                    if e[2] == m:
                        self.EF_CHARGING.append((e,f))
        
        

        # base capacity on a pair of arcs (ij/ji - mfr), fix to 0 since no charging infrastructure exists now
        self.Q_CHARGE_BASE = {(e,f): 0 for (e,f) in self.EF_CHARGING}
        self.C_CHARGE = {(e,f): 9999 for (e,f) in self.EF_CHARGING}  # for p in self.P_PRODUCTS}    # TO DO, pick the right value
        self.LEAD_TIME_CHARGING = {(e,f): 50 for (e,f) in self.EF_CHARGING}
        max_truck_cap = MAX_TRUCK_CAP  # HARDCODE random average in tonnes, should be product based? or fuel based??
        for ((i, j, m, r),f) in self.EF_CHARGING:
            e = (i, j, m, r) 
            data_index = charging_data.loc[(charging_data['Mode'] == m) & (charging_data['Fuel'] == f)]
            self.C_CHARGE[(e,f)] = round((self.AVG_DISTANCE[e]
                                                   / data_index.iloc[0]["Max_station_dist"]
                                                   * data_index.iloc[0]["Station_cost"]
                                                   / (data_index.iloc[0][
                                                          "Trucks_filled_daily"] * max_truck_cap * 365)),1)  # 0.7 or not??? #MKR/MTONNES, so no dividing here
            self.LEAD_TIME_CHARGING[(e,f)] = data_index.iloc[0]["Ledetid"]

        #Technological readiness/maturity (with Bass diffusion model)
        self.tech_readiness_data = pd.read_excel(self.prefix+r'technological_maturity_readiness.xlsx',sheet_name="technological_readiness_bass") #new sheet with different readiness paths
        self.tech_is_mature = {} # indicates whether technology is already mature
        self.tech_base_bass_model = {} # contains all bass diffusion models for the technologies (only for non-mature technologies)
        self.tech_active_bass_model = {} # active bass model (only for non-mature technologies)
        self.tech_scen_p_q_variation = {} # variation (as %) for parameters p and q in the bass diffusion model in each of the scenarios
        self.tech_scen_t_0_delay = {} # delay for parameter in the bass diffusion model in each of the scenarios
        for index, row in self.tech_readiness_data.iterrows():
            # store whether technology is already mature or not 
            if row["Mature?"] == "yes":
                self.tech_is_mature[(row['Mode'], row['Fuel'])] = True
            else:
                self.tech_is_mature[(row['Mode'], row['Fuel'])] = False
                # if not mature, add bass diffusion model
                self.tech_base_bass_model[(row['Mode'], row['Fuel'])] = BassDiffusion(float(row["p"]), float(row["q"]), float(row["m"]), int(row["t_0"]))
                # set base bass model as active bass model
                #self.tech_active_bass_model[(row['Mode'] ,row['Fuel'])] = BassDiffusion(float(row["p"]), float(row["q"]), float(row["m"]), int(row["t_0"]))
                # store variations
                self.tech_scen_p_q_variation[(row['Mode'], row['Fuel'])] = row["p_q_variation"]
                self.tech_scen_t_0_delay[(row['Mode'], row['Fuel'])] = row["t_0_delay"]
        
        self.R_TECH_READINESS_MATURITY = {} # contains the active maturity path (number between 0 and 100)
        # initialize R_TECH_READINESS_MATURITY at base path
        for s in self.S_SCENARIOS:
            for (m,f) in self.tech_is_mature:
                if self.tech_is_mature[(m,f)]:
                    for year in self.T_TIME_PERIODS:    
                        self.R_TECH_READINESS_MATURITY[(m, f, year,s)] = 100 # assumption: all mature technologies have 100% market potential
                else:
                    for year in self.T_TIME_PERIODS:
                        #we can remove this one!
                        self.R_TECH_READINESS_MATURITY[(m, f, year,s)] = self.tech_base_bass_model[(m,f)].A(year) # compute maturity level based on base Bass diffusion model 
            

        #Initializing transport work share in base year
        self.init_transport_share = pd.read_excel(self.prefix+r'init_mode_fuel_mix.xlsx',sheet_name="InitMix")
        self.Q_SHARE_INIT_MAX = {}
        self.MFT_INIT_TRANSP_SHARE = []
        for index, row in self.init_transport_share.iterrows():
            (m,f,t) = (row['Mode'], row['Fuel'],row['Year'])
            self.Q_SHARE_INIT_MAX[(m,f,t)] = row['Max_transp_share']
            self.MFT_INIT_TRANSP_SHARE.append((m,f,t))

        #lifetime / lifespan
        self.lifespan_data = pd.read_excel(self.prefix+r'transport_costs_emissions_raw.xlsx', sheet_name='lifetimes')
        self.LIFETIME = {}
        for index, row in self.lifespan_data.iterrows():
            self.LIFETIME[(row['Mode'], row['Fuel'])] = row['Lifetime']

          
        #update R_TECH_READINESS_MATURITY based on scenario information
        for s in self.S_SCENARIOS:
            active_scenario_nr = self.scenario_information.scen_name_to_nr[s]
            for m in self.M_MODES:
                for f in self.FM_FUEL[m]:
                    if not self.tech_is_mature[(m,f)]: # only vary maturity information by scenario for non-mature technologies
                        cur_fg = self.scenario_information.mf_to_fg[(m,f)]
                        cur_path_name = self.scenario_information.fg_maturity_path_name[active_scenario_nr][cur_fg] # find name of current maturity path [base, fast, slow]
                        # extract info from current base Bass model
                        cur_base_bass_model = self.tech_base_bass_model[(m,f)] # current base Bass diffusion model
                        cur_base_p_q_variation = self.tech_scen_p_q_variation[(m,f)] # level of variation for this m,f 
                        cur_base_t_0_delay = self.tech_scen_t_0_delay[(m,f)] # time delay for t_0 for this m,f
                                    
                        # find current scenario's level of variation for q and p and delay for t_0 from base case
                        cur_scen_p_q_variation = 0.0 
                        cur_scen_t_0_delay = 0.0
                        if cur_path_name == "base":
                            cur_scen_p_q_variation = 0.0
                            cur_scen_t_0_delay = 0.0
                        if cur_path_name == "fast":
                            cur_scen_p_q_variation = cur_base_p_q_variation # increase p and q by cur_base_p_q_variation (e.g., 50%)
                            cur_scen_t_0_delay = - cur_base_t_0_delay # negative delay (faster development)
                        elif cur_path_name == "slow":
                            cur_scen_p_q_variation = - cur_base_p_q_variation # decrease p and q by cur_base_p_q_variation (e.g., 50%)
                            cur_scen_t_0_delay = cur_base_t_0_delay # positive delay (slower development)

                        # construct scenario bass model
                        cur_scen_bass_model = BassDiffusion(cur_base_bass_model.p * (1 + cur_scen_p_q_variation), # adjust p with cur_scen_variations
                                                            cur_base_bass_model.q * (1 + cur_scen_p_q_variation),     # adjust q with cur_scen_variations
                                                            cur_base_bass_model.m, 
                                                            cur_base_bass_model.t_0 + cur_scen_t_0_delay)
                        
                        # set as active bass model
                        self.tech_active_bass_model[(m,f,s)] = cur_scen_bass_model

                        # find start of second stage
                        for t in self.T_TIME_PERIODS:
                            if t not in self.T_TIME_FIRST_STAGE_BASE:
                                start_of_second_stage = t
                                break

                        # fill R_TECH_READINESS_MATURITY based on current scenario bass model
                        for t in self.T_TIME_PERIODS:
                            if t in self.T_TIME_FIRST_STAGE_BASE:
                                # first stage: follow base bass model
                                self.R_TECH_READINESS_MATURITY[(m,f,t,s)] = cur_base_bass_model.A(t)
                            else:
                                # second stage: use scenario bass model, with starting point A(2030) from base bass model
                                t_init = start_of_second_stage #initialize diffusion at start of second stage
                                A_init = cur_base_bass_model.A(t_init) # diffusion value at start of second stage 
                                self.R_TECH_READINESS_MATURITY[(m,f,t,s)] = cur_scen_bass_model.A_from_starting_point(t,A_init,t_init)



    def combined_sets(self):

        self.SS_SCENARIOS_NONANT = []
        for s in self.S_SCENARIOS:
            for ss in self.S_SCENARIOS:
                if (s != ss) and ((ss,s) not in self.SS_SCENARIOS_NONANT ):
                    self.SS_SCENARIOS_NONANT.append((s,ss))

        self.T_TIME_FIRST_STAGE = [t for t in self.T_TIME_FIRST_STAGE_BASE if t in self.T_TIME_PERIODS]  
        self.T_TIME_SECOND_STAGE = [t for t in self.T_TIME_SECOND_STAGE_BASE if t in self.T_TIME_PERIODS]  

        start = self.T_TIME_PERIODS[0]
        #if len(self.T_TIME_PERIODS) == len(self.T_TIME_PERIODS_ALL):
        end = self.T_TIME_PERIODS[len(self.T_TIME_PERIODS)-1] + 1   #we only need to model the development until the end
        #else:
        #    end = 
        self.T_YEARLY_TIME_PERIODS = [*range(start,end)] #all years from 2022 up to 2050
        self.T_YEARLY_TIME_PERIODS_ALL = [*range(start,self.T_TIME_PERIODS_ALL[len(self.T_TIME_PERIODS_ALL)-1] + 1)] #all years from 2022 up to 2050
        
        self.T_YEARLY_TIME_FIRST_STAGE = [ty for ty in self.T_YEARLY_TIME_PERIODS if ty < self.T_TIME_SECOND_STAGE_BASE[0] ]
        #self.T_YEARLY_TIME_FIRST_STAGE_NO_TODAY = [*range(self.T_TIME_PERIODS[0] + 1, 2030)] #first-stage years without the first period
        self.T_YEARLY_TIME_SECOND_STAGE = [ty for ty in self.T_YEARLY_TIME_PERIODS if ty >= self.T_TIME_SECOND_STAGE_BASE[0] ]
        
        self.Y_YEARS = {t:[] for t in self.T_TIME_PERIODS_ALL}
        t0 = self.T_TIME_PERIODS[0]
        num_periods = len(self.T_TIME_PERIODS_ALL)
        
        for i in range(num_periods):
            t = self.T_TIME_PERIODS_ALL[i]
            if i < num_periods-1:
                tp1 = self.T_TIME_PERIODS_ALL[i+1]
                self.Y_YEARS[t] = list(range(t-t0,tp1-t0))
            elif i == (num_periods - 1):  #this is the last time period. Lasts only a year?? 
                duration_previous = len(self.Y_YEARS[self.T_TIME_PERIODS_ALL[i-1]])
                self.Y_YEARS[t] = [self.T_TIME_PERIODS_ALL[i]-t0 + j for j in range(duration_previous)]

        self.T_MOST_RECENT_DECISION_PERIOD = {}
        for ty in self.T_YEARLY_TIME_PERIODS: #loop over all (yearly) years
            cur_most_recent_dec_period = self.T_TIME_PERIODS[0] #initialize at 2022
            for t in self.T_TIME_PERIODS: # loop over all decision periods
                if t <= ty:
                    cur_most_recent_dec_period = t 
            self.T_MOST_RECENT_DECISION_PERIOD[ty] = cur_most_recent_dec_period


        self.T_TIME_PERIODS_OPERATIONAL = self.T_TIME_PERIODS
        if self.last_time_period:
            self.T_TIME_PERIODS_OPERATIONAL = [self.T_TIME_PERIODS[-1]]

        #
        #       WITHOUT SCENARIOS
        #
        
        #------------------------
        "Combined sets - time independent"
        #------------------------

        self.NCM = [(i,c,m) for (i,m) in self.NM_LIST_CAP for c in self.TERMINAL_TYPE[m]]
        self.MF = [(m,f) for m in self.M_MODES for f in self.FM_FUEL[m]]

        "Combined sets - time dependent"

        self.TS = [(t,) for t in self.T_TIME_PERIODS]
        self.TS_CONSTR = [(t,) for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.TS_NO_BASE_YEAR = [(t,) for t in self.T_TIME_PERIODS if t is not self.T_TIME_PERIODS[0]]
        self.TS_NO_BASE_YEAR_CONSTR = [(t,) for t in self.T_TIME_PERIODS_OPERATIONAL if t is not self.T_TIME_PERIODS[0]]


        self.APT = [(i,j,m,r) + (p,) + (t,) for (i,j,m,r) in self.A_ARCS for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS] 
        self.AVT = [(i,j,m,r) + (v,) + (t,) for (i,j,m,r) in self.A_ARCS for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS] 
        self.APT_CONSTR = [(i,j,m,r) + (p,) + (t,) for (i,j,m,r) in self.A_ARCS for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS_OPERATIONAL] 
        self.AVT_CONSTR = [(i,j,m,r) + (v,) + (t,) for (i,j,m,r) in self.A_ARCS for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS_OPERATIONAL] 
        
        
        self.AFPT = [(i,j,m,r) + (f,) + (p,) + (t,)  for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] for p in self.P_PRODUCTS for t in
                         self.T_TIME_PERIODS ]
        self.AFVT = [(i,j,m,r) + (f,) + (v,) + (t,) for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] for v in self.VEHICLE_TYPES_M[m] for t in
                         self.T_TIME_PERIODS ]  
        self.KPT = [(k, p, t) for k in self.K_PATHS for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS]
        self.KVT = [(k, v, t) for k in self.K_PATHS for v in self.V_VEHICLE_TYPES for t in self.T_TIME_PERIODS]
        self.ET_RAIL= [l+(t,) for l in self.E_EDGES_RAIL for t in self.T_TIME_PERIODS]
        self.EAT_RAIL = [e+(a,)+(t,) for e in self.E_EDGES_RAIL for a in self.AE_ARCS[e] for t in self.T_TIME_PERIODS]
        self.EAT_RAIL_CONSTR = [e+(a,)+(t,) for e in self.E_EDGES_RAIL for a in self.AE_ARCS[e] for t in self.T_TIME_PERIODS_OPERATIONAL]        
        self.EFT_CHARGE = [(e,f,t) for (e,f) in self.EF_CHARGING for t in self.T_TIME_PERIODS]
        self.EFT_CHARGE_CONSTR = [(e,f,t) for (e,f) in self.EF_CHARGING for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.NCMT = [(i,c,m,t) for (i,c,m) in self.NCM for t in self.T_TIME_PERIODS]
        self.NCMT_CONSTR = [(i,c,m,t) for (i,c,m) in self.NCM for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.NMFVT = [(i,m,f,v,t) for m in self.M_MODES for f in self.FM_FUEL[m] for i in self.NM_NODES[m]
                                    for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS]
        self.NMFVT_CONSTR = [(i,m,f,v,t) for m in self.M_MODES for f in self.FM_FUEL[m] for i in self.NM_NODES[m]
                                    for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.ODPTS = [odp + (t,) for odp in self.ODP for t in self.T_TIME_PERIODS]
        self.ODPTS_CONSTR = [odp + (t,) for odp in self.ODP for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.EPT = [l + (p,) + (t,) for l in self.E_EDGES for p in self.P_PRODUCTS for t in
                         self.T_TIME_PERIODS]
        self.MFT_MATURITY = [mf + (t,) for mf in self.NEW_MF_LIST for t in self.T_TIME_PERIODS]
        self.MFT_MATURITY_CONSTR = [mf + (t,) for mf in self.NEW_MF_LIST for t in self.T_TIME_PERIODS_OPERATIONAL]
        self.MFT = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS]
        self.MFT_CONSTR = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS_OPERATIONAL]
        
        self.MFTT = [(m,f,t,tau) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS 
                            for tau in self.T_TIME_PERIODS if tau <= t]
        self.MFT_MIN0 = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] 
                                    for t in self.T_TIME_PERIODS if t!=self.T_TIME_PERIODS[0]]

        self.MT = [(m,t) for m in self.M_MODES for t in self.T_TIME_PERIODS]

        self.MFT_NEW = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS if not self.tech_is_mature[(m,f)]]
        self.MFT_NEW_YEARLY = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_YEARLY_TIME_PERIODS if not self.tech_is_mature[(m,f)]] #only new technologies (not mature yet)
        self.MFT_NEW_YEARLY_FIRST_STAGE_MIN0 = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_YEARLY_TIME_FIRST_STAGE if (not self.tech_is_mature[(m,f)] and t!=self.T_YEARLY_TIME_FIRST_STAGE[0])]
        self.MFT_NEW_YEARLY_SECOND_STAGE = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_YEARLY_TIME_SECOND_STAGE if not self.tech_is_mature[(m,f)]]
        self.MFT_NEW_FIRST_PERIOD = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in [self.T_TIME_PERIODS[0]] if not self.tech_is_mature[(m,f)]]

        self.UT_UPG = [(e,f,t) for (e,f) in self.U_UPGRADE for t in self.T_TIME_PERIODS]        
        self.UT_UPG_CONSTR = [(e,f,t) for (e,f) in self.U_UPGRADE for t in self.T_TIME_PERIODS_OPERATIONAL]  

        #
        #       WITH SCENARIOS
        #

        def combinations(list_of_tuples, list):
            list_of_tuples2 = []
            for tpl in list_of_tuples:
                for l in list:
                    tpl2 = tpl + (l,)
                    list_of_tuples2.append(tpl2) 
            return list_of_tuples2

        self.AFPT_S =          combinations(self.AFPT,self.S_SCENARIOS)
        self.APT_CONSTR_S =    combinations(self.APT_CONSTR,self.S_SCENARIOS)
        self.AFVT_S =          combinations(self.AFVT,self.S_SCENARIOS)
        self.AVT_CONSTR_S =    combinations(self.AVT_CONSTR,self.S_SCENARIOS)
        self.EAT_RAIL_CONSTR_S = combinations(self.EAT_RAIL_CONSTR,self.S_SCENARIOS)
        self.E_EDGES_RAIL_S = combinations(self.E_EDGES_RAIL,self.S_SCENARIOS)
        self.EFT_CHARGE_S = combinations(self.EFT_CHARGE,self.S_SCENARIOS)
        self.EFT_CHARGE_CONSTR_S = combinations(self.EFT_CHARGE_CONSTR,self.S_SCENARIOS)
        self.ET_RAIL_S = combinations(self.ET_RAIL,self.S_SCENARIOS)
        self.KPT_S = combinations(self.KPT,self.S_SCENARIOS)
        self.KVT_S = combinations(self.KVT,self.S_SCENARIOS)
        self.NCMT_S = combinations(self.NCMT,self.S_SCENARIOS)
        self.MFT_S = combinations(self.MFT,self.S_SCENARIOS)
        self.MFT_MATURITY_CONSTR_S = combinations(self.MFT_MATURITY_CONSTR,self.S_SCENARIOS)
        self.MFT_NEW_YEARLY_S = combinations(self.MFT_NEW_YEARLY,self.S_SCENARIOS)
        self.MFT_NEW_S = combinations(self.MFT_NEW,self.S_SCENARIOS)
        self.MFT_MIN0_S = combinations(self.MFT_MIN0,self.S_SCENARIOS)
        self.MFT_INIT_TRANSP_SHARE_S = combinations(self.MFT_INIT_TRANSP_SHARE,self.S_SCENARIOS)
        self.MFT_NEW_FIRST_PERIOD_S = combinations(self.MFT_NEW_FIRST_PERIOD,self.S_SCENARIOS)
        self.MFT_NEW_YEARLY_FIRST_STAGE_MIN0_S = combinations(self.MFT_NEW_YEARLY_FIRST_STAGE_MIN0,self.S_SCENARIOS)
        self.MFT_NEW_YEARLY_SECOND_STAGE_S = combinations(self.MFT_NEW_YEARLY_SECOND_STAGE,self.S_SCENARIOS)
        self.MF_S = combinations(self.MF,self.S_SCENARIOS)
        self.MT_S = combinations(self.MT,self.S_SCENARIOS)
        self.MFT_CONSTR_S = combinations(self.MFT_CONSTR,self.S_SCENARIOS)
        self.M_MODES_S = combinations([(m,) for m in self.M_MODES],self.S_SCENARIOS)
        self.NCM_S = combinations(self.NCM,self.S_SCENARIOS)
        self.NCMT_CONSTR_S = combinations(self.NCMT_CONSTR,self.S_SCENARIOS)
        self.NMFVT_CONSTR_S = combinations(self.NMFVT_CONSTR,self.S_SCENARIOS)
        self.ODPTS_CONSTR_S = combinations(self.ODPTS_CONSTR,self.S_SCENARIOS)
        self.TS_S = combinations(self.TS,self.S_SCENARIOS)
        self.TS_CONSTR_S = combinations(self.TS_CONSTR,self.S_SCENARIOS)
        self.TS_NO_BASE_YEAR_CONSTR_S = combinations(self.TS_NO_BASE_YEAR_CONSTR,self.S_SCENARIOS)
        self.T_TIME_PERIODS_S = combinations([(t,) for t in self.T_TIME_PERIODS],self.S_SCENARIOS)
        self.UT_UPG_S = combinations(self.UT_UPG,self.S_SCENARIOS)
        self.UT_UPG_CONSTR_S = combinations(self.UT_UPG_CONSTR,self.S_SCENARIOS)



    # TODO: FIX THIS FOR THE MATURITY PATHS
    def update_time_periods(self, time_periods):
        self.T_TIME_PERIODS = time_periods
        self.combined_sets()

        

    

print("Finished reading sets and classes.")



#Testing:
"""
base_data = TransportSets(sheet_name_scenarios='scenarios_base')
base_data.scenario_information

base_data.scenario_information.scenario_names

#base_data.update_scenario_dependent_parameters("LLH")
for i in base_data.R_TECH_READINESS_MATURITY:
    print(i, ": ", base_data.R_TECH_READINESS_MATURITY[i])

for cur_scen in base_data.scenario_information.scenario_names:
    base_data.update_scenario_dependent_parameters(cur_scen)
    #base_data.tech_active_bass_model[('Road', 'Hydrogen')]
    base_data.tech_active_bass_model[('Road', 'Hydrogen')].plot_A(show=False)

plt.show()




print("Finished testing")
"""

