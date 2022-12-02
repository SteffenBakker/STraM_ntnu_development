# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:31:04 2021

@author: ingvi
Adapted by Ruben
"""

"Example data"

import os
#os.chdir('M/Documents/GitHub/AIM_Norwegian_Freight_Model') #uncomment this for stand-alone testing of this fille
#os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')


from Data.settings import *
from collections import Counter
import pandas as pd
import itertools
import numpy as np

from math import cos, asin, sqrt, pi
import networkx as nx
from itertools import islice
pd.set_option("display.max_rows", None, "display.max_columns", None)

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

        #read and process scenario data
        scenario_data = pd.read_excel(self.prefix+self.scenario_file, sheet_name=sheet_name)
        self.num_scenarios = len(scenario_data)
        self.scenario_names = ["scen_" + str(i).zfill(len(str(self.num_scenarios))) for i in range(self.num_scenarios)] #initialize as scen_00, scen_01, scen_02, etc.
        self.probabilities = [1.0/self.num_scenarios] * self.num_scenarios #initialize as equal probabilities
                
        self.fg_cost_factor = [{}] * self.num_scenarios
        for index, row in scenario_data.iterrows():
            if "Name" in scenario_data:
                self.scenario_names[index] = row["Name"] #update scenario names if available
            if "Probability" in scenario_data:
                self.probabilities[index] = row["Probability"] #update probabilities if available
            for fg in self.fuel_groups:
                new_entry = {fg : row[fg]} #new entry for the dictionary fg_cost_factor[index]
                self.fg_cost_factor[index] = dict(self.fg_cost_factor[index], **new_entry) #add new entry to existing dict (trick from internet)

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
        self.init_data = init_data

        #read/construct data                
        self.construct_pyomo_data()
        if init_data:
            self.T_TIME_PERIODS = self.T_TIME_PERIODS_INIT
        self.combined_sets()

        #read/construct scenario information
        self.active_scenario_name = "benchmark" #no scenario has been activated; all data is from benchmark setting
        self.scenario_information = ScenarioInformation(self.prefix,sheet_name_scenarios) #TODO: check performance of this

    def construct_pyomo_data(self):

        self.pwc_aggr = pd.read_csv(self.prefix+r'demand.csv')
        self.city_coords = pd.read_csv(self.prefix+r'zonal_aggregation.csv', sep=';')


        print("Reading and constructing data")

        self.scaling_factor = SCALING_FACTOR #10E-5
        self.precision_digits = 6

        self.N_NODES = ["Oslo", "Bergen", "Trondheim", "Hamar", "Bodø", "Tromsø", "Kristiansand",
                        "Ålesund", "Stavanger", "Skien", "Sør-Sverige", "Nord-Sverige",
                        "Kontinentalsokkelen", "Europa", "Verden"]
        self.N_NODES_NORWAY = ["Oslo", "Bergen", "Trondheim", "Hamar", "Bodø", "Tromsø", "Kristiansand",
                        "Ålesund", "Stavanger", "Skien", "Kontinentalsokkelen"]
        
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
        
        self.M_MODES = ["Road", "Rail", "Sea"]

        self.NM_NODES = {m:None for m in self.M_MODES}
        self.NM_NODES["Road"] = self.ROAD_NODES
        self.NM_NODES["Sea"] = self.SEA_NODES
        self.NM_NODES["Rail"] = self.RAIL_NODES

        self.M_MODES_CAP = ["Rail", "Sea"]
        
        self.RAIL_NODES_NORWAY = self.N_NODES_NORWAY.copy()
        self.RAIL_NODES_NORWAY.remove("Kontinentalsokkelen")
        
        self.N_NODES_CAP_NORWAY = {"Rail": self.RAIL_NODES_NORWAY,
                            "Sea": self.SEA_NODES_NORWAY}

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
        
        self.T_TIME_PERIODS = [2020, 2025, 2030, 2040, 2050]
        self.T_TIME_PERIODS_NOT_NOW = self.T_TIME_PERIODS[1:]
        self.T_TIME_FIRST_STAGE = [2020,2025]
        self.T_MIN1 = {self.T_TIME_PERIODS[tt]:self.T_TIME_PERIODS[tt-1] for tt in range(1,len(self.T_TIME_PERIODS))}
        
        self.T_TIME_PERIODS_ALL = self.T_TIME_PERIODS
        self.T_TIME_PERIODS_INIT = [self.T_TIME_PERIODS[0]]
                
        self.Y_YEARS = {t:[] for t in self.T_TIME_PERIODS}
        t0 = self.T_TIME_PERIODS[0]
        num_periods = len(self.T_TIME_PERIODS)
        
        for i in range(num_periods):
            t = self.T_TIME_PERIODS[i]
            if i < num_periods-1:
                tp1 = self.T_TIME_PERIODS[i+1]
                self.Y_YEARS[t] = list(range(t-t0,tp1-t0))
            elif i == (num_periods - 1):  #this is the last time period. Lasts only a year?? 
                duration_previous = len(self.Y_YEARS[self.T_TIME_PERIODS[i-1]])
                self.Y_YEARS[t] = [self.T_TIME_PERIODS[i]-t0 + j for j in range(duration_previous)]

        self.P_PRODUCTS = ['Dry bulk', 'Fish', 'General cargo', 'Industrial goods', 'Other thermo',
                           'Timber', 'Wet bulk']
        
        self.TERMINAL_TYPE = {"Rail": ["Combination", "Timber"], "Sea": ["All"]}
        
        self.PT = {"Combination": ['Dry bulk', 'Fish', 'General cargo', 'Industrial goods', 'Other thermo','Wet bulk'],
                   "Timber": ['Timber'],
                   "All": self.P_PRODUCTS}

        self.E_EDGES = []
        self.E_EDGES_RAIL = []
        self.E_EDGES_UPG = []
        self.A_ARCS = []
        self.U_UPGRADE=[] #only one option, upgrade to electrify by means of electric train
        self.OD_PAIRS = []
        self.K_PATHS = []
      

        #self.UF_UPG = {"Battery train": ["Partially electrified rail", "Fully electrified rail"],
        #               "Electric train (CL)": ["Fully electrified rail"]}
        

        # Create initial edges              modal links
        for i in self.N_NODES:
            for j in self.N_NODES:
                if (j!=i):
                    for m in self.M_MODES:
                        if (i == "Hamar" and j == "Trondheim" and m == "Rail") or (
                                i == "Trondheim" and j == "Hamar" and m == "Rail"):
                            self.R_ROUTES = [1, 2]  # 1 = Dovrebanen, 2 = Rørosbanen
                        else:
                            self.R_ROUTES = [1]
                        for r in self.R_ROUTES:
                            edge = (i, j, m, r)
                            if ((j, i, m, r) not in self.E_EDGES):
                                self.E_EDGES.append(edge)

        # Defined allowed railway modal links
        self.allowed_rail = {"Oslo": ["Hamar", "Bergen", "Skien", "Sør-Sverige"],
                             "Bergen": ["Oslo"],
                             "Trondheim": ["Bodø", "Hamar", "Nord-Sverige"],
                             "Bodø": ["Trondheim", "Tromsø", "Nord-Sverige"],
                             "Tromsø": ["Bodø"],
                             "Hamar": ["Oslo", "Trondheim", "Ålesund","Sør-Sverige"],
                             "Kristiansand": ["Stavanger", "Skien"],
                             "Skien":["Kristiansand","Oslo"],
                             "Ålesund": ["Hamar"],
                             "Stavanger": ["Kristiansand"],
                             "Sør-Sverige": ["Oslo", "Hamar", "Nord-Sverige","Europa"],
                             "Nord-Sverige": ["Sør-Sverige", "Bodø", "Trondheim"],
                             "Europa": ["Sør-Sverige"]}

        self.allowed_road = {"Oslo": ["Hamar", "Bergen", "Skien", "Sør-Sverige"],
                             "Bergen": ["Oslo","Ålesund","Stavanger"],
                             "Trondheim": ["Bodø", "Hamar", "Nord-Sverige","Ålesund"],
                             "Bodø": ["Trondheim", "Tromsø", "Nord-Sverige"],
                             "Tromsø": ["Bodø"],
                             "Hamar": ["Oslo", "Trondheim", "Ålesund"],
                             "Kristiansand": ["Stavanger", "Skien"],
                             "Skien": ["Kristiansand", "Oslo"],
                             "Ålesund": ["Hamar","Bergen","Trondheim"],
                             "Stavanger": ["Kristiansand","Bergen"],
                             "Sør-Sverige": ["Oslo", "Nord-Sverige", "Europa"],
                             "Nord-Sverige": ["Sør-Sverige", "Bodø", "Trondheim"],
                             "Europa": ["Sør-Sverige"]}

        edges_for_deletion = []
        for e in self.E_EDGES:
            if e[2] == "Road":
                if e[0] not in self.allowed_road.keys() or e[1] not in self.allowed_road.keys():
                    edges_for_deletion.append(e)
                elif e[1] not in self.allowed_road[e[0]]:
                    edges_for_deletion.append(e)
            elif e[2] == "Sea":
                if (e[0] == "Hamar") or (e[1] == "Hamar"):
                    edges_for_deletion.append(e)
            elif e[2] == "Rail":
                if e[0] not in self.allowed_rail.keys() or e[1] not in self.allowed_rail.keys():
                    edges_for_deletion.append(e)
                elif e[1] not in self.allowed_rail[e[0]]:
                    edges_for_deletion.append(e)
        self.E_EDGES = set(self.E_EDGES) - set(edges_for_deletion)

        self.E_EDGES_NORWAY = []
        for (i,j,m,r) in self.E_EDGES:
            if i in self.N_NODES_NORWAY or j in self.N_NODES_NORWAY:
                self.E_EDGES_NORWAY.append((i,j,m,r))

        self.AE_ARCS = {e:[] for e in self.E_EDGES}
        self.AM_ARCS = {m:[] for m in self.M_MODES}
        # Create all arcs from allowed modal links THE F CAN BE REMOVED HERE!
        for (i, j, m, r) in self.E_EDGES:
            e = (i, j, m, r)
            a1 = (i, j, m, r)
            a2 = (j, i, m, r)
            self.A_ARCS.append(a1)
            self.A_ARCS.append(a2)
            self.AE_ARCS[e].append(a1)
            self.AE_ARCS[e].append(a2)
            self.AM_ARCS[m].append(a1)
            self.AM_ARCS[m].append(a2)
                
        
        for l in self.E_EDGES_NORWAY:
            if l[2] == "Rail":
                self.E_EDGES_RAIL.append(l)
        
        
        rail_cap_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Cap rail')
        inv_rail_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Invest rail')
        inv_sea_data = pd.read_excel(self.prefix+r'capacities_and_investments.xlsx', sheet_name='Invest sea')
     
        for index, row in inv_rail_data.iterrows():
            if pd.isnull(row["From"]):
                pass
            else:
                self.E_EDGES_UPG.append((row["From"],row["To"],row["Mode"],row["Route"]))   #this was L_LINKS_UPG
        
        for e in self.E_EDGES_UPG:
            self.U_UPGRADE.append((e,'Electric train (CL)')) #removed 'Battery electric' train as an option.
            

        ####################################
        ### ORIGIN, DESTINATION AND DEMAND #
        ####################################


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
        all_generated_paths = pd.read_csv(self.prefix+r'generated_paths.csv', converters={'paths': eval})
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

        

        # ---------------------------------------
        # -------CALCULATE LINK DISTANCES--------
        # ---------------------------------------

        sea_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Sea')
        road_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Road')
        rail_distance = pd.read_excel(self.prefix+r'distances.xlsx', sheet_name='Rail')

        road_distances_dict = {}
        sea_distances_dict = {}
        rail_distances_dict = {}
        for i in self.N_NODES:
            for j in self.N_NODES:
                for index, row in sea_distance.iterrows():
                    if row["Fra"] in [i, j] and row["Til"] in [i, j]:
                        sea_distances_dict[(i, j)] = row["Km - sjø"]
                for index, row in road_distance.iterrows():
                    if row["Fra"] in [i, j] and row["Til"] in [i, j]:
                        road_distances_dict[(i,j)] = row["Km - road"]
                for index, row in rail_distance.iterrows():
                    if row["Fra"] in [i, j] and row["Til"] in [i, j]:
                        rail_distances_dict[(i, j)] = row["Km - rail"]

        self.AVG_DISTANCE = {e: 0 for e in self.E_EDGES}
        for l in self.E_EDGES:
            if l[2] == "Road":
                if (l[0], l[1]) in road_distances_dict.keys():
                    self.AVG_DISTANCE[l] = road_distances_dict[(l[0], l[1])]
                    self.AVG_DISTANCE[(l[1], l[0], l[2], l[3])] = road_distances_dict[(l[1], l[0])]
            elif l[2] == "Sea":
                if (l[0], l[1]) in sea_distances_dict.keys():
                    self.AVG_DISTANCE[l] = sea_distances_dict[(l[0], l[1])]
                    self.AVG_DISTANCE[(l[1], l[0], l[2], l[3])] = sea_distances_dict[(l[1], l[0])]
            elif l[2] == "Rail":
                if (l[0], l[1]) in rail_distances_dict.keys():
                    self.AVG_DISTANCE[l] = rail_distances_dict[(l[0], l[1])]
                    self.AVG_DISTANCE[(l[1], l[0], l[2], l[3])] = rail_distances_dict[(l[1], l[0])]
            if l[0] not in self.N_NODES_NORWAY or l[1] not in self.N_NODES_NORWAY:
                self.AVG_DISTANCE[l] = self.AVG_DISTANCE[l] / 2        #We have to account for half the costs of international transport
                self.AVG_DISTANCE[(l[1], l[0], l[2], l[3])] = self.AVG_DISTANCE[(l[1], l[0], l[2], l[3])] / 2
            #if (l[0] == "Verden") and (l[1] == "Oslo") or (l[1] == "Verden") and (l[0] == "Oslo"):
            #    self.AVG_DISTANCE[l] = 0

        for key, value in self.AVG_DISTANCE.items():  
            self.AVG_DISTANCE[key] = round(value,1)

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
        
        self.CO2_CAP = dict(zip(self.emission_data['Year'], self.emission_data['Percentage']))
        #self.CO2_CAP = {year:round(cap/self.scaling_factor,0) for year,cap in self.CO2_CAP.items()}   #this was max 4*10^13, now 4*10^7
        
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
        self.C_TRANSP_COST = {(i,j,m,r, f, p, t): 1000000 for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] 
                              for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS}   #UNIT: NOK/T
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
                        #initially, set scenario-dependent cost equal to base cost (will be changed by separate function later)
                        self.C_TRANSP_COST[(i, j, m, r, f, p, y)] = self.C_TRANSP_COST_BASE[(i, j, m, r, f, p, y)] #(using same indices as above)
                        #^: MINIMUM 6.7, , median = 114.8, 90%quantile = 2562.9,  max 9.6*10^7!!!
                        self.E_EMISSIONS[(i, j, m, r, f, p, y)] = round(self.AVG_DISTANCE[a] * row['Emissions (gCO2/Tkm)'], 1)
                        #CO2 costs per tonne:
                        self.C_CO2[(i, j, m, r, f, p, y)] =  round(self.E_EMISSIONS[(i, j, m, r, f, p, y)] * self.CO2_fee[row["Year"]], 1)

        
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
                        self.cheapest_product_per_vehicle[(m,f,t,v)] = p

        #################
        #  INVESTMENTS  #
        #################
        
        self.C_EDGE_RAIL = {e: 100000000 for e in self.E_EDGES_RAIL}  #NOK  -> MNOK
        self.Q_EDGE_RAIL = {e: 0 for e in self.E_EDGES_RAIL}   # TONNES ->MTONNES
        self.Q_EDGE_BASE_RAIL = {e: 100000 for e in self.E_EDGES_RAIL}   # TONNES
        
        self.C_NODE = {(i,c,m) : 100000000 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]}  # NOK
        self.Q_NODE_BASE = {(i,c,m): 100000 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]} #endret    # TONNES
        self.Q_NODE = {(i,c,m): 100000 for m in self.M_MODES_CAP for i in self.N_NODES_CAP_NORWAY[m] for c in self.TERMINAL_TYPE[m]} #lagt til 03.05    # TONNES

        self.C_UPG = {(e,f) : 100000000 for (e,f) in self.U_UPGRADE}  #NOK
        self.BIG_M_UPG = {e: [] for e in self.E_EDGES_UPG}        # TONNES
        #how many times can you invest?
        #self.INV_NODE = {(i,m,b): 4 for (i,m,b) in self.NMB_CAP}
        #self.INV_LINK = {(l): 1 for l in self.E_EDGES_RAIL}
        

        for index, row in inv_rail_data.iterrows():
            for ((i,j,m,r),f) in self.U_UPGRADE:
                #for u in self.U_UPG:
                if i == row["From"] and j == row["To"] and m == row["Mode"] and r == row["Route"] and f == "Electric train (CL)": # u == 'Fully electrified rail':
                    self.C_UPG[((i,j,m,r),f)] = round(row['Elektrifisering (NOK)']/self.scaling_factor,2)
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
                    if m == "Rail" and c=="Timber":
                        cap_data = rail_cap_data.loc[(rail_cap_data['Fylke'] == i)]
                        self.Q_NODE_BASE[i,c,m] = cap_data.iloc[0]['Kapasitet tømmer (tonn)']/self.scaling_factor   #MTONNES
                        cap_exp_data = inv_rail_data.loc[(inv_rail_data['Fylke'] == i)]
                        self.Q_NODE[i,c,m] = cap_exp_data.iloc[0]['Økning av kapasitet (tømmer)']/self.scaling_factor   #MTONNES
                        self.C_NODE[i,c,m] = cap_exp_data.iloc[0]['Kostnad (tømmer)']/self.scaling_factor  #MNOK
                    if m == "Sea":
                        cap_data = inv_sea_data.loc[(inv_sea_data['Fylke'] == i)]
                        self.Q_NODE_BASE[i,c,m] = cap_data.iloc[0]['Kapasitet']/self.scaling_factor  #MTONNES
                        self.Q_NODE[i,c,m] = cap_data.iloc[0]['Kapasitetsøkning']/self.scaling_factor  #MTONNES
                        self.C_NODE[i,c,m] = cap_data.iloc[0]['Kostnad']/self.scaling_factor  #MNOK

        for (i, j, m, r) in self.E_EDGES_RAIL:
            capacity_data1 = rail_cap_data.loc[(rail_cap_data['Fra'] == i) & (rail_cap_data['Til'] == j) & (rail_cap_data['Rute'] == r)]
            capacity_data2 = rail_cap_data.loc[(rail_cap_data['Fra'] == j) & (rail_cap_data['Til'] == i) & (rail_cap_data['Rute'] == r)]
            capacity_exp_data1 = inv_rail_data.loc[(inv_rail_data['Fra'] == i) & (inv_rail_data['Til'] == j) & (inv_rail_data['Rute'] == r)]
            capacity_exp_data2 = inv_rail_data.loc[(inv_rail_data['Fra'] == j) & (inv_rail_data['Til'] == i) & (inv_rail_data['Rute'] == r)]
            if len(capacity_data1) > 0:
                self.Q_EDGE_BASE_RAIL[i, j, m, r] = capacity_data1.iloc[0]['Maks kapasitet']/self.scaling_factor
            if len(capacity_data2) > 0:
                self.Q_EDGE_BASE_RAIL[i, j, m, r] = capacity_data2.iloc[0]['Maks kapasitet']/self.scaling_factor
            if len(capacity_exp_data1) > 0:
                self.Q_EDGE_RAIL[i, j, m, r] = capacity_exp_data1.iloc[0]['Kapasitetsøkning']/self.scaling_factor
                self.C_EDGE_RAIL[i, j, m, r] = round(capacity_exp_data1.iloc[0]['Kostnad']/self.scaling_factor,1)
            if len(capacity_exp_data2) > 0:
                self.Q_EDGE_RAIL[i, j, m, r] = capacity_exp_data2.iloc[0]['Kapasitetsøkning']/self.scaling_factor
                self.C_EDGE_RAIL[i, j, m, r] = round(capacity_exp_data2.iloc[0]['Kostnad']/self.scaling_factor,1)
        
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
        # ALLE DISTANSER PÅ ROAD SOM TRENGER CHARGING INFRASTUCTURE
        self.CHARGE_ROAD_DISTANCE = {(e,f): road_distances_dict[(e[0], e[1])] for (e,f) in self.EF_CHARGING}
        # self.CHARGE_ROAD_DISTANCE = {mf:  for mf in self.CHARGING_TECH}
        self.C_CHARGE = {(e,f): 9999 for (e,f) in self.EF_CHARGING}  # for p in self.P_PRODUCTS}    # TO DO, pick the right value
        max_truck_cap = MAX_TRUCK_CAP  # HARDCODE random average in tonnes, should be product based? or fuel based??
        for ((i, j, m, r),f) in self.EF_CHARGING:
            e = (i, j, m, r) 
            data_index = charging_data.loc[(charging_data['Mode'] == m) & (charging_data['Fuel'] == f)]
            self.C_CHARGE[(e,f)] = round((self.CHARGE_ROAD_DISTANCE[(e,f)]
                                                   / data_index.iloc[0]["Max_station_dist"]
                                                   * data_index.iloc[0]["Station_cost"]
                                                   / (data_index.iloc[0][
                                                          "Trucks_filled_daily"] * max_truck_cap * 365)),1)  # 0.7 or not???
            #MKR/MTONNES, so no dividing here


        #Technological readiness/maturity
        self.tech_readiness = pd.read_excel(self.prefix+r'technological_maturity_readiness.xlsx',sheet_name="technological_readiness")
        self.R_TECH_READINESS_MATURITY = {}
        for index, row in self.tech_readiness.iterrows():
            for year in self.T_TIME_PERIODS:
                self.R_TECH_READINESS_MATURITY[(row['Mode'], row['Fuel'],year)] = row[str(year)]

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

    def combined_sets(self):

        


        #------------------------
        "Combined sets - time independent"
        #------------------------

        self.NCM = [(i,c,m) for (i,m) in self.NM_LIST_CAP for c in self.TERMINAL_TYPE[m]]
        self.MF = [(m,f) for m in self.M_MODES for f in self.FM_FUEL[m]]

        "Combined sets - time dependent"

        self.TS = [(t) for t in self.T_TIME_PERIODS]
        self.APT = [(i,j,m,r) + (p,) + (t,) for (i,j,m,r) in self.A_ARCS for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS] 
        self.AVT = [(i,j,m,r) + (v,) + (t,) for (i,j,m,r) in self.A_ARCS for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS] 
        self.AFPT = [(i,j,m,r) + (f,) + (p,) + (t,) for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] for p in self.P_PRODUCTS for t in
                         self.T_TIME_PERIODS]
        self.AFVT = [(i,j,m,r) + (f,) + (v,) + (t,) for (i,j,m,r) in self.A_ARCS for f in self.FM_FUEL[m] for v in self.VEHICLE_TYPES_M[m] for t in
                         self.T_TIME_PERIODS]  
        self.KPT = [(k, p, t) for k in self.K_PATHS for p in self.P_PRODUCTS for t in self.T_TIME_PERIODS]
        self.KVT = [(k, v, t) for k in self.K_PATHS for v in self.V_VEHICLE_TYPES for t in self.T_TIME_PERIODS]
        self.ET_RAIL= [l+(t,) for l in self.E_EDGES_RAIL for t in self.T_TIME_PERIODS]
        self.EAT_RAIL = [e+(a,)+(t,) for e in self.E_EDGES_RAIL for a in self.AE_ARCS[e] for t in self.T_TIME_PERIODS]
        self.EFT_CHARGE = [(e,f,t) for (e,f) in self.EF_CHARGING for t in self.T_TIME_PERIODS]
        self.NCMT = [(i,c,m,t) for (i,c,m) in self.NCM for t in self.T_TIME_PERIODS]
        self.NMFVT = [(i,m,f,v,t) for m in self.M_MODES for f in self.FM_FUEL[m] for i in self.NM_NODES[m]
                                    for v in self.VEHICLE_TYPES_M[m] for t in self.T_TIME_PERIODS]
        self.ODPTS = [odp + (t,) for odp in self.ODP for t in self.T_TIME_PERIODS]
        self.EPT = [l + (p,) + (t,) for l in self.E_EDGES for p in self.P_PRODUCTS for t in
                         self.T_TIME_PERIODS]
        self.MFT_MATURITY = [mf + (t,) for mf in self.NEW_MF_LIST for t in self.T_TIME_PERIODS]
        self.MFT = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS]
        
        self.MFTT = [(m,f,t,tau) for m in self.M_MODES for f in self.FM_FUEL[m] for t in self.T_TIME_PERIODS 
                            for tau in self.T_TIME_PERIODS if tau <= t]
        self.MFT_MIN0 = [(m,f,t) for m in self.M_MODES for f in self.FM_FUEL[m] 
                                    for t in self.T_TIME_PERIODS if t!=self.T_TIME_PERIODS[0]]
        self.UT_UPG = [(e,f,t) for (e,f) in self.U_UPGRADE for t in self.T_TIME_PERIODS]        

    def update_time_periods(self, init_data):
        if init_data==False:
            self.T_TIME_PERIODS = self.T_TIME_PERIODS_ALL
        else:
            self.T_TIME_PERIODS = self.T_TIME_PERIODS_INIT
        self.combined_sets()

    #Function that updates all information that depends on the current scenario number
    #Currently: update transport costs based on fuel group scenarios
    def update_scenario_dependent_parameters(self,scenario_name):

        #set active scenario
        self.active_scenario_name = scenario_name       
        
        if self.active_scenario_name in self.scenario_information.scenario_names: #we are in an exising scenario           
            #Find associated active scenario number (only store temporarily)
            active_scenario_nr = self.scenario_information.scen_name_to_nr[self.active_scenario_name]
            #update C_TRANSP_COST based on scenario information
            for (i, j, m, r) in self.A_ARCS:
                for f in self.FM_FUEL[m]:
                    for p in self.P_PRODUCTS:
                        for y in self.T_TIME_PERIODS:
                            #transport cost = base transport cost * cost factor for fuel group associated with (m,f) for current active scenario:
                            self.C_TRANSP_COST[(i, j, m, r, f, p, y)] = self.C_TRANSP_COST_BASE[(i, j, m, r, f, p, y)] * self.scenario_information.mode_fuel_cost_factor[active_scenario_nr][(m,f)] 

        else: #we should be in the benchmark scenario
            if self.active_scenario_name == "benchmark":
                #set C_TRANSP_COST to benchmark levels
                self.C_TRANSP_COST = self.C_TRANSP_COST_BASE
            else:
                raise Exception("Active scenario name is not in scenario list, but also not equal to benchmark")


                        


#Testing:
#base_data = TransportSets()
#base_data.scenario_information