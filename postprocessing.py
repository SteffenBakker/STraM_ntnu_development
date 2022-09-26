# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:33:33 2022

@author: steffejb
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import mpisppy.utils.sputils as sputils


###############################################################
#               postprocessing: PLOTTING AND DATA EXTRACTION                  # 
###############################################################

#data is base data, dataset is output from model
def plot_figures(data,dataset,scenarios,instance_run,solution_method):

    N_NODES_NO_SEA = ["Oslo", "Bergen", "Trondheim", "Hamar", "Bodø", "Tromsø", "Kristiansand",
                            "Ålesund", "Stavanger", "Skien", "Sør-Sverige", "Nord-Sverige","Europa"]
    # to do. Where is this used? REMOVE?
    
    for s in scenarios:
        if solution_method == "ef":
            s = s[0]
        dataset_scen = dataset[dataset["scenario"]==s]
        fuel_list = []
        fuel_list1 = []
        for m in ["Road","Rail","Sea"]:
            for f in dataset_scen[dataset_scen["Mode"] == m]["fuel"].unique():
                for t in data.T_TIME_PERIODS:
                    dataset_scen_dom = dataset_scen[(dataset_scen["from"].isin(data.N_NODES_NORWAY)) & (dataset_scen["to"].isin(data.N_NODES_NORWAY))]
                    dataset_scen_int = dataset_scen
                    yearly_weight_int = dataset_scen_int[dataset_scen_int["time_period"] == t]["weight"].sum()
                    dataset_temp_int = dataset_scen_int[(dataset_scen_int["Mode"] == m) & (dataset_scen_int["fuel"] == f) & (dataset_scen_int["time_period"] == t)]

                    yearly_weight_dom = dataset_scen_dom[dataset_scen_dom["time_period"] == t]["weight"].sum()
                    dataset_temp_dom = dataset_scen_dom[
                        (dataset_scen_dom["Mode"] == m) & (dataset_scen_dom["fuel"] == f) & (
                                    dataset_scen_dom["time_period"] == t)]
                    if len(dataset_temp_int) > 0:
                        fuel_list.append((m,f,t,dataset_temp_int["weight"].sum()*100/yearly_weight_int))
                    else:
                        fuel_list.append((m,f,t,0))
                    if len(dataset_temp_dom) > 0:
                        fuel_list1.append((m,f,t,dataset_temp_dom["weight"].sum()*100/yearly_weight_dom))
                    else:
                        fuel_list1.append((m,f,t,0))

        for plot_nr in range(2):
            if plot_nr == 0:
                fuel_list = fuel_list
            if plot_nr == 1:
                fuel_list = fuel_list1
            fuel_list_road = []
            fuel_list_rail = []
            fuel_list_sea = []

            for e in fuel_list:
                if e[0] == "Road":
                    fuel_list_road.append(e)
                elif e[0] == "Rail":
                    fuel_list_rail.append(e)
                elif e[0] == "Sea":
                    fuel_list_sea.append(e)

            color_sea = iter(cm.Blues(np.linspace(0.3,1,7)))
            color_road = iter(cm.Reds(np.linspace(0.4,1,5)))
            color_rail = iter(cm.Greens(np.linspace(0.25,1,5)))

            labels = ['2022', '2025', '2030', '2040', '2050']
            width = 0.35       # the width of the bars: can also be len(x) sequence
            bottom = np.array([0,0,0,0,0])

            FM_FUEL = data.FM_FUEL

            color_dict = {}
            for m in ["Road", "Rail", "Sea"]:
                for f in FM_FUEL[m]:
                    if m == "Road":
                        color_dict[m,f] = next(color_road)
                    elif m == "Rail":
                        color_dict[m, f] = next(color_rail)
                    elif m == "Sea":
                        color_dict[m, f] = next(color_sea)

            fig, ax = plt.subplots()
            for i in range(0,len(fuel_list),5):
                chunk = fuel_list[i:i + 5]
                fuel_flow = []
                for elem in chunk:
                    fuel_flow.append(elem[3])
                if sum(fuel_flow) > 0.0001:
                    ax.bar(labels, fuel_flow, width, bottom=bottom,
                        label=str(chunk[0][0])+" "+str(chunk[0][1]), color=color_dict[chunk[0][0],chunk[0][1]])
                    bottom = np.add(bottom,np.array(fuel_flow))

            if plot_nr == 0:
                ax.set_title("Instance "+instance_run+' Scen '+str(s)+" All")
            elif plot_nr == 1:
                ax.set_title("Instance "+instance_run+' Scen '+str(s)+" Domestic")
            box = ax.get_position()
            #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height*0.9]) #legends!!!
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height]) #correct
            plt.xticks(fontsize=16)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct
            #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
            #     ncol=3, fancybox=True, shadow=True)  #LEGENDS!
            if plot_nr == 0:
                plt.savefig("Data/Instance_results_write_to_here/Instance"+instance_run+"/Instance"+instance_run+'Scen' + str(s)+'_international.png')
            elif plot_nr == 1:
                plt.savefig("Data/Instance_results_write_to_here/Instance"+instance_run+"/Instance"+instance_run+'Scen' + str(s)+'_domestic.png')
            plt.show()

def extract_output_ph(ph,data,instance_run):
    dataset = pd.DataFrame(columns = ['from','to','Mode',"route","fuel",'product','weight','time_period', 'scenario'])
    for e in ph.local_subproblems:
        modell = ph.local_subproblems[e]
        for (i,j,m,r) in data.A_ARCS:
            a = (i,j,m,r)
            for f in data.FM_FUEL[m]:
                for t in data.T_TIME_PERIODS:
                    for p in data.P_PRODUCTS:
                        weight = modell.x_flow[(a,f,p,t)].value*data.AVG_DISTANCE[a]
                        if weight > 1:
                            a_series = pd.Series([i,j,m,r,f,p,weight, t, e], index=dataset.columns)
                            #a_series = pd.Series([l[0], l[1], l[2], l[3], weight, t, e], index=dataset.columns)
                            dataset = dataset.append(a_series, ignore_index=True)
                        
    return dataset

print('test')
#data = base_data
def extract_aggregated_output_ef(ef,data,instance_run):
    dataset_x_flow = pd.DataFrame(columns = ['from','to','Mode',"route","fuel",'product','weight','time_period', 'scenario'])
    dataset_h_paths = pd.DataFrame(columns = ['path','product','weight','time_period', 'scenario'])
    dataset_charging = pd.DataFrame(columns = ['from','to','Mode',"route","fuel",'time_period','weight','scenario'])
    dataset_w_node = pd.DataFrame(columns=['Node', "terminal_type",'Mode', 'time_period', 'weight', 'scenario'])
    dataset_v_edge = pd.DataFrame(columns=['from','to','Mode',"route",'time_period','weight','scenario'])
    dataset_u_upgrade = pd.DataFrame(columns=['from', 'to', 'Mode', "route","fuel", 'time_period', 'weight', 'scenario'])
    
    dataset_violation = pd.DataFrame(columns = ['time_period','weight','scenario'])
    dataset_emissions = pd.DataFrame(columns = ['time_period','emissions','scenario'])
    
    
    for scen in sputils.ef_scenarios(ef):
        modell = scen[1]
        for (i,j,m,r) in data.A_ARCS:
            a = (i,j,m,r)
            for f in data.FM_FUEL[m]:
                for t in data.T_TIME_PERIODS:
                    for p in data.P_PRODUCTS:
                        weight = modell.x_flow[(a,f,p,t)].value*data.AVG_DISTANCE[a]
                        if weight > 0:
                            a_series = pd.Series([i,j,m,r,f,p,weight, t, scen[0]], index=dataset_x_flow.columns)
                            dataset_x_flow = dataset_x_flow.append(a_series, ignore_index=True)
        for kk in data.K_PATHS:
            #k = self.K_PATH_DICT[kk]
            for t in data.T_TIME_PERIODS:
                for p in data.P_PRODUCTS:
                    weight = modell.h_flow[(kk, p, t)].value
                    if weight > 0:
                        a_series = pd.Series([kk, p, t, weight, scen[0]], index=dataset_h_paths.columns)
                        dataset_h_paths = dataset_h_paths.append(a_series, ignore_index=True)

        for t in data.T_TIME_PERIODS:
            for i,j,m,r in data.E_EDGES_RAIL:
                e = (i,j,m,r)
                weight = modell.v_edge[(e, t)].value
                if weight > 0:
                    a_series = pd.Series([i,j,m,r, t, weight, scen[0]], index=dataset_v_edge.columns)
                    dataset_v_edge = dataset_v_edge.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (e,f) in data.U_UPGRADE:
                (i,j,m,r) = e
                weight1 = modell.u_upg[(i,j,m,r,f,t)].value
                if weight1 > 0:
                    a_series = pd.Series([i,j,m,r, f,t, weight1, scen[0]],
                                         index=dataset_u_upgrade.columns)
                    dataset_u_upgrade = dataset_u_upgrade.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (i, m) in data.NM_LIST_CAP:
                for c in data.TERMINAL_TYPE[m]:
                    weight2 = modell.w_node[(i, c, m, t)].value
                    if weight2 > 0:
                        a_series = pd.Series([i, c, m, t, weight2, scen[0]],
                                             index=dataset_w_node.columns)
                        dataset_w_node = dataset_w_node.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (e,f) in data.EF_CHARGING:
                (i,j,m,r) = e
                weight3 = modell.y_charge[(i,j,m,r,f,t)].value
                if weight3 > 0:
                    a_series = pd.Series([i,j,m,r,f,t, weight3, scen[0]],
                                         index=dataset_charging.columns)
                    dataset_charging = dataset_charging.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            weight4 = modell.z_emission[t].value
            weight5 = modell.total_emissions[t].value
            a_series = pd.Series([t, weight4, e[0]],index=dataset_violation.columns)
            dataset_violation = dataset_violation.append(a_series, ignore_index=True)
            a_series2 = pd.Series([t, weight5, e[0]],index=dataset_emissions.columns)
            dataset_emissions = dataset_emissions.append(a_series2, ignore_index=True)
            #print('--------- Total emissions -----------')
            #print(e[0],t, "Total emissions: ",modell.total_emissions[t].value,", emission violation: ",
            #    modell.z_emission[t].value,", violation/emission_cap: ", 1-(modell.total_emissions[t].value/(data.CO2_CAP[2020])))
    #print("Number of variables: ",modell.nvariables())
    #print("Number of constraints: ",modell.nconstraints())
   
    return dataset_x_flow, dataset_charging, dataset_w_node, \
            dataset_v_edge, dataset_u_upgrade, dataset_violation, dataset_emissions

def extract_output_ef(ef,data,instance_run):
    dataset = pd.DataFrame(columns = ['from','to','Mode',"route","fuel",'product','weight','time_period', 'scenario'])
    
    dataset_violation = pd.DataFrame(columns = ['time_period','weight','scenario'])
    dataset_charging = pd.DataFrame(columns = ['from','to','Mode',"route","fuel",'time_period','weight','scenario'])
    dataset_w_node = pd.DataFrame(columns=['Node', "terminal_type",'Mode', 'time_period', 'weight', 'scenario'])
    dataset_v_edge = pd.DataFrame(columns=['from','to','Mode',"route",'time_period','weight','scenario'])
    dataset_u_upgrade = pd.DataFrame(columns=['from', 'to', 'Mode', "route","fuel", 'time_period', 'weight', 'scenario'])
    dataset_paths = pd.DataFrame(columns=['path',"product","time_period","weight","scenario"])
    
    
    for e in sputils.ef_scenarios(ef):
        modell = e[1]
        for (i,j,m,r) in data.A_ARCS:
            a = (i,j,m,r)
            for f in data.FM_FUEL[m]:
                for t in data.T_TIME_PERIODS:
                    for p in data.P_PRODUCTS:
                        weight = modell.x_flow[(a,f,p,t)].value*data.AVG_DISTANCE[a]
                        if weight > 0:
                            a_series = pd.Series([i,j,m,r,f,p,weight, t, e[0]], index=dataset.columns)
                            dataset = dataset.append(a_series, ignore_index=True)

        print("NEXT SCENARIO: ", e[0])

        for kk in data.K_PATHS:
            #k = self.K_PATH_DICT[kk]
            for t in data.T_TIME_PERIODS:
                for p in data.P_PRODUCTS:
                    weight = modell.h_flow[(kk, p, t)].value
                    if weight > 0:
                        a_series = pd.Series([kk, p, t, weight, e[0]], index=dataset_paths.columns)
                        dataset_paths = dataset_paths.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for i,j,m,r in data.E_EDGES_RAIL:
                e = (i,j,m,r)
                weight = modell.v_edge[(e, t)].value
                if weight > 0:
                    a_series = pd.Series([i,j,m,r, t, weight, e[0]], index=dataset_v_edge.columns)
                    dataset_v_edge = dataset_v_edge.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (e,f) in data.U_UPGRADE:
                (i,j,m,r) = e
                weight1 = modell.z_inv_upg[(i,j,m,r,f,t)].value
                if weight1 > 0:
                    a_series = pd.Series([i,j,m,r, f,t, weight1, e[0]],
                                         index=dataset_u_upgrade.columns)
                    dataset_u_upgrade = dataset_u_upgrade.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (i, m) in data.NM_LIST_CAP:
                for c in data.TERMINAL_TYPE[m]:
                    weight2 = modell.w_node[(i, c, m, t)].value
                    if weight2 > 0:
                        a_series = pd.Series([i, c, m, t, weight2, e[0]],
                                             index=dataset_w_node.columns)
                        dataset_w_node = dataset_w_node.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (e,f) in data.EF_CHARGING:
                (i,j,m,r) = e
                weight3 = modell.y_charge[(i,j,m,r,f,t)].value
                if weight3 > 0:
                    a_series = pd.Series([i,j,m,r,f,t, weight3, e[0]],
                                         index=dataset_charging.columns)
                    dataset_charging = dataset_charging.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            weight4 = modell.z_emission[t].value
            if weight4 > 0:
                a_series = pd.Series([t, weight4, e[0]],
                                     index=dataset_violation.columns)
                dataset_violation = dataset_violation.append(a_series, ignore_index=True)
        print('--------- Total emissions -----------')
        for t in data.T_TIME_PERIODS:
            print(e[0],t, "Total emissions: ",modell.total_emissions[t].value,", emission violation: ",
                modell.z_emission[t].value,", violation/emission_cap: ", 1-(modell.total_emissions[t].value/(data.CO2_CAP[2020]/data.factor)))
    #print("Number of variables: ",modell.nvariables())
    #print("Number of constraints: ",modell.nconstraints())
    dataset.to_csv("Data/Instance_results_write_to_here/Instance" +instance_run+ "/Inst_" +instance_run+ "_x_flow.csv")
    dataset_paths.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_h_flow.csv")
    dataset_v_edge.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_v_edge.csv")
    dataset_u_upgrade.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_u_upg.csv")
    dataset_w_node.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_w_node.csv")
    dataset_charging.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_y_charge.csv")
    dataset_violation.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_z_emission.csv")
    
    return dataset
    