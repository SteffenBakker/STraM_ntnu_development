#Remember to set the right workingdirectory. Otherwise errors with loading the classes
import os
#os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\AIM_Norwegian_Freight_Model\\AIM_Norwegian_Freight_Model')


import pandas as pd
from TranspModelClass import *
from Data.Create_Sets_Class import TransportSets
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.opt.ph import PH
import mpisppy.scenario_tree as scenario_tree
import time
import sys

start = time.time()

#################################################
#                   user input                  #
#################################################

instance = '2'     #change instance_run to choose which instance you want to run
distribution_on_cluster = False  #is the code to be run on the cluster using the distribution package?
plotting = True

#READ INSTANCES FROM EXCEL OR MANUAL?

#################################################
#                   main code                   #
#################################################


if __name__ == "__main__":

    
    if distribution_on_cluster:
        instance_run = sys.argv[1]
    else:
        instance_run = instance # or sys.argv[1] if master.bash is run
    
    # READ DATA
    
    instance_data = pd.read_excel(r'Data/instances.xlsx', sheet_name='Sheet1')
    instances = [str(i) for i in list(instance_data['instance'])]
    instance_dict = {instances[i]:instance_data.set_index('instance').to_dict('records')[i] for i in range(len(instances))}
    
    print("Running and logging instance ", instance_run)
    print(instance_dict[instance_run])
    
    solution_method = instance_dict[instance_run]['sol_met']
    scenario_structure = instance_dict[instance_run]['scen_struct']
    CO2_price = instance_dict[instance_run]['co2_price']
    fuel_cost = instance_dict[instance_run]['costs']
    probabilities = instance_dict[instance_run]['probs']
    emission_reduction = instance_dict[instance_run]['emission_reduction']
    
    data = TransportSets('HHH',CO2_price,fuel_cost, emission_reduction) #needs to be initialized with some scenario I guess.
    options = option_settings()
    all_scenario_names = get_all_scenario_names(scenario_structure)    

    if solution_method == "ef":
        solver = pyo.SolverFactory(options["solvername"])

        scenario_creator_kwargs = {'CO2_price':CO2_price, 'fuel_cost':fuel_cost, 'emission_reduction':emission_reduction}
        ef = sputils.create_EF(
                all_scenario_names,
                scenario_creator,
                scenario_creator_kwargs = scenario_creator_kwargs
            )

        results = solver.solve(ef, logfile= r'Data/Instance_results_write_to_here/Instance'+instance_run+'/logfile'+instance_run+'.log', tee= True)
        print('EF objective value:', pyo.value(ef.EF_Obj))
        stop = time.time()
        print("The time of the run:", stop - start)

        dataset = extract_output_ef(ef,data)
        
        scenarios = sputils.ef_scenarios(ef)

    if solution_method == "ph":

        ############### HUB STUFF
        options["xhat_specific_options"] = {"xhat_solver_options":
                                              options["iterk_solver_options"],
                                              "xhat_scenario_dict": \
                                              {"ROOT": "Scen1",
                                               "ROOT_0": "Scen1",
                                               "ROOT_1": "Scen4",
                                               "ROOT_2": "Scen7"},
                                              "csvname": "specific.csv"}
        
        scenario_creator_kwargs = {'CO2_price':CO2_price, 'fuel_cost':fuel_cost, 'emission_reduction':emission_reduction}
        ph = PH(
            options,
            all_scenario_names,
            scenario_creator, scenario_denouement,
            scenario_creator_kwargs = scenario_creator_kwargs
            )
        ph.ph_main()
        print(ph)

        dataset = extract_output_ph(ph,data)
        
        scenarios = ph.local_subproblems

    if plotting == True:
        plotting(scenarios,instance_run)


###############################################################
#               postprocessing: PLOTTING AND DATA EXTRACTION                  # 
###############################################################


def plotting(scenarios,instance_run):

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

def extract_output_ph(ph,data):
    dataset = pd.DataFrame(columns = ['from','to','Mode',"fuel","route",'product','weight','time_period', 'scenario'])
    for e in ph.local_subproblems:
        modell = ph.local_subproblems[e]
        for a in data.A_ARCS:
            for t in data.T_TIME_PERIODS:
                for p in data.P_PRODUCTS:
                    weight = modell.x_flow[(a[0], a[1], a[2], a[3], a[4], p, t)].value*data.AVG_DISTANCE[(a[0], a[1], a[2], a[4])]
                    if weight > 1:
                        a_series = pd.Series([a[0], a[1], a[2], a[3], a[4], p, weight, t, e], index=dataset.columns)
                        #a_series = pd.Series([l[0], l[1], l[2], l[3], weight, t, e], index=dataset.columns)
                        dataset = dataset.append(a_series, ignore_index=True)
                        
    return dataset
            
def extract_output_ef(ef,data,instance_run):
    
    dataset = pd.DataFrame(columns = ['from','to','Mode',"fuel","route",'product','weight','time_period', 'scenario'])
    
    dataset_violation = pd.DataFrame(columns = ['time_period','weight','scenario'])
    dataset_charging = pd.DataFrame(columns = ['from','to','Mode',"fuel","route",'time_period','weight','scenario'])
    dataset_z_inv_node = pd.DataFrame(columns=['Node', 'Mode', "terminal_type", 'time_period', 'weight', 'scenario'])
    dataset_z_inv_cap = pd.DataFrame(columns=['from','to','Mode',"route",'time_period','weight','scenario'])
    dataset_z_inv_upg = pd.DataFrame(columns=['from', 'to', 'Mode', "route","upgrade", 'time_period', 'weight', 'scenario'])
    dataset_paths = pd.DataFrame(columns=['path',"product","time_period","weight","scenario"])
    
    
    for e in sputils.ef_scenarios(ef):
        modell = e[1]
        for a in data.A_ARCS:
            for t in data.T_TIME_PERIODS:
                for p in data.P_PRODUCTS:
                    weight = modell.x_flow[(a[0], a[1], a[2], a[3], a[4], p, t)].value*data.AVG_DISTANCE[(a[0], a[1], a[2], a[4])]
                    if weight > 0:
                        a_series = pd.Series([a[0], a[1], a[2], a[3], a[4], p, weight, t, e[0]], index=dataset.columns)
                        dataset = dataset.append(a_series, ignore_index=True)

        print("NEXT SCENARIO: ", e[0])

        for k in data.K_PATHS:
            for t in data.T_TIME_PERIODS:
                for p in data.P_PRODUCTS:
                    weight = modell.h_flow[(str(k), p, t)].value
                    if weight > 0:
                        a_series = pd.Series([k, p, t, weight, e[0]], index=dataset_paths.columns)
                        dataset_paths = dataset_paths.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for l in data.L_LINKS_CAP:
                weight = modell.z_inv_cap[(l[0], l[1], l[2], l[3], t)].value
                if weight > 0:
                    a_series = pd.Series([l[0], l[1], l[2], l[3], t, weight, e[0]], index=dataset_z_inv_cap.columns)
                    dataset_z_inv_cap = dataset_z_inv_cap.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for l in data.L_LINKS_UPG:
                for u in data.UL_UPG[l]:
                    weight1 = modell.z_inv_upg[(l[0], l[1], l[2], l[3], u, t)].value
                    if weight1 > 0:
                        a_series = pd.Series([l[0], l[1], l[2], l[3], u,t, weight1, e[0]],
                                             index=dataset_z_inv_upg.columns)
                        dataset_z_inv_upg = dataset_z_inv_upg.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for (i, m) in data.NM_LIST_CAP:
                for b in data.TERMINAL_TYPE[m]:
                    weight2 = modell.z_inv_node[(i, m, b, t)].value
                    if weight2 > 0:
                        a_series = pd.Series([i, m, b, t, weight2, e[0]],
                                             index=dataset_z_inv_node.columns)
                        dataset_z_inv_node = dataset_z_inv_node.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            for a in data.CHARGING_ARCS:
                weight3 = modell.charge_link[(a[0],a[1],a[2],a[3],a[4],t)].value
                if weight3 > 0:
                    a_series = pd.Series([a[0],a[1],a[2],a[3],a[4],t, weight3, e[0]],
                                         index=dataset_charging.columns)
                    dataset_charging = dataset_charging.append(a_series, ignore_index=True)
        for t in data.T_TIME_PERIODS:
            weight4 = modell.emission_violation[t].value
            if weight4 > 0:
                a_series = pd.Series([t, weight4, e[0]],
                                     index=dataset_violation.columns)
                dataset_violation = dataset_violation.append(a_series, ignore_index=True)
        print('--------- Total emissions -----------')
        for t in data.T_TIME_PERIODS:
            print(e[0],t, "Total emissions: ",modell.total_emissions[t].value,", emission violation: ",
                modell.emission_violation[t].value,", violation/emission_cap: ", 1-(modell.total_emissions[t].value/(data.CO2_CAP[2020]/data.factor)))
    #print("Number of variables: ",modell.nvariables())
    #print("Number of constraints: ",modell.nconstraints())
    dataset.to_csv("Data/Instance_results_write_to_here/Instance" +instance_run+ "/Inst_" +instance_run+ "_X_flow.csv")
    dataset_paths.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_H_flow.csv")
    dataset_z_inv_cap.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_z_inv_cap.csv")
    dataset_z_inv_upg.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_z_inv_upg.csv")
    dataset_z_inv_node.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_z_inv_node.csv")
    dataset_charging.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_charge_link.csv")
    dataset_violation.to_csv("Data/Instance_results_write_to_here/Instance" + instance_run + "/Inst_" + instance_run + "_emission_violation.csv")
    
    return dataset

