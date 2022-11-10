#THIS FILE IS USED AS A SKETCHBOOK FOR DEVELOPING THE SCENARIO CONSTRUCTION STUFF

import os
os.chdir("M:\Documents\GitHub\AIM_Norwegian_Freight_Model")

import pandas as pd

prefix = ''

#read and process fuel group data
fuel_group_data = pd.read_excel(prefix+r'Data/scenarios.xlsx', sheet_name='fuel_groups')

fuel_group_names = [] #list of fuel group names
fuel_groups = {} #dict from fuel group names to (m,f) combinations
for index, row in fuel_group_data.iterrows():
    if row["Fuel group"] not in fuel_group_names:
        fuel_group_names = fuel_group_names + [row["Fuel group"]]    #add fuel group name to list
        fuel_groups[row["Fuel group"]] = [(row["Mode"], row["Fuel"])] #create new entry in dict
    else:
        fuel_groups[row["Fuel group"]] =  fuel_groups[row["Fuel group"]] + [(row["Mode"], row["Fuel"])]  #append entry in dict

#read and process scenario data
scenario_data = pd.read_excel(prefix+r'Data/scenarios.xlsx', sheet_name='scenarios')
num_scenarios = len(scenario_data)

probabilities = [1.0/num_scenarios] * num_scenarios #initialize as equal probabilities
scenario_names = ["scen_" + str(i).zfill(len(str(num_scenarios))) for i in range(num_scenarios)] #initialize as scen_00, scen_01, scen_02, etc.
  
fg_cost_factor = [{}] * num_scenarios
for index, row in scenario_data.iterrows():
    if "Name" in scenario_data:
        scenario_names[index] = row["Name"] #update scenario names if available
    if "Probability" in scenario_data:
        probabilities[index] = row["Probability"] #update probabilities if available
    for fg in fuel_groups:
        new_entry = {fg : row[fg]} #new entry for the dictionary fg_cost_factor[index]
        fg_cost_factor[index] = dict(fg_cost_factor[index], **new_entry) #add new entry to existing dict (trick from internet)

scen_name_to_nr = {}
scen_nr_to_name = {}
for i in range(len(scenario_names)):
    scen_name_to_nr[scenario_names[i]] = i
    scen_nr_to_name[i] = scenario_names[i]


mode_fuel_cost_factor = [] #list of dictionaries (per scenario) from (m,f) pair to transport cost factor (relative to base cost)
for s in range(num_scenarios):
    mode_fuel_cost_factor.append({})
    for fg in fuel_group_names:
        for mf in fuel_groups[fg]:
            mode_fuel_cost_factor[s][mf] = fg_cost_factor[s][fg]

