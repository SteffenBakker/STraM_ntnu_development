from Data.settings import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

import pickle
import json

#---------------------------------------------------------#
#       User Settings
#---------------------------------------------------------#

scenarios = "4Scen"   # '4Scen', 'AllScen'


#---------------------------------------------------------#
#       Output data
#---------------------------------------------------------#

with open(r'Data\\output\\'+"SP"+'_'+scenarios+'.pickle', 'rb') as output_file:
    output_SP = pickle.load(output_file)
with open(r'Data\\output\\'+"EEV"+'_'+scenarios+'.pickle', 'rb') as output_file:
    output_EEV = pickle.load(output_file)

with open(r'Data\base_data\\'+scenarios+'.pickle', 'rb') as data_file:
    base_data = pickle.load(data_file)

if True:


    #---------------------------------------------------------#
    #       COSTS
    #---------------------------------------------------------#

    def cost_and_investment_table(base_data,output):

        keys = list(zip(output.z_emission_violation["time_period"],output.z_emission_violation["scenario"]))
        values = list(output.z_emission_violation["weight"])
        output.costs["EmissionCosts"] = {keys[i]:EMISSION_VIOLATION_PENALTY*values[i] for i in range(len(values))}
        
        cost_vars = ["TranspOpexCost","TranspOpexCostB","TranspCO2Cost","TranspCO2CostB","TransfCost","EdgeCost","NodeCost","UpgCost", "ChargeCost","EmissionCosts"]
        legend_names = {"TranspOpexCost":"OPEX",
            "TranspOpexCostB":"OPEX_Empty",
            "TranspCO2Cost":"Carbon",
            "TranspCO2CostB":"Carbon_Empty",
            "TransfCost":"Transfer",
            "EdgeCost":"Edge",
            "NodeCost":"Node",
            "UpgCost":"Upg", 
            "ChargeCost":"Charge",
            "EmissionCosts":"EmissionPenalty"}
        output.cost_var_colours =  {"OPEX":"royalblue",
            "OPEX_Empty":"cornflowerblue",
            "Carbon":"dimgrey",
            "Carbon_Empty":"silver",
            "Transfer":"brown",
            "Edge":"indianred",
            "Node":"darkred",
            "Upg":"teal", 
            "Charge":"forestgreen",
            "EmissionPenalty":"red"}
        output.all_costs = {legend_names[var]:output.costs[var] for var in cost_vars}
        
        
        #get the right measure:
        for var in cost_vars:
            var2 = legend_names[var]
            for key, value in output.all_costs[var2].items():
                output.all_costs[legend_names[var]][key] = round(value / 10**9*SCALING_FACTOR_MONETARY,3) # in GNOK

        output.all_costs_table = pd.DataFrame.from_dict(output.all_costs, orient='index')
        
        for t in base_data.T_TIME_PERIODS: 
            #t = base_data.T_TIME_PERIODS[0]
            level_values =  output.all_costs_table.columns.get_level_values(1) #move into loop?
            columns = ((output.all_costs_table.columns.get_level_values(0)==t) & 
                        ([level_values[i] in base_data.S_SCENARIOS for i in range(len(level_values))]))
            mean = output.all_costs_table.iloc[:,columns].mean(axis=1)
            std = output.all_costs_table.iloc[:,columns ].std(axis=1)
            output.all_costs_table[(t,'mean')] = mean
            output.all_costs_table[(t,'std')] = std
        output.all_costs_table = output.all_costs_table.fillna(0) #in case of a single scenario we get NA's

        #only select mean and std data (go away from scenarios)
        columns = ((output.all_costs_table.columns.get_level_values(1)=='mean') | (output.all_costs_table.columns.get_level_values(1)=='std'))
        output.all_costs_table = output.all_costs_table.iloc[:,columns ].sort_index(axis=1,level=0)
        
        discount_factors = pd.Series([round(base_data.D_DISCOUNT_RATE**n,3) for n in [t - base_data.T_TIME_PERIODS[0]  for t in base_data.T_TIME_PERIODS for dd in range(2)]],index = output.all_costs_table.columns).to_frame().T #index =
        discount_factors = discount_factors.rename({0:'discount_factor'})

        output.all_costs_table = pd.concat([output.all_costs_table, discount_factors],axis=0, ignore_index=False)

        pd.set_option('display.float_format', '{:.2g}'.format)
        print(round(output.all_costs_table,2))

        return output

    def plot_costs(output,which_costs,ylabel,filename):

        #which_costs = opex_variables
        #ylabel = 'Annual costs (GNOK)'
        
        #indices = [i for i in output.all_costs_table.index if i not in ['discount_factor']]
        all_costs_table2 = output.all_costs_table.loc[which_costs]

        mean_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='mean']
        mean_data = mean_data.droplevel(1, axis=1)
        std_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='std']
        std_data = std_data.droplevel(1, axis=1)
        yerrors = std_data.to_numpy()
        #fig, ax = plt.subplots()
        ax = mean_data.transpose().plot(kind='bar', yerr=yerrors, alpha=0.5, error_kw=dict(ecolor='k'), 
            stacked = True,
            xlabel = 'time periods',
            ylabel = ylabel,
            #title = title,
            color = output.cost_var_colours
            )  
        #print(ax.get_xticklabels())
        # NOT WORKING WITH CATEGORICAL AXIS
        #ax.vlines(60,0,50)
        ax.axvline(x = 1.5, color = 'black',ls='--') 
        ax.text(-0.2, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
        ax.text(1.8, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)

        if filename=='investment':
            ax.legend(loc='upper right')  #upper left
        else:
            ax.legend(loc='lower right')  #upper left

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
        #ax.spines[['right', 'top']].set_visible(False)   #https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        #fig = ax.get_figure()
        ax.get_figure().savefig(r"Data\\Figures\\"+run_identifier+"_costs_"+filename+".pdf",dpi=300,bbox_inches='tight')
    
    #output_SP = cost_and_investment_table(base_data,output_SP)
    #output_EEV = cost_and_investment_table(base_data,output_EEV)

    
    #---------------------------------------------------------#
    #       COST AND EMISSION TRADE-OFF
    #---------------------------------------------------------#
    print('--------------------------')
    for i in [0,1]:
        if i == 0:
            type_analysis = 'SP'
        else:
            type_analysis = 'EEV'
        print('--------')
        print(type_analysis)
        print('--------')
        output = [output_SP,output_EEV][i]
        output.emission_stats = output.total_emissions.groupby('time_period').agg(
                AvgEmission=('weight', np.mean),
                Std=('weight', np.std))
        output.emission_stats = output.emission_stats.fillna(0) #in case of a single scenario we get NA's
        print('Total (average) emissions (in Million TonnesCo2):')
        print(round(sum(output.emission_stats['AvgEmission'])*SCALING_FACTOR_EMISSIONS/10**9,2)) #this becomes Million TonnesCO2         

        print('objective function value: ')
        print(round(output.ob_function_value*SCALING_FACTOR_MONETARY/10**9,2))
        print('objective function value without emission penalty (Billion NOK): ')
        print(round(output.ob_function_value_without_emission*SCALING_FACTOR_MONETARY/10**9,2)) #without emission penalty

    #2000NOK per Tonne CO2 is this in line? Do the comparison for

