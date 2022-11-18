from Data.settings import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

import pickle

#---------------------------------------------------------#
#       Output data
#---------------------------------------------------------#

with open(r'Data\output_data', 'rb') as output_file:
        output = pickle.load(output_file)

with open(r'Data\base_data', 'rb') as data_file:
        base_data = pickle.load(data_file)


#---------------------------------------------------------#
#       COSTS
#---------------------------------------------------------#
def plot_costs(output):
    for i in range(2):
        if i == 0:
            indices = output.all_costs_table.index
        elif i ==1:
            indices = [i for i in output.all_costs_table.index if i not in ['emission','max_transp_amount_penalty']]
        all_costs_table2 = output.all_costs_table.loc[indices]
        mean_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='mean']
        mean_data = mean_data.droplevel(1, axis=1)
        std_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='std']
        std_data = std_data.droplevel(1, axis=1)
        yerrors = std_data.to_numpy()
        ax = mean_data.transpose().plot(kind='bar', yerr=yerrors, alpha=0.5, error_kw=dict(ecolor='k'), stacked = True)  #xlabel, ylabel
        #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
        fig = ax.get_figure()
        #fig.savefig('/path/to/figure.pdf')



def cost_and_investment_table(base_data,output):
    
    transport_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    transport_costs_empty = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    transfer_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    edge_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    node_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    upgrade_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios} 
    charging_costs = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    emission_violation_penalty = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
    max_transport_amount_penalty = {(t,scen):0 for t in base_data.T_TIME_PERIODS for scen in output.scenarios}
     
    output.all_variables['cost_contribution'] = 0
    
    for index, row in output.all_variables.iterrows():
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
        v = row['vehicle_type']

        cost_contribution = 0
        if variable == 'x_flow':
            cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*(base_data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+
                                                                    base_data.C_CO2[(i,j,m,r,f,p,t)])*value 
                                                                    for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
            transport_costs[(t,s)] += cost_contribution
        if variable == 'b_flow':
            cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*(0)*value   # TO DO: set the right value
                                                                    for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
            transport_costs[(t,s)] += cost_contribution
        elif variable == 'h_path':
            cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*base_data.C_TRANSFER[(kk,p)]*value for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
            transfer_costs[(t,s)] += cost_contribution
        elif variable == 'epsilon_edge':
            cost_contribution = base_data.C_EDGE_RAIL[e]*value
            edge_costs[(t,s)] += cost_contribution
        elif variable == 'upsilon_upg':
            cost_contribution = base_data.C_UPG[(e,f)]*value
            upgrade_costs[(t,s)] += cost_contribution
        elif variable == 'nu_node':
            cost_contribution = base_data.C_NODE[(i,c,m)]*value
            node_costs[(t,s)] += cost_contribution
        elif variable == 'y_charging':
            cost_contribution = base_data.C_CHARGE[(e,f)]*value
            charging_costs[(t,s)] += cost_contribution
        elif variable == 'z_emission':
            cost_contribution = EMISSION_VIOLATION_PENALTY*value
            emission_violation_penalty[(t,s)] += cost_contribution
        elif variable == 'q_max_transp_amount':
            cost_contribution = MAX_TRANSPORT_AMOUNT_PENALTY*value
            max_transport_amount_penalty[(t,s)] += cost_contribution

        output.all_variables.at[index,'cost_contribution'] = cost_contribution
    
    #%columns_of_interest = output.all_variables.loc[:,('variable','time_period','scenario','cost_contribution')]
    output.aggregated_values =  output.all_variables.groupby(['variable', 'time_period', 'scenario']).agg({'cost_contribution':'sum', 'weight':'sum'})
    #https://stackoverflow.com/questions/46431243/pandas-dataframe-groupby-how-to-get-sum-of-multiple-columns
    
    
    output.all_costs = dict(transport=transport_costs,transfer=transfer_costs,edge=edge_costs, upgrade=upgrade_costs,node=node_costs,charging=charging_costs,
                            emission=emission_violation_penalty,max_transp_amount_penalty=max_transport_amount_penalty)
    output.all_costs_table = pd.DataFrame.from_dict(output.all_costs, orient='index')
    
    
    for t in base_data.T_TIME_PERIODS: 
        #t = base_data.T_TIME_PERIODS[0]
        level_values =  output.all_costs_table.columns.get_level_values(1)
        columns = ((output.all_costs_table.columns.get_level_values(0)==t) & 
                    ([level_values[i] in output.scenarios for i in range(len(level_values))]))
        mean = output.all_costs_table.iloc[:,columns].mean(axis=1)
        std = output.all_costs_table.iloc[:,columns ].std(axis=1)
        output.all_costs_table[(t,'mean')] = mean
        output.all_costs_table[(t,'std')] = std
    
    #only select mean and std data (go away from scenarios)
    columns = ((output.all_costs_table.columns.get_level_values(1)=='mean') | (output.all_costs_table.columns.get_level_values(1)=='std'))
    output.all_costs_table = output.all_costs_table.iloc[:,columns ].sort_index(axis=1,level=0)
    
    pd.set_option('display.float_format', '{:.2g}'.format)
    print(round(output.all_costs_table,1))

    plot_costs(output)



cost_and_investment_table(base_data,output)



#---------------------------------------------------------#
#       EMISSIONS 
#---------------------------------------------------------#

def emission_results(output,base_data):
    # print('to do')
    # output.total_emissions
    # for t in base_data.T_TIME_PERIODS:
    #     for scen in output.scenarios:
    #         print(modell.total_emissions[t].value / base_data.CO2_CAP[2020])
    
    #create bare chart figure -> See my drawing
    #https://stackoverflow.com/questions/46794373/make-a-bar-graph-of-2-variables-based-on-a-dataframe
    #https://pythonforundergradengineers.com/python-matplotlib-error-bars.html

    #total_emissions time_period weight scenario
    #means = list(output.total_emissions.groupby(['time_period']).agg({'weight':'mean'})['weight'])
    #errors = list(output.total_emissions.groupby(['time_period']).agg({'weight':'std'})['weight'])

    
    #z_emission_violation

    # https://stackoverflow.com/questions/23144784/plotting-error-bars-on-grouped-bars-in-pandas
    output.emission_stats = output.total_emissions.groupby('time_period').agg(
        AvgEmission=('weight', np.mean),
        Std=('weight', np.std))
    goals = list(base_data.CO2_CAP.values())
    output.emission_stats['Goal'] = goals
    output.emission_stats['StdGoals'] = [0 for g in goals]       
    
    #output.emission_stats['Std'] = 0.1*output.emission_stats['AvgEmission']  #it works when there is some deviation!!
    
    yerrors = output.emission_stats[['Std', 'StdGoals']].to_numpy().T
    
    ax = output.emission_stats[['AvgEmission', 'Goal']].plot(kind='bar', yerr=yerrors, alpha=0.5, error_kw=dict(ecolor='k'), stacked = False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    fig = ax.get_figure()
    #fig.savefig('/path/to/figure.pdf')
    



    #I 2021 var de samlede utslippene fra transport 16,2 millioner tonn CO2-ekvivalenter
    # https://miljostatus.miljodirektoratet.no/tema/klima/norske-utslipp-av-klimagasser/klimagassutslipp-fra-transport/
    # This is from all transport. We do not have all transport, 
    # But our base case in 2020 is 40 millioner tonn CO2! equivalents, probably because of international transport...?
    

#---------------------------------------------------------#
#       MODE MIX
#---------------------------------------------------------#

def plot_mode_mixes(TranspArbAvgScen,base_data):  #result data = TranspArbAvgScen
    color_sea = iter(cm.Blues(np.linspace(0.3,1,7)))
    color_road = iter(cm.Reds(np.linspace(0.4,1,5)))
    color_rail = iter(cm.Greens(np.linspace(0.25,1,5)))

    labels = [str(t) for t in  base_data.T_TIME_PERIODS]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    color_dict = {}
    for m in ["Road", "Rail", "Sea"]:
        for f in base_data.FM_FUEL[m]:
            if m == "Road":
                color_dict[m,f] = next(color_road)
            elif m == "Rail":
                color_dict[m, f] = next(color_rail)
            elif m == "Sea":
                color_dict[m, f] = next(color_sea)

    for m in ["Road", "Rail", "Sea"]:

        fig, ax = plt.subplots()

        bottom = [0 for i in range(len(base_data.T_TIME_PERIODS))]  
        for f in base_data.FM_FUEL[m]:
            subset = TranspArbAvgScen[(TranspArbAvgScen['mode']==m)&(TranspArbAvgScen['fuel']==f)]
            ax.bar(labels, subset['RelTranspArb'].tolist(), width, yerr=subset['RelTranspArb_std'].tolist(), 
                        bottom = bottom,label=f,color=color_dict[m,f])
            bottom = [subset['RelTranspArb'].tolist()[i]+bottom[i] for i in range(len(bottom))]
        ax.set_ylabel('Transport work share (%)')
        ax.set_title(m)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

        plt.show()


#DO THE FOLLOWING FOR DOMESTIC AND INTERNATIONAL (remove nodes from and to europe and the world)
def mode_mix_calculations(output,base_data):
    output.x_flow['Distance'] = 0 # in Tonnes KM
    for index, row in output.x_flow.iterrows():
            output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

    output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] # in Tonnes KM

    subset_x_flow = output.x_flow
    subset_x_flow_domestic = subset_x_flow[(subset_x_flow['from'].isin(base_data.N_NODES_NORWAY)) & (subset_x_flow['to'].isin( base_data.N_NODES_NORWAY)) ]

    for ss_x_flow in [subset_x_flow,subset_x_flow_domestic]:
        #ss_x_flow = subset_x_flow
        TranspArb = ss_x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
        TotalTranspArb = TranspArb.groupby(['time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
        TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})
        print(TotalTranspArb)

        TranspArb = pd.merge(TranspArb,TotalTranspArb,how='left',on=['time_period','scenario'])
        TranspArb['RelTranspArb'] = TranspArb['TransportArbeid'] / TranspArb['TransportArbeidTotal']
        
        TranspArb['RelTranspArb_std'] = TranspArb['RelTranspArb']
        MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in base_data.scenario_information.scenario_names]
        all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
        TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)

        TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','RelTranspArb','RelTranspArb_std']].groupby(['mode','fuel','time_period'], as_index=False).agg({'RelTranspArb':'mean','RelTranspArb_std':'std'})

        plot_mode_mixes(TranspArbAvgScen,base_data)

    return output

output = mode_mix_calculations(output,base_data)
