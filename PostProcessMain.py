from Data.settings import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

import pickle

#---------------------------------------------------------#
#       User Settings
#---------------------------------------------------------#

analyses_type = 'SP' # EEV, 'SP

#---------------------------------------------------------#
#       Output data
#---------------------------------------------------------#


with open(r'Data\output_data_'+analyses_type, 'rb') as output_file:
    output = pickle.load(output_file)

with open(r'Data\base_data', 'rb') as data_file:
    base_data = pickle.load(data_file)


print('objective function value: ', output.ob_function_value*SCALING_FACTOR/10**9)
#---------------------------------------------------------#
#       Accuracy of Q MAX approximation due to penalty -> should become zero
#---------------------------------------------------------#

def accuracy_of_q_max(output,base_data):
    max_transp_amount_df = None

    temp_df = output.q_max_transp_amount
    temp_df = temp_df.rename(columns={'weight': 'max_value'})

    max_transp_amount_df = pd.merge(output.q_transp_amount,temp_df.drop('variable',axis=1),how='left',on=['mode','fuel','scenario'])
    max_transp_amount_df = max_transp_amount_df.sort_values(by=['mode','fuel','scenario','time_period']).reset_index()
    max_transp_amount_df['max_value_true'] = 0

    for index,row in max_transp_amount_df.iterrows():
        m = row['mode']
        f = row['fuel']
        t = row['time_period']
        s = row['scenario']
        max_q = 0
        subset = max_transp_amount_df[(max_transp_amount_df['mode']==m) & (max_transp_amount_df['fuel']==f) &
                                (max_transp_amount_df['scenario']==s)]
        for tau in base_data.T_TIME_PERIODS:
            #if (tau <= t):
            if len(subset[subset['time_period']==tau])==1:
            # pick the specific row
                val = subset[subset['time_period']==tau]['weight'].iloc[0]
                if (val > max_q):
                    max_q = val
        max_transp_amount_df.at[index,'max_value_true'] = max_q
    max_transp_amount_df['diff'] = max_transp_amount_df['max_value']-max_transp_amount_df['max_value_true']

    return round(sum(max_transp_amount_df['diff']),2)

result_q_max = accuracy_of_q_max(output,base_data)
print('--------------------------------------------------------')
print('Total deviation from tha actual max transport amount: ' + str(result_q_max))
print('--------------------------------------------------------')
if result_q_max>1:
    #pass
    raise Exception('Total deviation from tha actual max transport amount: ' + str(result_q_max))


#---------------------------------------------------------#
#       COSTS
#---------------------------------------------------------#
def plot_costs(output,which_costs,ylabel,title):

    #indices = [i for i in output.all_costs_table.index if i not in ['discount_factor']]
    all_costs_table2 = output.all_costs_table.loc[which_costs]

    mean_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='mean']
    mean_data = mean_data.droplevel(1, axis=1)
    std_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='std']
    std_data = std_data.droplevel(1, axis=1)
    yerrors = std_data.to_numpy()
    ax = mean_data.transpose().plot(kind='bar', yerr=yerrors, alpha=0.5, error_kw=dict(ecolor='k'), 
        stacked = True,
        xlabel = 'time periods',
        ylabel = ylabel,
        title = title,
        color = output.cost_var_colours
        )  
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    fig = ax.get_figure()
    #fig.savefig('/path/to/figure.pdf')

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
            output.all_costs[legend_names[var]][key] = round(value / 10**9*SCALING_FACTOR,3) # in GNOK

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

output = cost_and_investment_table(base_data,output)
opex_variables = ['OPEX', 'OPEX_Empty', 'Carbon','Carbon_Empty', 'Transfer']
investment_variables = ['Edge', 'Node', 'Upg','Charge']
plot_costs(output,opex_variables,'Annual costs (GNOK)',"Annual operational costs")
plot_costs(output,investment_variables,'Investment costs (GNOK)',"(One-time) Investment costs")


if sum(output.z_emission_violation["weight"])>1:
    raise Exception('We cannot decarbonize in time scenario:  -> z_emission_violation is non-negative')

#---------------------------------------------------------#
#       EMISSIONS 
#---------------------------------------------------------#

def calculate_emissions_base_year(x_flow,b_flow,base_data,domestic=False):
    print('')
    if domestic:
        x_flow = x_flow[(x_flow['from'].isin(base_data.N_NODES_NORWAY))&(x_flow['to'].isin(base_data.N_NODES_NORWAY))]
        b_flow = b_flow[(b_flow['from'].isin(base_data.N_NODES_NORWAY))&(x_flow['to'].isin(base_data.N_NODES_NORWAY))]
        print('domestic emissions (millioner tonn CO2 equivalents)')
    else:
        print('all emissions (millioner tonn CO2 equivalents)')
    print('')
    emissions_direct = 0
    emission_empty = 0
    t0 = base_data.T_TIME_PERIODS[0]
    for index,row in x_flow[x_flow["time_period"]==t0].iterrows():
        (i,j,m,r,f,p,value) = (row['from'],row['to'],row['mode'],row['route'],row['fuel'],row['product'],row['weight'])
        emissions_direct += base_data.E_EMISSIONS[i,j,m,r,f, p, t0]*value/10**6*SCALING_FACTOR # in MTonnes CO2 equivalents
    for index,row in b_flow[b_flow["time_period"]==t0].iterrows():
        (i,j,m,r,f,v,value) = (row['from'],row['to'],row['mode'],row['route'],row['fuel'],row['vehicle_type'],row['weight'])
        emission_empty += base_data.E_EMISSIONS[i,j,m,r,f, base_data.cheapest_product_per_vehicle[(m,f,t0,v)], t0]*value/10**6*SCALING_FACTOR # in MTonnes CO2 equivalents
    print('direct: ',round(emissions_direct*10**(-6),2))
    print('indirect: ',round(emission_empty*10**(-6),2))
    print('both: ',round((emissions_direct+emission_empty)*10**(-6),2))

for domestic in [True,False]:
    calculate_emissions_base_year(output.x_flow,output.b_flow,base_data,domestic)


def plot_emission_results(output,base_data):

    #create bar chart figure -> See my drawing
    #https://stackoverflow.com/questions/46794373/make-a-bar-graph-of-2-variables-based-on-a-dataframe
    #https://pythonforundergradengineers.com/python-matplotlib-error-bars.html

    #total_emissions time_period weight scenario
    #means = list(output.total_emissions.groupby(['time_period']).agg({'weight':'mean'})['weight'])
    #errors = list(output.total_emissions.groupby(['time_period']).agg({'weight':'std'})['weight'])

    
    
    # https://stackoverflow.com/questions/23144784/plotting-error-bars-on-grouped-bars-in-pandas
    output.emission_stats = output.total_emissions.groupby('time_period').agg(
        AvgEmission=('weight', np.mean),
        Std=('weight', np.std))
    output.emission_stats = output.emission_stats.fillna(0) #in case of a single scenario we get NA's

    output.emission_stats['AvgEmission_perc'] = output.emission_stats['AvgEmission']/output.emission_stats.at[2020,'AvgEmission']*100
    output.emission_stats['Std_perc'] = output.emission_stats['Std']/output.emission_stats.at[2020,'AvgEmission']*100
    goals = list(base_data.CO2_CAP.values())
    output.emission_stats['Goal'] = goals
    output.emission_stats['StdGoals'] = [0 for g in goals]       
    print(output.emission_stats)

    #output.emission_stats['Std'] = 0.1*output.emission_stats['AvgEmission']  #it works when there is some deviation!!
    
    yerrors = output.emission_stats[['Std_perc', 'StdGoals']].to_numpy().T
    ax = output.emission_stats[['AvgEmission_perc', 'Goal']].plot(kind='bar', 
                xlabel = 'time periods',
                ylabel = 'Relative emissions (%)',
                title = "Emissions",
                yerr=yerrors, alpha=0.5, 
                error_kw=dict(ecolor='k'), stacked = False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    fig = ax.get_figure()
    #fig.savefig('/path/to/figure.pdf')
    
#I 2021 var de samlede utslippene fra transport 16,2 millioner tonn CO2-ekvivalenter, 8M tonnes er freight transport
# https://miljostatus.miljodirektoratet.no/tema/klima/norske-utslipp-av-klimagasser/klimagassutslipp-fra-transport/

    
plot_emission_results(output,base_data)

#---------------------------------------------------------#
#       MODE MIX
#---------------------------------------------------------#

def plot_mode_mixes(TranspArbAvgScen, base_data,absolute_transp_work=True, analysis_type="All transport"):  #result data = TranspArbAvgScen
    # color_sea = iter(cm.Blues(np.linspace(0.3,1,7)))
    # color_road = iter(cm.Reds(np.linspace(0.4,1,5)))
    # color_rail = iter(cm.Greens(np.linspace(0.25,1,5)))

    # color_dict = {}
    # for m in ["Road", "Rail", "Sea"]:
    #     for f in base_data.FM_FUEL[m]:
    #         if m == "Road":
    #             color_dict[m,f] = next(color_road)
    #         elif m == "Rail":
    #             color_dict[m, f] = next(color_rail)
    #         elif m == "Sea":
    #             color_dict[m, f] = next(color_sea)

    #https://matplotlib.org/stable/gallery/color/named_colors.html
    color_dict = {'Diesel':                 'firebrick', 
                    'Ammonia':              'royalblue', 
                    'Hydrogen':             'deepskyblue', 
                    'Battery electric':     'mediumseagreen',
                    'Battery train':        'darkolivegreen', 
                    'Electric train (CL)':  'mediumseagreen', 
                    'LNG':                  'blue', 
                    'MGO':                  'darkviolet', 
                    'Biogas':               'teal', 
                    'Biodiesel':            'darkorange', 
                    'Biodiesel (HVO)':      'darkorange', 
                    'HFO':                  'firebrick'           }


    labels = [str(t) for t in  base_data.T_TIME_PERIODS]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    base_string = 'TranspArb'
    ylabel = 'Transport work (GTonnes-kilometer)'
    if not absolute_transp_work:
        base_string = 'Rel'+base_string
        ylabel = 'Relative transport work (%)'

    for m in ["Road", "Rail", "Sea"]:

        fig, ax = plt.subplots()

        bottom = [0 for i in range(len(base_data.T_TIME_PERIODS))]  
        for f in base_data.FM_FUEL[m]:
            subset = TranspArbAvgScen[(TranspArbAvgScen['mode']==m)&(TranspArbAvgScen['fuel']==f)]
            ax.bar(labels, subset[base_string].tolist(), width, yerr=subset[base_string+'_std'].tolist(), 
                        bottom = bottom,label=f,color=color_dict[f])
            bottom = [subset[base_string].tolist()[i]+bottom[i] for i in range(len(bottom))]
        ax.set_ylabel(ylabel)
        ax.set_title(m + ' - ' + analysis_type)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

        plt.show()


#DO THE FOLLOWING FOR DOMESTIC AND INTERNATIONAL (remove nodes from and to europe and the world)
def mode_mix_calculations(output,base_data):
    output.x_flow['Distance'] = 0 # in Tonnes KM
    for index, row in output.x_flow.iterrows():
        output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

    output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] /10**9*SCALING_FACTOR # in GTonnes KM

    #for i in [1]:  #we only do one type, discuss what foreign transport needs to be excluded
    i = 1
    #if i == 0:
    #    analysis_type = 'Domestic'
    #    ss_x_flow = output.x_flow[(output.x_flow['from'].isin(base_data.N_NODES_NORWAY)) & (output.x_flow['to'].isin( base_data.N_NODES_NORWAY)) ]
    #if i == 1:
    analysis_type = 'All transport'
    ss_x_flow = output.x_flow

    TranspArb = ss_x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
    
    #output.q_transp_amount[output.q_transp_amount["time_period"]==2050]
    
    TotalTranspArb = TranspArb.groupby(['time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
    TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})
    TranspArb = TranspArb.rename(columns={"TransportArbeid": "TranspArb"})

    TranspArb = pd.merge(TranspArb,TotalTranspArb,how='left',on=['time_period','scenario'])
    TranspArb['RelTranspArb'] = 100*TranspArb['TranspArb'] / TranspArb['TransportArbeidTotal']
    
    TranspArb['RelTranspArb_std'] = TranspArb['RelTranspArb']
    TranspArb['TranspArb_std'] = TranspArb['TranspArb']
    MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in output.scenarios]
    all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
    TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)

    TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','TranspArb','TranspArb_std','RelTranspArb','RelTranspArb_std']].groupby(
        ['mode','fuel','time_period'], as_index=False).agg({'TranspArb':'mean','TranspArb_std':'std','RelTranspArb':'mean','RelTranspArb_std':'std'})
    TranspArbAvgScen = TranspArbAvgScen.fillna(0) #in case of a single scenario we get NA's

    plot_mode_mixes(TranspArbAvgScen,base_data,absolute_transp_work=True,analysis_type=analysis_type)

    return output

output = mode_mix_calculations(output,base_data)



# dir(output)
# decrease = self.model.q_transp_amount[(m,f,self.data.T_MIN1[t])] - self.model.q_transp_amount[(m,f,t)]
# factor = (t - self.data.T_MIN1[t]) / self.data.LIFETIME[(m,f)]
# return (decrease <= factor*self.model.q_max_transp_amount[m,f,t])

#---------------------------------------------------------#
#       DEMAND ANALYSIS
#---------------------------------------------------------#

base_data.D_DEMAND_AGGR
total_demand = {t:0 for t in base_data.T_TIME_PERIODS}
total_demand_european = {t:0 for t in base_data.T_TIME_PERIODS}
total_demand_domestic = {t:0 for t in base_data.T_TIME_PERIODS}
for key,value in base_data.D_DEMAND.items():
    (i,j,p,t) = key
    val = round(value/10**6*SCALING_FACTOR,2)
    total_demand[t] += val
    if (i not in ['Europa','Verden']) and (j not in ['Europa','Verden']):
        total_demand_domestic[t] += val
    if (i not in ['Verden']) and (j not in ['Verden']):
         total_demand_european[t] += val


data = [total_demand,total_demand_european,total_demand_domestic]
demand_overview = pd.DataFrame.from_dict(data,orient='columns') 
demand_overview.index = ['all','european','domestic']
print('total demand in MTonnes')
print(demand_overview)


DEMAND_PER_YEAR = [{(i,j,p):value for (i,j,p,t),value in base_data.D_DEMAND.items() if t==tt} for tt in [2030,2040,2050]]
pd.DataFrame.from_dict(DEMAND_PER_YEAR,orient='columns').transpose()


#conclusion: this is not the reason for the thing that is happening in 2040! We can go back to the estimate from TØI... 


#---------------------------------------------------------#
#       NO variation in 2050 (and 2030)
#---------------------------------------------------------#


base_data.scenario_information.mode_fuel_cost_factor

output.q_transp_amount[output.q_transp_amount['time_period']==2050].sort_values(by=['mode','fuel']).reset_index()

# use to check
#q_transp_amount	Sea	HFO	2030	1e+06	HLL
#25	95	q_transp_amount	Sea	HFO	2030	1e+06	LHL
#26	143	q_transp_amount	Sea	HFO	2030	1e+06	LLH























#---------------------------------------------------------#
#       Depreciated
#---------------------------------------------------------#



def cost_and_investment_table_old(base_data,output):

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

        cost_contribution = 0  #operational costs are discounted to their base years, not more!
        if variable == 'x_flow':
            cost_contribution = sum(base_data.D_DISCOUNT_RATE**n*(base_data.C_TRANSP_COST[(i,j,m,r,f,p,t)]+
                                                                    base_data.C_CO2[(i,j,m,r,f,p,t)])*value 
                                                                    for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
            transport_costs[(t,s)] += cost_contribution
        if variable == 'b_flow':
            cost_contribution = sum(EMPTY_VEHICLE_FACTOR*base_data.D_DISCOUNT_RATE**n*(base_data.C_TRANSP_COST[(i,j,m,r,f,base_data.cheapest_product_per_vehicle[(m,f,t,v)],t)]+
                                                                    base_data.C_CO2[(i,j,m,r,f,base_data.cheapest_product_per_vehicle[(m,f,t,v)],t)]
                                                                    )*value  
                                                                    for n in [nn-base_data.Y_YEARS[t][0] for nn in base_data.Y_YEARS[t]])
            transport_costs_empty[(t,s)] += cost_contribution
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
            for t in base_data.T_TIME_PERIODS:
                cost_contribution = MAX_TRANSPORT_AMOUNT_PENALTY/len(base_data.T_TIME_PERIODS)*value
                max_transport_amount_penalty[(t,s)] += cost_contribution

        output.all_variables.at[index,'cost_contribution'] = cost_contribution
    
    #%columns_of_interest = output.all_variables.loc[:,('variable','time_period','scenario','cost_contribution')]
    output.aggregated_values =  output.all_variables.groupby(['variable', 'time_period', 'scenario']).agg({'cost_contribution':'sum', 'weight':'sum'})
    #https://stackoverflow.com/questions/46431243/pandas-dataframe-groupby-how-to-get-sum-of-multiple-columns
    #this one seems not to be used actually    

    output.all_costs = dict(transport=transport_costs,transport_empty=transport_costs_empty,transfer=transfer_costs,edge=edge_costs, upgrade=upgrade_costs,node=node_costs,charging=charging_costs,
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
