from Data.settings import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import copy

import pickle
import json

#---------------------------------------------------------#
#       User Settings
#---------------------------------------------------------#

scenario_tree = "FuelScen"   # FuelScen, FuelDetScen, AllScen, 4Scen, 9Scen

analyses_info = {
#     name    type,    scen,    balancing,  single_tp   risk    carbon,     fee
    "base": ["SP",  scenario_tree,    False,      None,       None,   False,      "base"],
    "eev":  ["EEV", scenario_tree,    False,      None,       None,   False,      "base"],
    "risk1":["SP",  scenario_tree,    False,      None,     'averse',   False,      "base"],
    "risk2":["SP",  scenario_tree,    False,      None,     "neutral",   False,      "base"],
    #"stp1": ["SP",  scenario_tree,    False,      2034,       None,   False,      "base"],
    #"stp2": ["SP",  scenario_tree,    False,      2050,       None,   False,      "base"],
    "carbonhigh": ["SP",  scenario_tree,    False,      None,       None,   True,      "high"],
    "carbonlow": ["SP",  scenario_tree,    False,      None,       None,   True,      "low"],
}

run_all_analyses = False
analysis = "base"    # "base", "carbon1","eev","risk1"....



#COST FUNCTIONS

#create the all_cost_table
def cost_and_investment_table(base_data,output):
    
    cost_vars = ["TranspOpexCost","TranspOpexCostB","TranspCO2Cost","TranspCO2CostB","TranspTimeCost","TransfCost","EdgeCost","NodeCost","UpgCost", "ChargeCost"]
    legend_names = {"TranspOpexCost":"General",
        "TranspOpexCostB":"General (Empty Trips)",
        "TranspCO2Cost":"Carbon",
        "TranspCO2CostB":"Carbon (Empty Trips)",
        "TranspTimeCost":"Time value",
        "TransfCost":"Transfer",
        "EdgeCost":"Edge",
        "NodeCost":"Node",
        "UpgCost":"Upgrade", 
        "ChargeCost":"Charge",
        }
    output.cost_var_colours =  {"General":"royalblue",
        "General (Empty Trips)":"cornflowerblue",
        "Carbon":"dimgrey",
        "Carbon (Empty Trips)":"silver",
        "Time value":"forestgreen",
        "Transfer":"brown",
        "Edge":"indianred",
        "Node":"darkred",
        "Upgrade":"teal", 
        "Charge":"forestgreen",
        }
    
    output.all_costs = {legend_names[var]:output.costs[var] for var in cost_vars}
    
    
    #get the right measure:
    for var in cost_vars:
        var2 = legend_names[var]
        for key, value in output.all_costs[var2].items():
            output.all_costs[legend_names[var]][key] = round(value/10**9*SCALING_FACTOR_MONETARY,3) # in GNOK

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

    return output   

def plot_costs(base_data, output,which_costs,ylabel,filename,run_identifier):

    #which_costs = opex_variables
    #ylabel = 'Annual costs (GNOK)'
    
    #indices = [i for i in output.all_costs_table.index if i not in ['discount_factor']]
    all_costs_table2 = output.all_costs_table.loc[which_costs]

    mean_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='mean']
    mean_data = mean_data.droplevel(1, axis=1)
    mean_data = mean_data.transpose()
    std_data = all_costs_table2.iloc[:,all_costs_table2.columns.get_level_values(1)=='std']
    std_data = std_data.droplevel(1, axis=1)
    #yerrors = std_data.to_numpy()
    std_data = std_data.transpose()

    fig, ax = plt.subplots(figsize=(4, 4))
    
    leftright = -0.25
    bottom = [0 for i in range(len(base_data.T_TIME_PERIODS))]  
    for var in which_costs:

        yerror = list(std_data[var])
        
        trans1 = Affine2D().translate(leftright, 0.0) + ax.transData

        do_not_plot_lims = [False]*len(yerror)
        for i in range(len(yerror)):
            if yerror[i] > 0.001:
                do_not_plot_lims[i] = True

        n_start = 100  #do not plot any std at all 
        for i in range(len(yerror)):
            if yerror[i] > 0.001:
                n_start = i
                break
        #this works (preventing standard deviations to be plotted, when they are not there)

        ax.bar( [str(t) for t in  base_data.T_TIME_PERIODS], 
                    mean_data[var].to_list(),
                    width=0.6, 
                    yerr=yerror, 
                    bottom = bottom,
                    error_kw={#'elinewidth':6,'capsize':6,'capthick':6,'ecolor':'black',
                        'capsize':2,
                        'errorevery':(n_start,1),'transform':trans1, 
                        #'xlolims':do_not_plot_lims,'xuplims':do_not_plot_lims
                        },   #elinewidth, capthickfloat
                    label=var,
                    color=output.cost_var_colours[var]
                    )
        ax.set_ylabel(ylabel)

        bottom = [mean_data[var].to_list()[i]+bottom[i] for i in range(len(bottom))]
        leftright = leftright + 0.1


    if filename == 'investment':
        ax.axis(ymin=0,ymax=7)
    #print(ax.get_xticklabels())
    # NOT WORKING WITH CATEGORICAL AXIS
    #ax.vlines(60,0,50)
    ax.axvline(x = 1.5, color = 'black',ls='--') 
    ax.text(-0.2, 0.95*ax.get_ylim()[1], "First \n stage", fontdict=None)
    ax.text(1.8, 0.95*ax.get_ylim()[1], "Second \n stage", fontdict=None)

    if filename=='investment':
        ax.legend(loc="best")  #upper left      (0.06*ax.get_xlim()[1], 0.06*ax.get_ylim()[1])
        ax.set_ylabel(r"Investment cost (BNOK)")
    else:
        ax.legend(loc='lower right')  #upper left
        ax.set_ylabel(r"Transport cost (BNOK)")

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    #ax.spines[['right', 'top']].set_visible(False)   #https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
    
    fig.tight_layout()
    fig.savefig(r"Data//figures//"+run_identifier+"_costs_"+filename+".png",
                dpi=300,bbox_inches='tight')   #ax.get_figure().savefig

#EMISSION FUNCTIONS

def calculate_emissions(output,base_data,domestic=False):
    output.total_yearly_emissions = {(t,s):0 for t in base_data.T_TIME_PERIODS for s in base_data.S_SCENARIOS} # in MTonnes CO2 equivalents

    x_flow = output.x_flow 
    b_flow = output.b_flow
    
    if domestic:
        x_flow = x_flow[(x_flow['from'].isin(base_data.N_NODES_NORWAY))&(x_flow['to'].isin(base_data.N_NODES_NORWAY))]
        b_flow = b_flow[(b_flow['from'].isin(base_data.N_NODES_NORWAY))&(b_flow['to'].isin(base_data.N_NODES_NORWAY))]

    for index,row in x_flow.iterrows():
        (i,j,m,r,f,p,t,s,value) = (row['from'],row['to'],row['mode'],row['route'],row['fuel'],row['product'],row['time_period'],row['scenario'],row['weight']) 
        output.total_yearly_emissions[(t,s)] += ((base_data.E_EMISSIONS[i,j,m,r,f,p,t]*SCALING_FACTOR_EMISSIONS/SCALING_FACTOR_WEIGHT)*(value*SCALING_FACTOR_WEIGHT))/(10**12) #   gCO2 / tonnes*km     *   tonnes/km     ->  in MTonnes CO2 equivalents
    for index,row in b_flow.iterrows():
        (i,j,m,r,f,v,t,s,value) = (row['from'],row['to'],row['mode'],row['route'],row['fuel'],row['vehicle_type'],row['time_period'],row['scenario'],row['weight'])
        output.total_yearly_emissions[(t,s)] += ((base_data.E_EMISSIONS[i,j,m,r,f, base_data.cheapest_product_per_vehicle[(m,f,t,v)], t]*SCALING_FACTOR_EMISSIONS/SCALING_FACTOR_WEIGHT)*(value*SCALING_FACTOR_WEIGHT))/(10**6*10**6) # in MTonnes CO2 equivalents

    output.total_emissions = pd.DataFrame.from_dict({'time_period': [t for (t,s) in output.total_yearly_emissions.keys()],	
                                                        'weight': list(output.total_yearly_emissions.values())	,
                                                        'scenario': [s for (t,s) in output.total_yearly_emissions.keys()]})
        
    # https://stackoverflow.com/questions/23144784/plotting-error-bars-on-grouped-bars-in-pandas
    output.emission_stats = output.total_emissions.groupby('time_period').agg(
        AvgEmission=('weight', np.mean),
        Std=('weight', np.std))
    output.emission_stats = output.emission_stats.fillna(0) #in case of a single scenario we get NA's

    #output.emission_stats['AvgEmission_perc'] = output.emission_stats['AvgEmission']/output.emission_stats.at[2020,'AvgEmission']*100 #OLD: 2020
    output.emission_stats['AvgEmission_perc'] = output.emission_stats['AvgEmission']/output.total_yearly_emissions[(base_data.T_TIME_PERIODS[0],base_data.S_SCENARIOS[0])]*100  #NEW: 2022
    #output.emission_stats['Std_perc'] = output.emission_stats['Std']/output.emission_stats.at[2020,'AvgEmission']*100 #OLD: 2020
    output.emission_stats['Std_perc'] = output.emission_stats['Std']/output.emission_stats.at[base_data.T_TIME_PERIODS[0],'AvgEmission']*100  #NEW: 2022
    #goals = list(base_data.EMISSION_CAP_RELATIVE.values())
    #output.emission_stats['Goal'] = goals
    #output.emission_stats['StdGoals'] = [0 for g in goals]       

    return output

def plot_emission_results(output,base_data,run_identifier):

    #output.emission_stats['Std'] = 0.1*output.emission_stats['AvgEmission']  #it works when there is some deviation!!
    
    y = output.emission_stats[['AvgEmission_perc']].to_numpy() #to_list()
    y = [item for sublist in y for item in sublist]
    yerrors = output.emission_stats[['Std_perc']].to_numpy() 
    yerrors = [item for sublist in yerrors for item in sublist]
    
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.bar( [str(t) for t in  base_data.T_TIME_PERIODS], 
                y,
                width=0.6, 
                yerr=yerrors
                )
                
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    for year in [2030,2050]:
        ax.axhline(y = base_data.EMISSION_CAP_RELATIVE[year], color = 'black', linestyle = ':')
        ax.text(0.12*ax.get_xlim()[1],base_data.EMISSION_CAP_RELATIVE[year], 'target '+str(year), bbox=props, va='center', ha='center', backgroundcolor='w') #fontsize=12
        

    ax.axvline(x = 1.5, color = 'black',ls='--')
    ax.text(0.55, 0.95*ax.get_ylim()[1], "First \n stage", fontdict=None)
    ax.text(1.7, 0.95*ax.get_ylim()[1], "Second \n stage", fontdict=None)

    ax.axis(ymin=0)
    #ax.legend(loc='upper right')  #upper left
    
    ax.set_ylabel(r"Emissions (% of base year)")

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    #ax.spines[['right', 'top']].set_visible(False)   #https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
    ax.get_figure().savefig(r"Data//figures//"+run_identifier+"_emissions.png",dpi=300,bbox_inches='tight')

# MODE MIX

def mode_mix_calculations(output,base_data):
    
    x_flow = copy.deepcopy(output.x_flow)

    time_period = 2023
    if True:
        x_flow = x_flow[x_flow['time_period']==time_period] 

    x_flow.loc[:, 'Distance'] = 0 # in Tonnes KM
    for index, row in x_flow.iterrows():
        x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

    x_flow['TransportArbeid'] = x_flow['Distance']*x_flow['weight'] /10**9*SCALING_FACTOR_WEIGHT # in MTonnes KM
    
    include_col_in_aggr_no_scen = ['mode','product','time_period']
    include_col_in_aggr = include_col_in_aggr_no_scen +['scenario']
    sum_over = 'TransportArbeid'
    TranspArb = x_flow[include_col_in_aggr+[sum_over]].groupby(include_col_in_aggr, as_index=False).agg({sum_over:'sum'})
    TranspArb = TranspArb[include_col_in_aggr_no_scen+[sum_over]].groupby(include_col_in_aggr_no_scen, as_index=False).agg({sum_over:'mean'})
    TranspArb = TranspArb.sort_values(by=['mode','product'])

    print('Average transport work in GTONNES*km:')
    print(TranspArb)    
    print('')


def mode_fuel_mix_calculations(output,base_data):
    output.x_flow['Distance'] = 0 # in Tonnes KM
    for index, row in output.x_flow.iterrows():
        output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

    output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] /10**9*SCALING_FACTOR_WEIGHT # in GTonnes KM
    TranspArb = output.x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
        
    TotalTranspArb = TranspArb.groupby(['time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
    TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})
    TranspArb = TranspArb.rename(columns={"TransportArbeid": "TranspArb"})

    TranspArb = pd.merge(TranspArb,TotalTranspArb,how='left',on=['time_period','scenario'])
    TranspArb['RelTranspArb'] = 100*TranspArb['TranspArb'] / TranspArb['TransportArbeidTotal']
    
    TranspArb['RelTranspArb_std'] = TranspArb['RelTranspArb']
    TranspArb['TranspArb_std'] = TranspArb['TranspArb']
    MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in base_data.S_SCENARIOS]
    all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
    TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)

    TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','TranspArb','TranspArb_std','RelTranspArb','RelTranspArb_std']].groupby(
        ['mode','fuel','time_period'], as_index=False).agg({'TranspArb':'mean','TranspArb_std':'std','RelTranspArb':'mean','RelTranspArb_std':'std'})
    TranspArbAvgScen = TranspArbAvgScen.fillna(0) #in case of a single scenario we get NA's

    return output, TranspArbAvgScen

def plot_mode_mixes(TranspArbAvgScen, base_data,run_identifier,absolute_transp_work=True):  #result data = TranspArbAvgScen
        
    #https://matplotlib.org/stable/gallery/color/named_colors.html
    color_dict = {'Diesel':                 'firebrick', 
                    'Battery':              'mediumseagreen',
                    'Catenary':              'darkolivegreen',
                    'Ammonia':              'royalblue', 
                    'Hydrogen':             'deepskyblue',
                    'Methanol':              'orchid',     
                    #'Battery electric':     'mediumseagreen',
                    #'Battery train':        'darkolivegreen', 
                    #'Electric train (CL)':  'mediumseagreen', 
                    #'LNG':                  'blue', 
                    'MGO':                  'darkviolet', 
                    #'Biogas':               'teal', 
                    #'Biodiesel':            'darkorange', 
                    #'Biodiesel (HVO)':      'darkorange', 
                    'HFO':                  'firebrick'           
                    }


    labels = [str(t) for t in  base_data.T_TIME_PERIODS]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    base_string = 'TranspArb'
    ylabel = 'Transport work (GTonnes-kilometer)'
    if not absolute_transp_work:
        base_string = 'Rel'+base_string
        ylabel = 'Relative transport work (%)'

    for m in ["Road", "Rail", "Sea"]:

        fig, ax = plt.subplots(figsize=(4,5))

        leftright = -0.25
        bottom = [0 for i in range(len(base_data.T_TIME_PERIODS))]  
        for f in base_data.FM_FUEL[m]:
            subset = TranspArbAvgScen[(TranspArbAvgScen['mode']==m)&(TranspArbAvgScen['fuel']==f)]
            yerror = subset[base_string+'_std'].tolist()

            
            trans1 = Affine2D().translate(leftright, 0.0) + ax.transData

            do_not_plot_lims = [False]*len(yerror)
            for i in range(len(yerror)):
                if yerror[i] > 0.001:
                    do_not_plot_lims[i] = True

            n_start = 5
            for i in range(len(yerror)):
                if yerror[i] > 0.001:
                    n_start = i
                    break
            #this works (preventing standard deviations to be plotted, when they are not there)

            ax.bar(labels, subset[base_string].tolist(), 
                        width=0.6, 
                        yerr=yerror, 
                        bottom = bottom,
                        error_kw={#'elinewidth':6,'capsize':6,'capthick':6,'ecolor':'black',
                            'capsize':2,
                            'errorevery':(n_start,1),'transform':trans1, 
                            #'xlolims':do_not_plot_lims,'xuplims':do_not_plot_lims
                            },   #elinewidth, capthickfloat
                        label=f,
                        color=color_dict[f],)
            bottom = [subset[base_string].tolist()[i]+bottom[i] for i in range(len(bottom))]
            leftright = leftright + 0.08
        
        ax.axvline(x = 1.5, ymin=0, ymax=0.5,color = 'black',ls='--')   #ylim is relative!
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.2*ax.get_ylim()[1])
        #ax.set_title(m)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

        #ax.text(0.5, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
        #ax.text(1.6, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        #plt.show()
        fig.savefig(r"Data//figures//"+run_identifier+"_modemix"+m+".png",dpi=300,bbox_inches='tight')



# ALL TOGETHER
def visualize_results(analyses_type,scenario_tree,
                        noBalancingTrips=False,
                        single_time_period=None,
                        risk_aversion = None,  #None, "averse", "neutral"
                        scen_analysis_carbon = False,
                        carbon_fee = "base",
                        emission_cap = "False"
                      ):

    #---------------------------------------------------------#
    #       Output data
    #---------------------------------------------------------#

    
    run_identifier = scenario_tree+"_carbontax"+carbon_fee
    if emission_cap:
        run_identifier = run_identifier + "_emissioncap"
    run_identifier2 = run_identifier+"_"+analyses_type

    # if single_time_period is not None:
    #     run_identifier = run_identifier + '_single_time_period_'+str(single_time_period)
    #     data_file = data_file + '_single_time_period_'+str(single_time_period)
    # if risk_aversion is not None:
    #     run_identifier = run_identifier + '_' + risk_aversion


    with open(r'Data//Output//'+run_identifier+'_basedata.pickle', 'rb') as output_file:
        base_data = pickle.load(output_file)
    with open(r'Data//Output//'+run_identifier2+'_results.pickle', 'rb') as data_file:
        output = pickle.load(data_file)

    print('objective function value: ', output.ob_function_value)
    print('objective function value normalized (BILLION NOK): ', round(output.ob_function_value/10**9*SCALING_FACTOR_MONETARY,2))  


    #---------------------------------------------------------#
    #       COSTS
    #---------------------------------------------------------#

    #create the all_cost_table
    output = cost_and_investment_table(base_data,output)
    pd.set_option('display.float_format', '{:.2g}'.format)
    print(round(output.all_costs_table,2))

    opex_variables = ['General', 'General (Empty Trips)', 'Carbon','Carbon (Empty Trips)','Time value', 'Transfer']
    investment_variables = ['Edge', 'Node', 'Upgrade','Charge']
    plot_costs(base_data, output,which_costs=opex_variables,ylabel='Annual costs (GNOK)',filename="opex",run_identifier=run_identifier )
    plot_costs(base_data, output,investment_variables,'Investment costs (GNOK)',"investment",run_identifier=run_identifier)

    #---------------------------------------------------------#
    #       EMISSIONS 
    #---------------------------------------------------------#

    if single_time_period is None:
        
        output = calculate_emissions(output,base_data,domestic=False)
        print('emissions (MTonnes CO2 eq.):')
        print(output.emission_stats)
        print('----------------')
        #output_domestic = calculate_emissions(output,base_data,domestic=True)
        #print(output_domestic.emission_stats)
        

        #I 2021 var de samlede utslippene fra transport 16,2 millioner tonn CO2-ekvivalenter, 8M tonnes er freight transport
        # https://miljostatus.miljodirektoratet.no/tema/klima/norske-utslipp-av-klimagasser/klimagassutslipp-fra-transport/

        #We are off with a factor 100!?
            
        plot_emission_results(output,base_data,run_identifier)

    #---------------------------------------------------------#
    #       MODE MIX
    #---------------------------------------------------------#

    mode_mix_calculations(output,base_data)        
    


    output,TranspArbAvgScen = mode_fuel_mix_calculations(output,base_data)
    plot_mode_mixes(TranspArbAvgScen,base_data,run_identifier=run_identifier,absolute_transp_work=True)

if __name__ == "__main__":
    

    analyses = [analysis]
    if run_all_analyses == True:
        analyses = list(analyses_info.keys())
    
    for analysis in analyses:
        visualize_results(analyses_type=            analyses_info[analysis][0],
                            scenario_tree =             analyses_info[analysis][1],
                            noBalancingTrips=       analyses_info[analysis][2],
                            single_time_period=     analyses_info[analysis][3],
                            risk_aversion =         analyses_info[analysis][4],
                            scen_analysis_carbon =  analyses_info[analysis][5],
                            carbon_fee =         analyses_info[analysis][6],
                            )