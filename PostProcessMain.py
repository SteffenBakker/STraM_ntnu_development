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

analyses_type = "SP" #EV, EEV, 'SP
scenarios = "three_scenarios_new"   # 'three_scenarios_new', 'scenarios_base'
noBalancingTrips = False
last_time_period = True

#---------------------------------------------------------#
#       Output data
#---------------------------------------------------------#

output_file = 'output_data_'+analyses_type+'_'+scenarios
if noBalancingTrips:
    output_file = output_file + '_NoBalancingTrips'
if last_time_period:
    output_file = output_file + '_last_period'
with open(r'Data\\'+output_file, 'rb') as output_file:
    output = pickle.load(output_file)

with open(r'Data\base_data_'+scenarios, 'rb') as data_file:
    base_data = pickle.load(data_file)



print('objective function value: ', output.ob_function_value*SCALING_FACTOR/10**9)
#SHOULD REMOVE THE MISSION PENALTY: DOES NOT MAKE SENSE NOW

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
    ax.get_figure().savefig(r"Data\\Figures\\costs_"+filename+".pdf",dpi=300,bbox_inches='tight')

output = cost_and_investment_table(base_data,output)
opex_variables = ['OPEX', 'OPEX_Empty', 'Carbon','Carbon_Empty', 'Transfer']
investment_variables = ['Edge', 'Node', 'Upg','Charge']
plot_costs(output,opex_variables,'Annual costs (GNOK)',"opex")
plot_costs(output,investment_variables,'Investment costs (GNOK)',"investment")


#Total costs without emission penalty:

#output.ob_function_value

##total_emission_penalty = sum(output.costs["EmissionCosts"].values())]
#for t in base_data.T_TIME_PERIODS:
#    for t in output.scenarios:

#rather resolve the model. FIX EVERYTHING. But remove the penalty from objective...


if False:
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

    #output.emission_stats['AvgEmission_perc'] = output.emission_stats['AvgEmission']/output.emission_stats.at[2020,'AvgEmission']*100 #OLD: 2020
    output.emission_stats['AvgEmission_perc'] = output.emission_stats['AvgEmission']/base_data.EMISSION_CAP_ABSOLUTE_BASE_YEAR*100  #NEW: 2022
    #output.emission_stats['Std_perc'] = output.emission_stats['Std']/output.emission_stats.at[2020,'AvgEmission']*100 #OLD: 2020
    output.emission_stats['Std_perc'] = output.emission_stats['Std']/output.emission_stats.at[2022,'AvgEmission']*100  #NEW: 2022
    goals = list(base_data.EMISSION_CAP_RELATIVE.values())
    output.emission_stats['Goal'] = goals
    output.emission_stats['StdGoals'] = [0 for g in goals]       
    print(output.emission_stats)

    #output.emission_stats['Std'] = 0.1*output.emission_stats['AvgEmission']  #it works when there is some deviation!!
    
    yerrors = output.emission_stats[['Std_perc', 'StdGoals']].to_numpy().T
    ax = output.emission_stats[['AvgEmission_perc', 'Goal']].plot(kind='bar', 
                xlabel = 'time periods',
                ylabel = 'Relative emissions (%)',
                #title = "Emissions",
                yerr=yerrors, alpha=0.5, 
                error_kw=dict(ecolor='k'), stacked = False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    
    ax.axvline(x = 1.5, color = 'black',ls='--') 
    ax.text(0.5, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
    ax.text(1.6, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)

    ax.legend(loc='upper right')  #upper left
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    #ax.spines[['right', 'top']].set_visible(False)   #https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
    #fig = ax.get_figure()
    ax.get_figure().savefig(r"Data\\Figures\\emissions.pdf",dpi=300,bbox_inches='tight')
    
    #fig = ax.get_figure()
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

        leftright = -0.15
        bottom = [0 for i in range(len(base_data.T_TIME_PERIODS))]  
        for f in base_data.FM_FUEL[m]:
            subset = TranspArbAvgScen[(TranspArbAvgScen['mode']==m)&(TranspArbAvgScen['fuel']==f)]
            yerror = subset[base_string+'_std'].tolist()
            #print(yerror)

            #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
            
            trans1 = Affine2D().translate(leftright, 0.0) + ax.transData

            do_not_plot_lims = [False]*len(yerror)
            for i in range(len(yerror)):
                if yerror[i] > 0.001:
                    do_not_plot_lims[i] = True
            
            #print(f,m)
            #print(yerror)
            #print(do_not_plot_lims)
            #it does what it should, but xuplims does not work.. 

            n_start = 5
            for i in range(len(yerror)):
                if yerror[i] > 0.001:
                    n_start = i
                    break
            #this works! (preventing standard deviations to be plotted, when they are not there)

            ax.bar(labels, subset[base_string].tolist(), 
                        width, 
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
            leftright = leftright + 0.05
        ax.set_ylabel(ylabel)
        #ax.set_title(m + ' - ' + analysis_type)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

        ax.axvline(x = 1.5, color = 'black',ls='--') 
        #ax.text(0.5, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
        #ax.text(1.6, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)
        fig.savefig(r"Data\\Figures\\modemix"+m+".pdf",dpi=300,bbox_inches='tight')
        


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


#DEMAND_PER_YEAR = [{(i,j,p):value for (i,j,p,t),value in base_data.D_DEMAND.items() if t==tt} for tt in [2030,2040,2050]]
#pd.DataFrame.from_dict(DEMAND_PER_YEAR,orient='columns').transpose()


#conclusion: this is not the reason for the thing that is happening in 2040! We can go back to the estimate from TØI... 


















if False:


    #------------------------------------------
    #the following should be moved to extractmodelresults.py

    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return obj.to_json(orient='records') #
            return json.JSONEncoder.default(self, obj)

    with open(r'Data\\output_vars.json', 'w') as fp:
        json.dump({ '0':output.all_variables,
                    '1':output.x_flow,
                    '2':output.b_flow,
                    '3':output.h_path,
                    '4':output.y_charging,
                    '5':output.nu_node,
                    '6':output.epsilon_edge,
                    '7':output.upsilon_upgrade,
                    '8':output.z_emission_violation,
                    '9':output.total_emissions,
                    '10':output.q_transp_amount,
                    '11':output.q_max_transp_amount
                    },fp, cls=JSONEncoder)


    costs = output.costs
    for key in costs.keys():
        costs[key] = list(costs[key].items())
    
    with open(r'Data\\output_costs.json', "w") as outfile:
        json.dump(costs, outfile)


    #------------------------------------------
    #then we can start reading here 

    costs = json.load(open(r'Data\\output_costs.json'))
    json_file_output_vars = json.load(open(r'Data\\output_vars.json'))
    class Output():
        def __init__(self,costs,json_file_output_vars):
            
            for key in costs.keys():
                if isinstance(costs[key][0][0],list):
                    costs[key] = {tuple(x[0]):x[1] for x in costs[key]}
                else:
                    costs[key] = {x[0]:x[1] for x in costs[key]}
            self.costs = costs

            self.all_variables =            pd.read_json(json_file_output_vars['0'], orient='records')
            self.x_flow =                   pd.read_json(json_file_output_vars['1'], orient='records')
            self.b_flow=                    pd.read_json(json_file_output_vars['2'], orient='records')
            self.h_path=                    pd.read_json(json_file_output_vars['3'], orient='records')
            self.y_charging=                pd.read_json(json_file_output_vars['4'], orient='records')
            self.nu_node=                   pd.read_json(json_file_output_vars['5'], orient='records')
            self.epsilon_edge=              pd.read_json(json_file_output_vars['6'], orient='records')
            self.upsilon_upgrade=           pd.read_json(json_file_output_vars['7'], orient='records')
            self.z_emission_violation=      pd.read_json(json_file_output_vars['8'], orient='records')
            self.total_emissions=           pd.read_json(json_file_output_vars['9'], orient='records')
            self.q_transp_amount=           pd.read_json(json_file_output_vars['10'], orient='records')
            self.q_max_transp_amount=       pd.read_json(json_file_output_vars['11'], orient='records')
    output = Output(costs,json_file_output_vars)

