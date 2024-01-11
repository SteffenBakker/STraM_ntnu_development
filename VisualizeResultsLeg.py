import ast
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

scenarios = "4Scen"   # AllScen, 4Scen, 9Scen

analyses_info = {
#     name    type,    scen,    balancing,  single_tp   risk    carbon,     factor
    "base": ["SP",  scenarios,    False,      None,       None,   False,      1],
    "eev":  ["EEV", scenarios,    False,      None,       None,   False,      1],
    "risk1":["SP",  scenarios,    False,      None,     'averse',   False,      1],
    "risk2":["SP",  scenarios,    False,      None,     "neutral",   False,      1],
    #"stp1": ["SP",  scenarios,    False,      2034,       None,   False,      1],
    #"stp2": ["SP",  scenarios,    False,      2050,       None,   False,      1],
    "carbon1": ["SP",  scenarios,    False,      None,       None,   True,      0],
    "carbon2": ["SP",  scenarios,    False,      None,       None,   True,      2],
}

run_all_analyses = False
analysis = "base"    # "base", "eev","risk1"....



#------------------------------------------------------#



def visualize_results(analyses_type,scenarios,
                        noBalancingTrips=False,
                        single_time_period=None,
                        risk_aversion = None,  #None, "averse", "neutral"
                        scen_analysis_carbon = False,
                        carbon_factor = 1
                      ):

    #---------------------------------------------------------#
    #       Output data
    #---------------------------------------------------------#

    data_file = scenarios
    run_identifier = analyses_type+'_'+scenarios
    if noBalancingTrips:
        run_identifier = run_identifier + '_NoBalancingTrips'
    if single_time_period is not None:
        run_identifier = run_identifier + '_single_time_period_'+str(single_time_period)
        data_file = data_file + '_single_time_period_'+str(single_time_period)
    if risk_aversion is not None:
        run_identifier = run_identifier + '_' + risk_aversion
    if scen_analysis_carbon:
        add_string = "_co2_factor_"+str(carbon_factor)
        run_identifier = run_identifier + add_string
        data_file = data_file + add_string

    with open(r'Data//output//'+run_identifier+'.pickle', 'rb') as output_file:
        output = pickle.load(output_file)
    with open(r'Data//base_data//'+scenarios+'.pickle', 'rb') as data_file:
        base_data = pickle.load(data_file)


    print('objective function value: ', output.ob_function_value)
    print('objective function value normalized (BILLION NOK): ', round(output.ob_function_value/10**9*SCALING_FACTOR_MONETARY,2))  



    #---------------------------------------------------------#
    #       COSTS
    #---------------------------------------------------------#

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

    def plot_costs(output,which_costs,ylabel,filename):

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

    output = cost_and_investment_table(base_data,output)
    pd.set_option('display.float_format', '{:.2g}'.format)
    print(round(output.all_costs_table,2))

    opex_variables = ['General', 'General (Empty Trips)', 'Carbon','Carbon (Empty Trips)','Time value', 'Transfer']
    investment_variables = ['Edge', 'Node', 'Upgrade','Charge']
    plot_costs(output,which_costs=opex_variables,ylabel='Annual costs (GNOK)',filename="opex")
    plot_costs(output,investment_variables,'Investment costs (GNOK)',"investment")

    #---------------------------------------------------------#
    #       EMISSIONS 
    #---------------------------------------------------------#

    if single_time_period is None:
        def calculate_emissions(output,base_data,domestic=True):
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

        output = calculate_emissions(output,base_data,domestic=False)
        output_domestic = calculate_emissions(output,base_data,domestic=True)
        print('----------------')
        print('domestic emissions:')
        print(output_domestic.emission_stats)
        print(output.emission_stats)

        #I 2021 var de samlede utslippene fra transport 16,2 millioner tonn CO2-ekvivalenter, 8M tonnes er freight transport
        # https://miljostatus.miljodirektoratet.no/tema/klima/norske-utslipp-av-klimagasser/klimagassutslipp-fra-transport/

        #We are off with a factor 100!

        def plot_emission_results(output,base_data):

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
            
        plot_emission_results(output,base_data)

    #---------------------------------------------------------#
    #       MODE MIX
    #---------------------------------------------------------#

    def plot_mode_mixes(TranspArbAvgScen, base_data,absolute_transp_work=True, analysis_type="All transport"):  #result data = TranspArbAvgScen
        
        #https://matplotlib.org/stable/gallery/color/named_colors.html
        color_dict = {'Diesel':                 'firebrick', 
                        'Battery':              'mediumseagreen',
                        'Catenary':              'darkolivegreen',
                        'Ammonia':              'royalblue', 
                        'Hydrogen':             'deepskyblue',
                        'Methanol':              'firebrick',     
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
            #ax.set_title(m + ' - ' + analysis_type)
            ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

            #ax.text(0.5, 0.95*ax.get_ylim()[1], "First stage", fontdict=None)
            #ax.text(1.6, 0.95*ax.get_ylim()[1], "Second stage", fontdict=None)
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            #plt.show()
            fig.savefig(r"Data//figures//"+run_identifier+"_modemix"+m+".png",dpi=300,bbox_inches='tight')
            
    #DO THE FOLLOWING FOR DOMESTIC AND INTERNATIONAL (remove nodes from and to europe and the world)
    def mode_mix_calculations(output,base_data):
        output.x_flow['Distance'] = 0 # in Tonnes KM
        for index, row in output.x_flow.iterrows():
            output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

        output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] /10**9*SCALING_FACTOR_WEIGHT # in GTonnes KM

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
        MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in base_data.S_SCENARIOS]
        all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
        TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)

        TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','TranspArb','TranspArb_std','RelTranspArb','RelTranspArb_std']].groupby(
            ['mode','fuel','time_period'], as_index=False).agg({'TranspArb':'mean','TranspArb_std':'std','RelTranspArb':'mean','RelTranspArb_std':'std'})
        TranspArbAvgScen = TranspArbAvgScen.fillna(0) #in case of a single scenario we get NA's

        plot_mode_mixes(TranspArbAvgScen,base_data,absolute_transp_work=True,analysis_type=analysis_type)

        return output

    output = mode_mix_calculations(output,base_data)
    
    def plot_avg_transportwork(output,base_data):
        
        #Getting the generated paths that are currently used in the model
        generated_paths_file = "generated_paths_2_modes.csv"     # Select the current created paths file
        generated_paths_file = r"Data//SPATIAL//" + generated_paths_file
        generated_paths = pd.read_csv(generated_paths_file)
        generated_paths['paths'] = generated_paths['paths'].apply(ast.literal_eval)

        #Getting the average distance for each leg
        avg_distance = base_data.AVG_DISTANCE
        
        #load the x_flow and h_path dataframes
        x_flow = output.x_flow
        h_path = output.h_path
        
        #merge the x_flow and h_path dataframes on the path column, getting the legs for each path
        h_path_with_path = h_path.join(generated_paths, on='path')
        
        #setting the time periods to visualize
        y = [2023, 2028, 2034]
        
        ######################################################
        ### Functions to calculate the length of the legs ####
        ######################################################
        
        def lookup_values(path_list):
            if path_list == None:
                return None
            dist_legs = [avg_distance.get(path, None) for path in path_list if path[0] != 'World' and path[1] != 'World']
            return sum(dist_legs)
        
        def sea_length(path_list): 
            dist_legs = [avg_distance.get(path, None) for path in path_list if (path[0] != 'World' and path[1] != 'World') and path[2] == 'Sea']
            if sum(dist_legs) == 0:
                return None
            return sum(dist_legs)
        
        def rail_length(path_list): 
            dist_legs = [avg_distance.get(path, None) for path in path_list if (path[0] != 'World' and path[1] != 'World') and path[2] == 'Rail']
            if sum(dist_legs) == 0:
                return None
            return sum(dist_legs)
        
        def road_length(path_list): 
            dist_legs = [avg_distance.get(path, None) for path in path_list if (path[0] != 'World' and path[1] != 'World') and path[2] == 'Road']
            if sum(dist_legs) == 0:
                return None
            return sum(dist_legs)
        
        #divide the path_list into the different modes, labled leg 1 and leg 2
        def leg_split(path_list):
            leg_1 = []
            leg_2 = []
            leg_1_mode = path_list[0][2]
            leg_2_mode = None
            for path in path_list:
                if path[2] != leg_1_mode:
                    leg_2.append(path)
                    leg_2_mode = path[2]
                else:
                    leg_1.append(path)
                    
            if len(leg_2) < 1:
                leg_2 = None
            return leg_1, leg_2, leg_1_mode, leg_2_mode
                
        
        for i in y:
            
            if i == 2034:
                time_period = [2034, 2040, 2050]
            else: 
                time_period = [i]
            
            #filter out the rows where the time period is not 2023 in x_flow and h_path_sum
            h_path_sum = h_path_with_path.copy()[h_path_with_path['time_period'].isin(time_period)]
            x_flow = x_flow[x_flow['time_period']==i]
            
            #Remove the rows where the product is Dry bulk
            #h_path_sum = h_path_sum[h_path_sum['product'] != 'Dry bulk']
            
            h_path_sum = h_path_sum.drop(['variable', 'product'], axis=1)
            
            #Summing the amount of weight for each path in each scenario
            h_path_sum = h_path_sum.groupby(['path','scenario', 'time_period']).agg({'weight': 'sum',  'paths': 'first'})
            
            #Average the weight for each path across the scenarios
            h_path_sum = h_path_sum.groupby(['path', "time_period"]).agg({'weight': 'mean',  'paths': 'first'})
            
            #Average the weight for each path across the scenarios
            h_path_sum = h_path_sum.groupby(['path']).agg({'weight': 'mean',  'paths': 'first'})
                                        
            #Spliting the paths into the different legs and modes      
            h_path_sum[['leg_1', 'leg_2', 'leg_1_mode', 'leg_2_mode']] = h_path_sum['paths'].apply(leg_split).apply(lambda x: pd.Series(x[:4]))
            
            
            # Creating columns for the length of the paths of different modes
            h_path_sum['path_length'] = h_path_sum['paths'].apply(lookup_values)
            h_path_sum['leg_1_length'] = h_path_sum['leg_1'].apply(lookup_values)
            h_path_sum['leg_2_length'] = h_path_sum['leg_2'].apply(lookup_values)
    
            h_path_sum['sea_leg_length'] = h_path_sum['paths'].apply(sea_length) 
            
            h_path_sum['rail_leg_length'] = h_path_sum['paths'].apply(rail_length) 
            
            h_path_sum['road_leg_length'] = h_path_sum['paths'].apply(road_length) 
            
            
            #Creating a dataframe for the total weight for each mode in each interval
            interval_df = pd.DataFrame(columns=['Interval', 'Rail_Weight', 'Road_Weight', 'Sea_Weight'])

            # Define the interval size (100 km in this case)
            interval_size = 100

            # Calculate the total weight for each mode of transport on 1000 km intervals, up to longest path of 3000 km
            for start_interval in range(0, 3000, interval_size):
                
                end_interval = start_interval + interval_size
                             
                # Filter rows within the current interval
                interval_rows_leg_1 = h_path_sum[(h_path_sum['leg_1_length'] >= start_interval) & (h_path_sum['leg_1_length'] < end_interval)]
                interval_rows_leg_2 = h_path_sum[(h_path_sum['leg_2_length'] >= start_interval) & (h_path_sum['leg_2_length'] < end_interval)]
                
                # Calculate total weight for each mode of transport within the interval
                rail_weight = interval_rows_leg_1.loc[interval_rows_leg_1['leg_1_mode'] == 'Rail', 'weight'].sum() + interval_rows_leg_2.loc[interval_rows_leg_2['leg_2_mode'] == 'Rail', 'weight'].sum()
                road_weight = interval_rows_leg_1.loc[interval_rows_leg_1['leg_1_mode'] == 'Road', 'weight'].sum() + interval_rows_leg_2.loc[interval_rows_leg_2['leg_2_mode'] == 'Road', 'weight'].sum()
                sea_weight = interval_rows_leg_1.loc[interval_rows_leg_1['leg_1_mode'] == 'Sea', 'weight'].sum()   + interval_rows_leg_2.loc[interval_rows_leg_2['leg_2_mode'] == 'Sea', 'weight'].sum()           

                 # Create a new dataframe with the results
                new_row = {'Interval': f"{end_interval}", 'Rail_Weight': rail_weight, 'Road_Weight': road_weight, 'Sea_Weight': sea_weight}
                interval_df = pd.concat([interval_df, pd.DataFrame([new_row])], ignore_index=True)
           
            
                   

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plotting bars for each mode
            ax.plot(interval_df['Interval'], interval_df['Rail_Weight'], marker='o', label='Rail')
            ax.plot(interval_df['Interval'], interval_df['Road_Weight'], marker='o', label='Road')
            ax.plot(interval_df['Interval'], interval_df['Sea_Weight'], marker='o', label='Sea')

            # Adding labels and title
            ax.set_xlabel('Interval')
            ax.set_ylabel('Weight')
            ax.set_title('Weight Distribution by Mode and Interval')
            ax.legend()
            
            fig.savefig(r"Data//figures//"+run_identifier+"_avtransportwork_itr5["+str(i)+"].png",dpi=300,bbox_inches='tight')

         
            
            interval_df['Total_Weight'] = interval_df['Rail_Weight'] + interval_df['Road_Weight'] + interval_df['Sea_Weight']

            # Calculate weights as percentages of total weight
            interval_df['Rail_Percentage'] = interval_df['Rail_Weight'] / interval_df['Total_Weight'] * 100
            interval_df['Road_Percentage'] = interval_df['Road_Weight'] / interval_df['Total_Weight'] * 100
            interval_df['Sea_Percentage'] = interval_df['Sea_Weight'] / interval_df['Total_Weight'] * 100

            # Handle instances where total weight is 0
            interval_df.loc[interval_df['Total_Weight'] == 0, ['Rail_Percentage', 'Road_Percentage', 'Sea_Percentage']] = 0

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot lines for each mode
            ax.plot(interval_df['Interval'], interval_df['Rail_Percentage'], marker='o', label='Rail')
            ax.plot(interval_df['Interval'], interval_df['Road_Percentage'], marker='o', label='Road')
            ax.plot(interval_df['Interval'], interval_df['Sea_Percentage'], marker='o', label='Sea')

            # Adding labels and title
            ax.set_xlabel('Interval')
            ax.set_ylabel('Percentage of Total Weight')
            ax.set_title('Weight Distribution by Mode and Interval')
            ax.legend()

            # Display the plot
            fig.savefig(r"Data//figures//"+run_identifier+"_avtransportwork_itr6["+str(i)+"].png",dpi=300,bbox_inches='tight')
            
            fig, ax = plt.subplots()
            
            road_color = 'blue'
            sea_color = 'green'
            rail_color = 'red'
            

            #sort the values in h_path_sum by road_length
            h_path_sum = h_path_sum.sort_values(by=['road_leg_length'])
            
            
            
            
            #Plot the weight against the sea_leg_length
            ax.plot(h_path_sum['road_leg_length'], h_path_sum['weight'], label='Road', color=road_color)
            
            h_path_sum = h_path_sum.sort_values(by=['sea_leg_length'])
            

            
            ax.plot(h_path_sum['sea_leg_length'], h_path_sum['weight'], label='Sea', color=sea_color)
            h_path_sum = h_path_sum.sort_values(by=['rail_leg_length'])


            
            ax.plot(h_path_sum['rail_leg_length'], h_path_sum['weight'], label='Rail', color=rail_color)
            
            # Add legend with color patches
            ax.legend(handles=[
                plt.Line2D([0], [0], color=road_color, label='Road'),
                plt.Line2D([0], [0], color=sea_color, label='Sea'),
                plt.Line2D([0], [0], color=rail_color, label='Rail')
            ])

            ax.legend()

            # Additional plot settings and show the plot if needed
            ax.set_xlabel('Leg Length')
            ax.set_ylabel('Weight')
            ax.set_title('Weight vs Leg Length for Different Modes of Transport')
            
            fig.savefig(r"Data//figures//"+run_identifier+"_avtransportwork_itr4["+str(i)+"].png",dpi=300,bbox_inches='tight')
        
    output= plot_avg_transportwork(output,base_data)   

if __name__ == "__main__":
    
    analyses = [analysis]
    if run_all_analyses == True:
        analyses = list(analyses_info.keys())
    
    for analysis in analyses:
        visualize_results(analyses_type=            analyses_info[analysis][0],
                            scenarios =             analyses_info[analysis][1],
                            noBalancingTrips=       analyses_info[analysis][2],
                            single_time_period=     analyses_info[analysis][3],
                            risk_aversion =         analyses_info[analysis][4],
                            scen_analysis_carbon =  analyses_info[analysis][5],
                            carbon_factor =         analyses_info[analysis][6],
                            )