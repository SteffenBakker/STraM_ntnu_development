

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

import pickle

#---------------------------------------------------------#

with open(r'Data\output_data', 'rb') as output_file:
        output = pickle.load(output_file)

with open(r'Data\base_data', 'rb') as data_file:
        base_data = pickle.load(data_file)

round(output.all_costs_table,1)

output.plot_costs()
output.emission_results(base_data)
output.z_emission_violation
output.cost_and_investment_table(base_data)
round(output.all_costs_table,1)

#---------------------------------------------------------#


#DO THE FOLLOWING FOR DOMESTIC AND INTERNATIONAL (remove nodes from and to europe and the world)
output.x_flow['Distance'] = 0 # in Tonnes KM
for index, row in output.x_flow.iterrows():
        output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] # in Tonnes KM

subset_x_flow = output.x_flow
if True: #domestic
    subset_x_flow = subset_x_flow[(subset_x_flow['from'].isin(base_data.N_NODES_NORWAY)) & (subset_x_flow['to'].isin( base_data.N_NODES_NORWAY)) ]

TranspArb = subset_x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario'], as_index=False).agg(
                                                                                                                {'TransportArbeid':'sum'})
                                                                                                                
TotalTranspArb = TranspArb.groupby(['time_period','scenario'], as_index=False).agg({'TransportArbeid':'sum'})
TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})

TranspArb = pd.merge(TranspArb,TotalTranspArb,how='left',on=['time_period','scenario'])
TranspArb['RelTranspArb'] = TranspArb['TransportArbeid'] / TranspArb['TransportArbeidTotal']


TranspArb['RelTranspArb_std'] = TranspArb['RelTranspArb']
MFTS = [(m,f,t,s) for (m,f,t) in base_data.MFT for s in base_data.scenario_information.scenario_names]
all_rows = pd.DataFrame(MFTS, columns = ['mode', 'fuel', 'time_period','scenario'])
TranspArb = pd.merge(all_rows,TranspArb,how='left',on=['mode','fuel','time_period','scenario']).fillna(0)


TranspArbAvgScen = TranspArb[['mode','fuel','time_period','scenario','RelTranspArb','RelTranspArb_std']].groupby(['mode','fuel','time_period'], as_index=False).agg({'RelTranspArb':'mean','RelTranspArb_std':'std'})

#(TranspArb[(TranspArb['scenario']=='Low')&(TranspArb['time_period']==2020)]) # THIS WORKS!!
#sum(TranspArbAvgScen[(TranspArbAvgScen['time_period']==2020)]['RelTranspArb']) # THIS WORKS!!



#---------------------------------------------------------#


# PLOTTING

def plot_mode_mixes(result_data,base_data):

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
            subset = result_data[(result_data['mode']==m)&(result_data['fuel']==f)]
            ax.bar(labels, subset['RelTranspArb'].tolist(), width, yerr=subset['RelTranspArb_std'].tolist(), 
                        bottom = bottom,label=f,color=color_dict[m,f])
            bottom = [subset['RelTranspArb'].tolist()[i]+bottom[i] for i in range(len(bottom))]
        ax.set_ylabel('Transport work share (%)')
        ax.set_title(m)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #correct

        plt.show()

plot_mode_mixes(TranspArbAvgScen,base_data)