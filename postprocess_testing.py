

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


#DO THE FOLLOWING FOR DOMESTIC AND INTERNATIONAL (remove nodes from and to europe and the world)
output.x_flow['Distance'] = 0 # in Tonnes KM
for index, row in output.x_flow.iterrows():
        output.x_flow.at[index,'Distance'] = base_data.AVG_DISTANCE[(row['from'],row['to'],row['mode'],row['route'])]

output.x_flow['TransportArbeid'] = output.x_flow['Distance']*output.x_flow['weight'] # in Tonnes KM

TranspArb = output.x_flow[['mode','fuel','time_period','TransportArbeid','scenario']].groupby(['mode','fuel','time_period','scenario']).agg(
                                                                                                                {'TransportArbeid':'sum'})
TotalTranspArb = TranspArb.groupby(['time_period','scenario']).agg({'TransportArbeid':'sum'})
TotalTranspArb = TotalTranspArb.rename(columns={"TransportArbeid": "TransportArbeidTotal"})

len(TranspArb)
len(pd.merge(TranspArb,TotalTranspArb,how='outer',on=['time_period','scenario']))











#data is base data, dataset is output from model
def plot_figures(output,base_data):
    
    for s in output.scenarios:
        dataset_scen = 2 # to do
        fuel_list = []
        fuel_list1 = []
        for m in base_data.M_MODES:
            for f in base_data.FM_FUEL[m]:
                for t in base_data.T_TIME_PERIODS:
                    #Dataset for domestic and for international
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
