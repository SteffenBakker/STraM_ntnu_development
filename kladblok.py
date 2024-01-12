from Data.settings import *
import numpy as np
import pandas as pd
import pickle

emission_cap = False
analyses_type="SP"
scenarios="FuelDetScen"  # FuelScen, FuelDetScen, AllScen, 4Scen, 9Scen  
carbon_fee = "base" #"high", intermediate

run_identifier = scenarios+"_carbontax"+carbon_fee
if emission_cap:
    run_identifier = run_identifier + "_emissioncap"
run_identifier2 = run_identifier+"_"+analyses_type

with open(r'Data//Output//'+run_identifier+'_basedata.pickle', 'rb') as output_file:
    base_data = pickle.load(output_file)
with open(r'Data//Output//'+run_identifier2+'_results.pickle', 'rb') as data_file:
    output = pickle.load(data_file)


#DICTIONARY_TO_PD_DATAFRAME!!
def dict_to_pd_df(dict,col_names):
    data = [[*keys, value] for keys, value in dict.items()]
    df = pd.DataFrame(data, columns=col_names)
    return df


if True:
    #output.all_variables
    #output.epsilon_edge
    #output.upsilon_upgrade

    print(output.x_flow[(output.x_flow['mode']=='Road') & 
                  (output.x_flow['from'].isin(['Trondheim','Bodø'])) & 
                  (output.x_flow['to'].isin(['Trondheim','Bodø'])) &
                  (output.x_flow['time_period'].isin([2023]))
                  ])

if False:

    #------------------------------------------
    #Demand and Path Analysis#
    #------------------------------------------

    edge1 = ["Bodø","Trondheim"]
    edge2 = ["Umeå","Trondheim"]

    for edge in [edge1,edge2]:

        #Ingen veitransport fra Trondheim til Bodø (merkelig nok).
        output.x_flow[(output.x_flow['from']=='Tromsø')&
                    (output.x_flow['to']==edge1[0])&
                    (output.x_flow['time_period']==2023)&
                    (output.x_flow['scenario']=='BBB')]

    #Now check if there is Demand

    DEMAND = dict_to_pd_df(base_data.D_DEMAND, ["from","to","product","time_period","demand"])  
    DEMAND = DEMAND[(DEMAND["time_period"] ==2023)]

    DEMAND[(DEMAND["from"].isin(edge1)) & (DEMAND["to"].isin(edge1))]

    DEMAND[(DEMAND["from"].isin(edge2)) & (DEMAND["to"].isin(edge2))]

    # YESS, for both cases

    # -> Then check if there is a path

    edge3 = ["Oslo","Hamar"]
    edge = edge2
    for index,path in base_data.K_PATH_DICT.items():
        num_legs = len(path)
        for leg in range(num_legs):
            (i,j,m,r) = path[leg]
            if (i in edge) and (j in edge) and m=='Road':
                print(path)

    PATHS = pd.DataFrame(base_data.K_PATH_DICT.items(), columns=['index', 'Path'])


    #NO PATH!

    #but distances are ok
    DIST = dict_to_pd_df(base_data.AVG_DISTANCE,["from","to","mode","route","dist"])
    DIST = DIST.sort_values(by=['from'])

    #-----------------------------------------



    DEMAND = dict_to_pd_df(base_data.D_DEMAND, ["from","to","product","time_period","demand"])  
    DEMAND["demand"]=DEMAND["demand"]/10**6*SCALING_FACTOR_WEIGHT # in MTonnes KM

    DEMAND["product"].value_counts() #ONLY CONTAINER AND DRY BULK CURRENTLY!! No neo bulk and 