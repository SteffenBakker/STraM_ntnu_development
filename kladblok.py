from Data.settings import *
import numpy as np
import pandas as pd
import pickle

analyses_type="SP"
scenarios="4Scen"

run_identifier = analyses_type+'_'+scenarios

with open(r'Data//output//'+run_identifier+'.pickle', 'rb') as output_file:
    output = pickle.load(output_file)
with open(r'Data//base_data//'+scenarios+'.pickle', 'rb') as data_file:
    base_data = pickle.load(data_file)



#------------------------------------------

#Ingen veitransport fra Trondheim til Bodø (merkelig nok).
output.x_flow[(output.x_flow['from']=='Trondheim')&
              (output.x_flow['to']=='Bodø')&
              (output.x_flow['time_period']==2023)&
              (output.x_flow['scenario']=='BBB')]

#-----------------------------------------

#DICTIONARY_TO_PD_DATAFRAME!!

def dict_to_pd_df(dict,col_names):
    data = [[*keys, value] for keys, value in dict.items() for key in keys]
    df = pd.DataFrame(data, columns=col_names)
    return df

DEMAND = dict_to_pd_df(base_data.D_DEMAND, ["from","to","product","time_period","demand"])  
DEMAND["demand"]=DEMAND["demand"]/10**6*SCALING_FACTOR_WEIGHT # in MTonnes KM

DEMAND["product"].value_counts() #ONLY CONTAINER AND DRY BULK CURRENTLY!! No neo bulk and 