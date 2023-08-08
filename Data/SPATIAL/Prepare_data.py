
import pandas as pd
import numpy as np
import json
import os


#os.chdir('C://Users//steffejb//OneDrive - NTNU//Work//Projects//NTRANS (FME)//UC3 MultiModal Transport//Model//FreightTransportModel')


def extract_nth_letter(string, n):
    if n <= len(string):
        output = string[n - 1]
    else:
        output = None
    return output


############################
#### Zones NGM -> NTP ######
############################

zonal_aggregation_keys = pd.read_csv(r'Old\\NTP-aggregering.csv', sep=';')
zonal_aggregation_keys = zonal_aggregation_keys.astype({'NTP_zone_nr': 'str'})
#NGM_zone_name_mapping = dict(zip(zonal_aggregation_keys.NGM_zone_nr, zonal_aggregation_keys.NGM_zone_name))
NTP_zone_mapping_orig = dict(zip(zonal_aggregation_keys.NGM_zone_nr, zonal_aggregation_keys.NTP_zone_nr))

Zones_NGM_ALL = pd.read_csv(r'Old/ZONES_NGM_ALL.csv', sep=';')
Zones_NGM_ALL['Zone in NTP'] = Zones_NGM_ALL['Connected zone'].map(NTP_zone_mapping_orig)
NTP_zone_mapping = dict(zip(Zones_NGM_ALL['Zone in model'], Zones_NGM_ALL['Zone in NTP']))


######################
# Zones NTP -> Fylke #
######################

zones = pd.read_csv(r'Old/zonal_aggregation_steffen.csv', sep=';')
zones_mapping = dict(zip(zones.NTP_zone_nr, zones.Stef_zone_nr))
zones_fylke_aggr = zones[['Stef_zone_nr', 'Stef_zone_name', 'google_maps', 'Lat2', 'Long2']].drop_duplicates()
zones_fylke_aggr = zones_fylke_aggr.rename(columns={'Lat2': 'Lat', 'Long2': 'Long'})

# get the coordinates, this takes some time. So only do once.

'''
import requests
import urllib.parse
        
def get_coordinates(google_maps_name):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(google_maps_name) +'?format=json'
    response = requests.get(url).json()
    return [response[0]["lat"], response[0]["lon"]]

for index, row in zones.iterrows():
    name = row['Google_maps_position']
    output = get_coordinates(name)
    zones.loc[index,['Lat']] = output[0]
    zones.loc[index,['Long']] = output[1]

zones.to_csv(r'Data/MoZEES/Zones2.csv', sep=';',index=False,encoding='utf-8')

'''

#########
# Commodities
#########

commodities = pd.read_csv(r'Data/MoZEES/Commodities.csv', sep=';')
comm_key_mapping = dict(zip(commodities.Comm_nr, commodities.Comm_aggr_nr))
comm_aggr_name_mapping = commodities[['Comm_aggr_nr', 'Comm_aggr_name']].drop_duplicates()
comm_aggr_name_mapping = dict(zip(comm_aggr_name_mapping.Comm_aggr_nr, comm_aggr_name_mapping.Comm_aggr_name))


########################
#### PWC MATRICES ######
########################

years = [2018,2020,2025,2030,2040,2050]
commodities_list = list(comm_key_mapping.keys())
for i in commodities_list:
    for year in years:
        pwc_temp = pd.read_csv(r'Data/MoZEES/'+str(year)+'/pwc'+str(i)+'.dat', delim_whitespace=True, header=None, names=['from', 'to', 'type', 'amount_tons'])
        pwc_temp['commodity'] = i
        pwc_temp['year'] = year
        if i == commodities_list[0] and year == years[0]:
            pwc = pwc_temp
        else:
            pwc = pwc.append(pwc_temp)

for j in ['from', 'to']:
    pwc[j+'_NTP_zone'] = pwc[j].map(NTP_zone_mapping)
    pwc[j+'_NTP_zone'] = pd.to_numeric(pwc[j+'_NTP_zone'], downcast='float')
    pwc[j+'_fylke_zone'] = pwc[j+'_NTP_zone'].map(zones_mapping)
pwc['commodity_aggr'] = pwc['commodity'].map(comm_key_mapping)

pwc_aggr = pwc.drop(['from', 'to', 'type','commodity','from_NTP_zone', 'to_NTP_zone'],axis=1)
pwc_aggr = pwc_aggr.groupby(['from_fylke_zone', 'to_fylke_zone','commodity_aggr','year'], as_index=False).sum()

#pwc_aggr['to_fylke_zone'].unique() #THIS iS OK!!



#########################
# Required data for the next part #
#########################

if True:
    pwc_aggr.to_csv(r'Data/ModelData/pwc_aggr.csv')
#pwc_aggr ----------------   pwc data (demand for transport)

