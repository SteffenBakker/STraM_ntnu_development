import pandas as pd
import numpy as np

zone_aggregation_name = 'STRAM'  #Alternatively, NTP

##################
#### Zones  ######
##################

zones = pd.read_csv(r'zones.csv', sep=';')
zone_mapping = dict(zip(zones.NGM_zone_nr, zones[zone_aggregation_name+"_zone_nr"]))

###############
# Commodities #
###############

commodities = pd.read_csv(r'commodities.csv', sep=';')
comm_key_mapping = dict(zip(commodities.Comm_nr, commodities.Comm_aggr_nr))
commodities_list = list(comm_key_mapping.keys())  #39 commodities

########################
#### PWC MATRICES ######
########################

#read all OD matrix files into a single object
years = [2018,2020,2025,2030,2040,2050]
for i in commodities_list:
    for year in years:
        pwc_temp = pd.read_csv(r'PWC/'+str(year)+'/pwc'+str(i)+'.dat', delim_whitespace=True, header=None, names=['from', 'to', 'type', 'amount_tons'])
        pwc_temp['commodity'] = i
        pwc_temp['year'] = year
        if i == commodities_list[0] and year == years[0]:
            pwc = pwc_temp
        else:
            pwc = pd.concat([pwc,pwc_temp])

#then, calculate the aggregated data
for j in ['from', 'to']:
    pwc[j+'_aggr_zone'] = pd.to_numeric(pwc[j].map(zone_mapping), downcast='float')
pwc['commodity_aggr'] = pwc['commodity'].map(comm_key_mapping)

#finally, aggregate the data
pwc_aggr = pwc.drop(['from', 'to', 'type','commodity'],axis=1)
pwc_aggr = pwc_aggr.groupby(['from_aggr_zone', 'to_aggr_zone','commodity_aggr','year'], as_index=False).sum()

#########################
# Save data
#########################

if True:
    pwc_aggr.to_csv(r'pwc_aggr.csv')
#pwc_aggr ----------------   pwc data (demand for transport)