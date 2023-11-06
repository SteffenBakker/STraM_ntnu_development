import pandas as pd
import numpy as np


##################
#### Zones  ######
##################

zone_aggregation_name = 'STRAM'  #Alternatively, NTP
zone_mapping_data = pd.read_excel(r'Data/SPATIAL/spatial_data.xlsx', sheet_name="zone_mapping")
zone_mapping = dict(zip(zone_mapping_data.NGM_zone_nr, zone_mapping_data[zone_aggregation_name+"_zone_name"]))

###############
# Commodities #
###############

#commodities = pd.read_csv(r'Data/commodities.csv', sep=';')
commodities = pd.read_excel(r'Data/commodities.xlsx', sheet_name="commodities")

comm_key_mapping = dict(zip(commodities.Comm_nr, commodities.product_group_stram))
commodities_list = list(comm_key_mapping.keys())  #39 commodities

########################
#### PWC MATRICES ######
########################

#read all OD matrix files into a single object
years = [2018,2020,2025,2030,2040,2050]  #harcoding, update when getting new data
for i in commodities_list:
    for year in years:
        pwc_temp = pd.read_csv(r'Data/SPATIAL/PWC/'+str(year)+'/pwc'+str(i)+'.dat', delim_whitespace=True, header=None, names=['from', 'to', 'type', 'amount_tons'])
        pwc_temp['commodity'] = i
        pwc_temp['year'] = year
        if i == commodities_list[0] and year == years[0]:
            pwc = pwc_temp
        else:
            pwc = pd.concat([pwc,pwc_temp])

if True:  #Input to Jonas, time value of transport
    pwc_aggregated = pwc.groupby('commodity')['amount_tons'].sum().reset_index()
    print(pwc_aggregated)
    pwc_aggregated.to_csv(r'Data/SPATIAL/demand_per_commodity.csv')

#then, calculate the aggregated data
for j in ['from', 'to']:
    pwc[j+'_aggr_zone'] = pwc[j].map(zone_mapping)    #pd.to_numeric(..., downcast='float')
pwc['product_group'] = pwc['commodity'].map(comm_key_mapping)

#finally, aggregate the data
pwc_aggr = pwc.drop(['from', 'to', 'type','commodity'],axis=1)
pwc_aggr = pwc_aggr.groupby(['from_aggr_zone', 'to_aggr_zone','product_group','year'], as_index=False).sum()
#3.14 GT

pwc_filtered = pwc_aggr[pwc_aggr['from_aggr_zone'] != pwc_aggr['to_aggr_zone']]
pwc_filtered = pwc_filtered.astype({'from_aggr_zone':'string','to_aggr_zone':'string' })
#1.71 GigaTonnes

#########################
# Save data
#########################

if True:
    pwc_filtered.to_csv(r'Data/SPATIAL/demand.csv')
#pwc_aggr ----------------   pwc data (demand for transport)


