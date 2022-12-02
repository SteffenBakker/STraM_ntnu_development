import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
import os

#from mpl_toolkits.basemap import Basemap

os.chdir("M:\Documents\GitHub\AIM_Norwegian_Freight_Model")

data_dir = "data/maps/NUTS/"

#regions
path_rg = data_dir + "NUTS_RG_01M_2021_3035_LEVL_3.json"
gdf_rg = gpd.read_file(path_rg)

#boundaries
path_bn = data_dir + "NUTS_BN_01M_2021_3035_LEVL_3.json"
gdf_bn = gpd.read_file(path_bn)

#labels
path_lb = data_dir + "NUTS_LB_2021_3035_LEVL_3.json"
gdf_lb = gpd.read_file(path_lb)

#projections: 
# 3857 is mercator projection
# 3035 is lambert azimuthal equal area projection (preserves area)
gdf_rg.crs = "EPSG:3035" 
gdf_bn.crs = "EPSG:3035" 
gdf_lb.crs = "EPSG:3035" 

#Slice the table to keep just data for Germany:
gdf_rg_no = gdf_rg[gdf_rg.CNTR_CODE == "NO"]

# we need to transform the data into the 3857 CRS because of the Contextily base map
gdf_rg_no = gdf_rg_no.to_crs("EPSG:3857")
# plot the German federated states (Bundesländer)
ax = gdf_rg_no.plot(figsize=(20,15), color="lightgray")
# plot borders between the states green
gdf_rg_no.boundary.plot(color="darkgreen", linewidth=0.2, ax=ax)
# add background map by OpenStreetMap
#ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.HOT)





plt.show()
