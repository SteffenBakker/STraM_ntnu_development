### Copy from PostProcessMain.py:
#from Data.settings import *
#from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import pickle

#open old data
analyses_type = 'SP_visualization_ruben' # EV , EEV, 'SP
with open(r'Data\output_data_'+analyses_type, 'rb') as output_file:
    output = pickle.load(output_file)

with open(r'Data\base_data', 'rb') as data_file:
    base_data = pickle.load(data_file)

################   NEW CODE BELOW  ##########################################################


##################
# 1. CREATE MAP
##################

# a. Extract nodes and coordinates

#extract nodes from base_data
N_NODES = base_data.N_NODES

#import norwegian city coordinates
NO_coordinates = pd.read_csv("Data/maps/NO_cities_coordinates.csv")
#extract latitudes and longitudes
lats = [0.0] * len(N_NODES)
lons = [0.0] * len(N_NODES)
for index, row in NO_coordinates.iterrows():
    if row["city"] in N_NODES:
        n_ind = N_NODES.index(row["city"]) #index of this city in list N_NODES
        lats[n_ind] = row["lat"]
        lons[n_ind] = row["lng"]

#Manually define foreign city coordinates
foreign_cities = pd.DataFrame()
foreign_cities["city"] = ["Sør-Sverige", "Nord-Sverige", "Kontinentalsokkelen", "Europa", "Verden"]
foreign_cities["lat"] = [59.33, 63.82, 60, 56.2, 56.5]
foreign_cities["lon"] = [18.06, 20.26, 2,  9, 3]
#add to vectors lats and lons
for index, row in foreign_cities.iterrows():
    if row["city"] in N_NODES:
        n_ind = N_NODES.index(row["city"]) #index of this city in list N_NODES
        lats[n_ind] = row["lat"]
        lons[n_ind] = row["lon"]
#add colors
node_colors = ["black"]*len(N_NODES)
node_colors[-1] = "red"    #Verden                ok
node_colors[-2] = "green"  #Europa                ok
node_colors[-3] = "yellow" #Kontinentalsokkelen   ok
node_colors[-4] = "blue"   #Nord-Sverige          ok
node_colors[-5] = "grey"   #Sør-Sverige           ok

# b. Build a map

from mpl_toolkits.basemap import Basemap #for creating the background map
import matplotlib.pyplot as plt #for plotting on top of the background map

#draw the basic map including country borders
map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgrey', lake_color='aqua')
map.drawcoastlines(linewidth=0.3)
map.drawcountries(linewidth=0.3)

#draw nodes
node_x, node_y = map(lons, lats)
map.scatter(node_x, node_y, color=node_colors, zorder=100)

# c. Process flow data

#Desired input: dataframe with:
    #from, to, flow (aka weight)

df_flow = pd.DataFrame()
arcs = []
flows = []

for index, row in output.x_flow.iterrows():
    if row["scenario"] == "MMM" and row["time_period"] == 2050:
        cur_arc = (row["from"], row["to"])
        if cur_arc not in arcs: #new arc
            arcs.append(cur_arc) 
            flows.append(0.0)
        cur_arc_ind = arcs.index(cur_arc)
        flows[cur_arc_ind] += row["weight"]

#put everything in a dataframe
df_flow["arc"] = arcs
df_flow["orig"] = [""]*len(arcs)
df_flow["dest"] = [""]*len(arcs)
df_flow["flow"] = flows
for i in range(len(df_flow)):
    df_flow.orig[i] = str(df_flow.arc[i][0])
    df_flow.dest[i] = str(df_flow.arc[i][1])


# d. Plot flow in the map

import matplotlib.patches as patches

#arrow settings
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")

for index, row in df_flow.iterrows():
    cur_orig = row["orig"]
    cur_dest = row["dest"]
    cur_orig_index = N_NODES.index(cur_orig)
    cur_dest_index = N_NODES.index(cur_dest)
    new_arc = patches.FancyArrowPatch(
        (node_x[cur_orig_index], node_y[cur_orig_index]), 
        (node_x[cur_dest_index], node_y[cur_dest_index]), 
        connectionstyle="arc3,rad=.2",
        **kw
        )
    plt.gca().add_patch(new_arc)


# d. Show and save the figure
plt.gcf().set_size_inches(8.5,10.5, forward=True)

plt.show()









########## old stuff



"""
a1 = patches.FancyArrowPatch((node_x[0], node_y[0]), (node_x[1], node_y[1]), **kw)
a2 = patches.FancyArrowPatch((node_x[2], node_y[2]), (node_x[1], node_y[1]), connectionstyle="arc3,rad=.5", **kw)
a3 = patches.FancyArrowPatch((node_x[1], node_y[1]), (node_x[2], node_y[2]), connectionstyle="arc3,rad=-.2", **kw)

for a in [a1, a2, a3]:
    plt.gca().add_patch(a)
"""
