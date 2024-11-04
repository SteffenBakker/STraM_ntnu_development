"""Plot the network (edges) on the map"""

from Data.settings import *
from _plot_base_map import plot_base_map_start
import warnings
warnings.filterwarnings("ignore")

# IMPORTS

import numpy as np
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap #for creating the background map
import matplotlib.pyplot as plt #for plotting on top of the background map
import matplotlib.patches as patches #import library for fancy arrows/edges
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
import copy


emission_cap = False
analyses_type="SP"
scenarios="FuelScen"  # FuelScen, FuelDetScen, AllScen, 4Scen, 9Scen  
carbon_fee = "base" #"high", intermediate

run_identifier = scenarios+"_carbontax"+carbon_fee
if emission_cap:
    run_identifier = run_identifier + "_emissioncap"

with open(r'Data//Output//'+run_identifier+'_basedata.pickle', 'rb') as output_file:
    base_data = pickle.load(output_file)


# USER INPUT
show_fig = True
save_fig = True




####################
# b. Build a map

fig, ax, mapp, node_xy_offset, coordinate_mapping, node_x, node_y = plot_base_map_start(base_data)

# draw labels on the map

node_labels = {i:i for i in base_data.N_NODES}
# translate labels
translate_dict = {"JohanSverdrupPlatform":"Cont. Shelf",   #use \n for new line
                "Hamburg":" Hamburg/Europe", 
                }
for key,value in translate_dict.items():
    node_labels[key] = value

for i in base_data.N_NODES:
    plt.annotate(node_labels[i], (coordinate_mapping[i][0] + 10000*node_xy_offset[i][0], 
                     coordinate_mapping[i][1] + 10000*node_xy_offset[i][1]), zorder = 1000)  #10000*offset


##########################
# c. Plot edges in the map

#arrow settings
line_width = 2.5
base_curvature = 0.2
#arrow settings for the different modes
mode_color_dict = {"Road": rgb_constructor(207, 65, 84), #"violet", 
                   "Sea":rgb_constructor(47, 85, 151), #"blue", 
                   "Rail":rgb_constructor(55, 0, 30), #"saddlebrown", 
                   "total":"black"}
#mode_linestyle_dict = {"Road":"-", "Sea":"--", "Rail":(0, (1, 5)), "Total":"-"}
mode_linestyle_dict = {"Road":"-", "Sea":"-", "Rail":"-", "Total":"-"}
curvature_fact_dict = {"Road":0, "Sea":-2, "Rail":+1, "Total":0}
zorder_dict = {"Road":20, "Sea":30, "Rail":40, "Total":20}

nodes_sea_order = ["Umeå", "Stockholm", "Hamar", "Oslo", "Skien", "Kristiansand", "Stavanger", 
                        "Bergen", "Førde","Ålesund", "Trondheim", "Bodø", "Tromsø","Narvik", "Alta",
                        "Hamburg", "World", "JohanSverdrupPlatform"] #HARDCODED


unique_edges = []



# loop over all edges to plot them
for (i,j,m,r) in base_data.E_EDGES:
    # put edge in right order
    i_sea_index = nodes_sea_order.index(i)
    j_sea_index = nodes_sea_order.index(j)
    cur_orig = ""   # init
    cur_dest = ""   # init
    if i_sea_index < j_sea_index:
        cur_orig = i
        cur_dest = j
    else:
        cur_orig = j
        cur_dest = i
    cur_mode = m
    if ((cur_orig, cur_dest, cur_mode) not in unique_edges):  # unobserved edge
        
        # add edge to list
        unique_edges.append((cur_orig, cur_dest, cur_mode)) 
            
        # extract more edge information
        cur_orig_index = base_data.N_NODES.index(cur_orig)
        cur_dest_index = base_data.N_NODES.index(cur_dest)
        
        # get plotting options for current mode
        curvature_factor = curvature_fact_dict[cur_mode]
        cur_color = mode_color_dict[cur_mode]

        # construct new edge
        line_stl = mode_linestyle_dict[cur_mode]
        if (cur_mode == "Rail") and (cur_orig, cur_dest) in [("Bodø","Narvik"),("Narvik","Bodø"),("Narvik","Tromsø"),("Tromsø","Narvik")]:
            line_stl = "dotted"#":"

        new_edge = patches.FancyArrowPatch(
                        (node_x[cur_orig_index], node_y[cur_orig_index]),  #origin coordinates
                        (node_x[cur_dest_index], node_y[cur_dest_index]),  #destination coordinates
                        connectionstyle=f"arc3,rad={base_curvature * curvature_factor}", #curvature of the edge
                        linewidth = line_width,
                        arrowstyle='-',
                        linestyle=line_stl, #requires arrowstyle to be defined in order to work
                        color=cur_color,
                        zorder = zorder_dict[cur_mode]
                        )    
        # add edge to plot
        plt.gca().add_patch(new_edge) 

        if (cur_orig in ['Hamar','Trondheim']) and (cur_dest in ['Hamar','Trondheim']) and (m=='Rail'):
            new_edge = patches.FancyArrowPatch(
                        (node_x[cur_orig_index], node_y[cur_orig_index]),  #origin coordinates
                        (node_x[cur_dest_index], node_y[cur_dest_index]),  #destination coordinates
                        connectionstyle=f"arc3,rad={base_curvature * 2.2}", #curvature of the edge
                        linewidth = line_width,
                        linestyle=mode_linestyle_dict[cur_mode],
                        color=cur_color,
                        zorder = zorder_dict[cur_mode]
                        ) 
            plt.gca().add_patch(new_edge) 

# add legend

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=mode_color_dict["Road"], lw=3),
                Line2D([0], [0], color=mode_color_dict["Sea"], lw=3),
                Line2D([0], [0], color=mode_color_dict["Rail"], lw=3)]
plt.legend(custom_lines, ['Road', 'Sea', 'Rail'])



###############################
# d. Show and save the figure

#set size
scale = 1.2 #
plot_width = 5 #in inches
plot_height = scale * plot_width
plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGHT SIZE
#save figure
if save_fig:
    filename = f"Data/Plots/edge_plot.pdf"
    plt.savefig(filename,bbox_inches='tight')
#show figure
if show_fig:
    plt.show()

