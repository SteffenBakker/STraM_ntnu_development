"""Plot the network (edges) on the map"""


import warnings
warnings.filterwarnings("ignore")

# IMPORTS

import numpy as np
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap #for creating the background map
import matplotlib.pyplot as plt #for plotting on top of the background map
import matplotlib.patches as patches #import library for fancy arrows/edges
import copy

# load base_data (which includes the network)
with open(r'Data\base_data', 'rb') as data_file:
    base_data = pickle.load(data_file)


    # USER INPUT
    show_fig = True
    save_fig = False


    ####################################
    # a. Extract nodes and coordinates

    #extract nodes from base_data
    N_NODES = base_data.N_NODES

    #import norwegian city coordinates
    NO_coordinates = pd.read_csv("Data/NO_cities_coordinates.csv")
    #extract latitudes and longitudes
    lats = [0.0] * len(N_NODES)
    lons = [0.0] * len(N_NODES)
    for index, row in NO_coordinates.iterrows():
        if row["city"] in N_NODES:
            n_ind = N_NODES.index(row["city"]) #index of this city in list N_NODES
            lats[n_ind] = row["lat"]
            lons[n_ind] = row["lng"]

    #Manually define foreign city coordinates (HARDCODED)
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
    #add colors (for checking and perhaps plotting)
    node_colors = ["black"]*len(N_NODES)     


    ####################
    # b. Build a map

    #draw the basic map including country borders
    map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='aeqd', lat_0=63.4, lon_0=10.4) # Azimuthal Equidistant Projection
    # map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0) # mercator projection
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='lightgrey', lake_color='aqua')
    map.drawcoastlines(linewidth=0.2)
    map.drawcountries(linewidth=0.2)

    #draw nodes on the map
    node_x, node_y = map(lons, lats)
    map.scatter(node_x, node_y, color=node_colors, zorder=100)

    # draw labels on the map
    #                0  1   2  3    4   5    6    7   8    9    10  11    12   13  14  
    node_x_offset = [2, 2,  2, -18, 2,  -13, -12, 2,  -31, -27, 3,   2,   3,   2,  2]
    node_y_offset = [3, -3, 3, 3,   -5, -6,  3,   -6, -6,  -6,  -4,  -5,  -4,  2,  0]
    node_labels = copy.deepcopy(N_NODES)
    node_labels[1] = " North\nSweden"
    node_labels[5] = "World"
    node_labels[6] = "Continental\n   shelf"
    node_labels[10] = " South\nSweden"
    node_labels[11] = "Europe"
    
    for i in range(len(N_NODES)):
        plt.annotate(node_labels[i], (node_x[i] + 10000*node_x_offset[i], node_y[i] + 10000*node_y_offset[i]), zorder = 1000)
    

    ##########################
    # c. Plot edges in the map

    #arrow settings
    line_width = 1.2
    base_curvature = 0.2
    #arrow settings for the different modes
    mode_color_dict = {"Road":"dimgray", "Sea":"blue", "Rail":"magenta", "Total":"black"}
    #mode_linestyle_dict = {"Road":"-", "Sea":"--", "Rail":(0, (1, 5)), "Total":"-"}
    mode_linestyle_dict = {"Road":"-", "Sea":"-", "Rail":"-", "Total":"-"}
    curvature_fact_dict = {"Road":0, "Sea":-2, "Rail":+1, "Total":0}
    zorder_dict = {"Road":30, "Sea":20, "Rail":40, "Total":20}

    nodes_sea_order = ["Nord-Sverige", "Sør-Sverige", "Hamar", "Oslo", "Skien", "Kristiansand", "Stavanger", 
                            "Bergen", "Ålesund", "Trondheim", "Bodø", "Tromsø", "Europa", "Verden", "Kontinentalsokkelen"] #HARDCODED


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
            cur_orig_index = N_NODES.index(cur_orig)
            cur_dest_index = N_NODES.index(cur_dest)
            
            # get plotting options for current mode
            curvature_factor = curvature_fact_dict[cur_mode]
            cur_color = mode_color_dict[cur_mode]

            # construct new edge
            new_edge = patches.FancyArrowPatch(
                            (node_x[cur_orig_index], node_y[cur_orig_index]),  #origin coordinates
                            (node_x[cur_dest_index], node_y[cur_dest_index]),  #destination coordinates
                            connectionstyle=f"arc3,rad={base_curvature * curvature_factor}", #curvature of the edge
                            linewidth = line_width,
                            linestyle=mode_linestyle_dict[cur_mode],
                            color=cur_color,
                            zorder = zorder_dict[cur_mode]
                            )    
            # add edge to plot
            plt.gca().add_patch(new_edge) 


    ###############################
    # d. Show and save the figure

    #set size
    scale = 1.3
    plot_width = 5 #in inches
    plot_height = scale * plot_width
    plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGHT SIZE
    #save figure
    if save_fig:
        filename = f"Plots/edge_plot.png"
        plt.savefig(filename)
    #show figure
    if show_fig:
        for i in range(len(node_labels)):
            print(i, ": ", node_labels[i])
        plt.show()    

    