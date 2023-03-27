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
from matplotlib.patches import Circle



# load base_data (which includes the network)
with open(r'Data\\base_data\\9Scen.pickle', 'rb') as data_file:
    base_data = pickle.load(data_file)

    # USER INPUT
    show_fig = True
    save_fig = True

    for d in [0,1]:  

        fig = plt.figure(figsize=(6,3))
        ax = plt.axes([0,0,1,1])
        
        ####################################
        # 0. Extract demand data


        demand = {i:0 for i in base_data.N_NODES}

        for key,val in base_data.D_DEMAND.items():
            (i,j,p,t) = key
            dd = key[d] #0 = from,  1= to
            if t == base_data.T_TIME_PERIODS[0]:    
                demand[dd] += val

        for key,val in demand.items():
            demand[key] = round(val,2)
        

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
        map.drawmapboundary(fill_color='paleturquoise')
        map.fillcontinents(color='lightgrey', lake_color='paleturquoise')
        map.drawcoastlines(linewidth=0.2)
        map.drawcountries(linewidth=0.2)

        #draw nodes on the map
        node_x, node_y = map(lons, lats)
        map.scatter(node_x, node_y, color=node_colors, zorder=100, s=10)


        ####################
        # c. Plot demand
        
        #rad =0.2*(max(node_x)-(min(node_x))) # demand[N_NODES[i]]
        scale = 3500
        clr = "green"
        if d == 1:
            clr = "red"

        for i in range(len(N_NODES)):
            x,y = map(lons[i],lats[i])
            circle = plt.Circle(xy=(x,y), radius=scale*demand[N_NODES[i]], 
                                fill=True,alpha=1, color=clr,
                                edgecolor="black") #
            #plt.gca().add_patch(circle)
            ax.add_patch(circle)


        ###############################
        # d. Show and save the figure

        #set size
        scale = 1.2 #
        plot_width = 5 #in inches
        plot_height = scale * plot_width
        plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGHT SIZE
        #save figure
        if save_fig:
            if d == 0:
                filename = f"Data/Plots/demand_plot_from.png"
            elif d == 1:
                filename = f"Data/Plots/demand_plot_to.png"
            plt.savefig(filename,bbox_inches='tight')
        #show figure
        if show_fig:
            #for i in range(len(node_labels)):
            #    print(i, ": ", node_labels[i]) 
            plt.show()

        