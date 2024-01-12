"""
The basis of the maps that we use
"""

import warnings
warnings.filterwarnings("ignore")

# IMPORTS

import numpy as np
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap #for creating the background map
import matplotlib.pyplot as plt #for plotting on top of the background map
import matplotlib.patches as patches #import library for fancy arrows/edges
from Data.settings import *


####################################################################

# FUNCTIONS

def plot_base_map_start(base_data):

    ####################################
    # a. Extract nodes and coordinates

    #extract nodes from base_data
    N_NODES = base_data.N_NODES
    lats = base_data.N_LATITUDE_PLOT
    longs = base_data.N_LONGITUDE_PLOT
    node_xy_offset = base_data.N_COORD_OFFSETS

    #add colors (for checking and perhaps plotting)
    node_colors = ["black"]*len(N_NODES)     


    ####################
    # b. Build a map

    # create underlying figure/axis (to get rid of whitespace)
    fig = plt.figure(figsize=(6,3))
    ax = plt.axes([0,0,1,1])

    #draw the basic map including country borders
    mapp = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='aeqd', 
                lat_0=63.4, lon_0=10.4
                ) # Azimuthal Equidistant Projection
    # map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0) # mercator projection
    mapp.drawmapboundary(fill_color= color_map_stram["ocean"])#'paleturquoise')
    mapp.fillcontinents(color=       color_map_stram["land"],           #'lightgrey', 
                    lake_color=  color_map_stram["ocean"]      #'paleturquoise'     
                    )
    mapp.drawcoastlines(linewidth=0.2, color="grey")
    mapp.drawcountries(linewidth=0.2, color="grey")

    #draw nodes on the map
    node_x, node_y = mapp(list(longs.values()), list(lats.values()))
    coordinate_mapping={N_NODES[i]:(node_x[i],node_y[i]) for i in range(len(N_NODES))}
    mapp.scatter(node_x, node_y, color=node_colors, zorder=100)

    #remove the black
    fig.patch.set_facecolor('white')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    return fig, ax, mapp, node_xy_offset, coordinate_mapping, node_x, node_y 