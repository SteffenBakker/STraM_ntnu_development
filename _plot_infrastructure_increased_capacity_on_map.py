"""
In this file we plot infrastructure investments on a map of Norway

(much code is copied from plot_flow_on_map.py)
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
from _plot_base_map import plot_base_map_start

def process_epsilon_edges(epsilon_edge, sel_time_period, sel_scenario, cumulative):
    
    
    epsilon_edge = epsilon_edge.copy()
    
    epsilon_edge = epsilon_edge[epsilon_edge['scenario']==sel_scenario]
    
    if cumulative:
        epsilon_edge = epsilon_edge[epsilon_edge['time_period']<=sel_time_period]
        epsilon_edge = epsilon_edge.groupby(['from', 'to', 'scenario']).sum().reset_index()
    else:
        epsilon_edge = epsilon_edge[epsilon_edge['time_period'] == sel_time_period]
    
    return epsilon_edge

def plot_edge_expansion(df_edges, base_data, show_fig, save_fig, filename):
    """
    Create a plot on the map of Norway with infrastructure investments

    INPUT
    def_edges:      dataframe with binary expansion variables for edges
    base_data:      base model data (only used to extract N_NODES)
    show_fig:       indicate whether to show the figure
    save_fig:       indicate whether to save the figure (filename is determined automatically)
    
    OUTPUT
    figure that is shown and/or saved to disk if requested
    """
    
    
    fig = plt.figure(figsize=(6,3))
    ax = plt.axes([0,0,1,1])

    
    ####################
    # b. Build a map

    fig, ax, mapp, node_xy_offset, coordinate_mapping, node_x, node_y = plot_base_map_start(base_data)

    for index, row in df_edges.iterrows():
        From_node = row['from']
        To_node = row['to']
        weight = row['weight']
        
        if weight > 0:
            #get coordinates
            x1, y1 = coordinate_mapping[From_node]
            x2, y2 = coordinate_mapping[To_node]
            #plot arrow
            ax.arrow(x1, y1, x2-x1, y2-y1, width=0.1, color='blue', length_includes_head=True, head_width=0.3, head_length=0.3, zorder=100)
    
    ###############################
    # d. Show and save the figure

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=3),]
    plt.legend(custom_lines, ['Expanded infrstructure'])

    #set size
    scale = 1.3
    plot_width = 5 #in inches
    plot_height = scale * plot_width
    plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGH TSIZE
    #save figure
    if save_fig:
        plt.savefig(filename, bbox_inches="tight")
    #show figure
    if show_fig:
        plt.show()


    
    

def process_and_plot_expansion_edge(output, base_data, sel_time_period, sel_scenario="BB", cumulative=False, show_fig=True, save_fig=False):
    # process data 
    print("Processing terminal infrasructure investments...")
    df_edges = process_epsilon_edges(output.epsilon_edge, sel_time_period, sel_scenario, cumulative)

    # make plot
    print("Making plot...")
    filename = f"Data/Output/Plots/EdgeExpansion/edge_expansion_plot_{sel_time_period}_{sel_scenario}_{cumulative}.png"
    plot_edge_expansion(df_edges, base_data, show_fig, save_fig, filename)
    
def process_and_plot_expansion_node(output, base_data, sel_time_period, sel_scenario="BB", cumulative=False, show_fig=True, save_fig=False):
    # process data 
    print("Processing terminal infrasructure investments...")
    df_nodes = process_nu_nodes(output.nu_node, sel_time_period, sel_scenario, cumulative)

    for mode in ["Sea", "Rail"]:
        # make plot
        print("Making plot...")
        filename = f"Data/Output/Plots/NodeExpansion/{mode}/node_expansion_plot_{sel_time_period}_{sel_scenario}_{cumulative}.png"
        plot_node_expansion(df_nodes, base_data, mode, show_fig, save_fig, filename)


def process_nu_nodes(nu_node, sel_time_period, sel_scenario, cumulative):
    nu_node = nu_node.copy()
    
    nu_node = nu_node[nu_node['scenario']==sel_scenario]
    
    if cumulative:
        nu_node = nu_node[nu_node['time_period']<=sel_time_period]
        nu_node = nu_node.groupby(['from', 'scenario', 'mode']).sum().reset_index()
    else:
        nu_node = nu_node[nu_node['time_period'] == sel_time_period]
    
    return nu_node

def plot_node_expansion(df_nodes, base_data, mode, show_fig, save_fig, filename):
    
    df_nodes = df_nodes.copy()
    df_nodes = df_nodes[df_nodes['mode']==mode]
    
    fig, ax, mapp, node_xy_offset, coordinate_mapping, node_x, node_y = plot_base_map_start(base_data)
    
    #if the weight is larger than 0, plot the node
    for index, row in df_nodes.iterrows():
        node = row['from']
        weight = row['weight']
        if weight > 0:
            x, y = coordinate_mapping[node]
            ax.scatter(x, y, s=100, c='red', zorder=100)
            
    #show and save the figure
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=3),]
    plt.legend(custom_lines, [f'Expanded Nodes for {mode}'])

    #set size
    scale = 1.3
    plot_width = 5 #in inches
    plot_height = scale * plot_width
    plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGH TSIZE
    #save figure
    if save_fig:
        plt.savefig(filename, bbox_inches="tight")
    #show figure
    if show_fig:
        plt.show()


#####################################################################################

# RUN ANALYSIS

# Read model output
analyses_type = 'SP' # EV , EEV, 'SP
scenario_type = "FuelScen" # 4Scen
emission_cap = False
carbon_fee = 'base' #high, intermediate, base

run_identifier = scenario_type+"_carbontax"+carbon_fee
if emission_cap:
    run_identifier = run_identifier + "_emissioncap"
run_identifier2 = run_identifier+"_"+analyses_type

with open(r'Data//Output//'+run_identifier+'_basedata.pickle', 'rb') as output_file:
    base_data = pickle.load(output_file)
with open(r'Data//Output//'+run_identifier2+'_results.pickle', 'rb') as data_file:
    output = pickle.load(data_file)

# plot charging infra for multiple years
sel_scenario = "BB"  #Choose "BB" or "PB", "OO"

if True:
    #for t in [2023, 2028, 2034, 2040, 2050]:
    for t in [2023, 2028, 2034, 2040, 2050]:
        process_and_plot_expansion_edge(output, base_data, t, sel_scenario, cumulative=True, show_fig=False, save_fig=True)
        process_and_plot_expansion_node(output, base_data, t, sel_scenario, cumulative=True, show_fig=False, save_fig=True)