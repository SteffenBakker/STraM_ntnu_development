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


#########################
# 1. PROCESS FLOW DATA
#########################

# function that processes and aggregates flows
def process_and_aggregate_flows(x_flow, b_flow, sel_scenario, sel_time_period):
    """
    Process model output (x_flow and b_flow), aggregate the flow per edge, and output a dataframe.
    All for a selected scenario and time period

    INPUT
    x_flow:           dataframe with flow of goods
    b_flow:           dataframe with balancing flow (empty vehicles)
    sel_scenario:     scenario name or "average"
    sel_time_period:  time period in [2020, 2025, 2030, 2040, 2050]
    
    OUTPUT
    df_flow:          dataframe with aggregated flows   
    """

    #copy all flows into one dataframe: product flow and balancing flow
    all_flow = pd.concat([
        x_flow[["from", "to", "mode", "fuel", "scenario", "time_period", "weight"]], 
        b_flow[["from", "to", "mode", "fuel", "scenario", "time_period", "weight"]]
        ])

    #create lists that will store aggregate flows (these will be the columns of df_flow)
    arcs = []
    flows = []
    flows_road = []
    flows_sea = []
    flows_rail = []

    #scenario list and counter in order to take average when necessary
    all_scenarios = []

    #list all nodes clockwise, to make sure sea edges curve in the right direction (HARDCODED)
    nodes_sea_order = ["Nord-Sverige", "Sør-Sverige", "Hamar", "Oslo", "Skien", "Kristiansand", "Stavanger", 
                        "Bergen", "Ålesund", "Trondheim", "Bodø", "Tromsø", "Europa", "Verden", "Kontinentalsokkelen"]

    #add arcs and corresponding flow to the right lists
    #Note: I use the word arc, but we treat them as edges. That is, we look at undirectional flow by aggregating over both directions
    for index, row in all_flow.iterrows():
        if row["time_period"] == sel_time_period:
            if sel_scenario == "average" or row["scenario"] == sel_scenario: #if "average", we add everything and divide by number of scenarios at the end
                #add scenario to list if not observed yet (for taking average)
                if row["scenario"] not in all_scenarios:
                    all_scenarios.append(row["scenario"])
                #temporarily store current arc and its opposite
                cur_arc = (row["from"], row["to"])
                cur_cra = (row["to"], row["from"]) #opposite arc
                #check if new arc
                if cur_arc not in arcs and cur_cra not in arcs: #new arc
                    #determine direction of arc based on nodes_sea_order and append the arc
                    from_order = nodes_sea_order.index(row["from"]) 
                    to_order = nodes_sea_order.index(row["to"])
                    if from_order < to_order:
                        arcs.append(cur_arc) #append forward arc
                    else:
                        arcs.append(cur_cra) #append backward arc
                    #add zero values for the corresponding flows (initialization)
                    flows.append(0.0)
                    flows_road.append(0.0)
                    flows_sea.append(0.0)
                    flows_rail.append(0.0)
                #find index of current arc (or cra) in list "arcs"
                cur_arc_ind = None
                if cur_arc in arcs:
                    cur_arc_ind = arcs.index(cur_arc)
                elif cur_cra in arcs:
                    cur_arc_ind = arcs.index(cur_cra)
                #store corresponding flows in lists
                flows[cur_arc_ind] += row["weight"]
                if row["mode"] == "Road":
                    flows_road[cur_arc_ind] += row["weight"]
                elif row["mode"] == "Sea":
                    flows_sea[cur_arc_ind] += row["weight"]
                elif row["mode"] == "Rail":
                    flows_rail[cur_arc_ind] += row["weight"]

    #divide everything by number of scenarios if we have selected sel_scenario="average"
    if sel_scenario == "average":
        num_scenarios = len(all_scenarios)
        flows = [(1.0/num_scenarios) * f for f in flows]
        flows_road = [(1.0/num_scenarios) * f for f in flows_road]
        flows_sea = [(1.0/num_scenarios) * f for f in flows_sea]
        flows_rail = [(1.0/num_scenarios) * f for f in flows_rail]

    #store aggregate flows in a dataframe
    df_flow = pd.DataFrame()
    df_flow["arc"] = arcs
    df_flow["orig"] = [""]*len(arcs)
    df_flow["dest"] = [""]*len(arcs)
    df_flow["flow"] = flows
    df_flow["flow_road"] = flows_road
    df_flow["flow_sea"] = flows_sea
    df_flow["flow_rail"] = flows_rail

    #fix origins and destinations
    for i in range(len(df_flow)):
        df_flow.orig[i] = str(df_flow.arc[i][0])
        df_flow.dest[i] = str(df_flow.arc[i][1])

    #return a dataframe with aggregated flows
    return df_flow

# select scenario and time period
sel_scenario = "average" #either a scenarioi name or "average"
sel_time_period = 2050 #one of [2020, 2025, 2030, 2040, 2050]

# process and aggregate flows
df_flow = process_and_aggregate_flows(output.x_flow, output.b_flow, sel_scenario, sel_time_period)

######################
# 2. CREATE MAP PLOT
######################

#TODO: FIGURE THAT PLOTS DIFFERENCES

def plot_flow_on_map(df_flow, base_data, flow_variant, mode_variant, plot_overseas=True, plot_up_north=True, show_fig=True, save_fig=False):    
    """
    Create a plot on the map of Norway with all the flows

    INPUT
    df_flow:        dataframe with aggregated flows per edge (split out by mode)
    base_data:      base model data (only used to extract N_NODES)
    flow_variant:   TODO
    mode_variant:   what mode to plot; choose from ["road", "sea", "rail", "all", "total"]
    plot_overseas:  indicate whether to plot flow to oversees nodes ("Kontinentalsokkelen", "Europa", and "Verden")
    plot_up_north:  indicate whether to plot flow to nodes up north ("Bodø" and "Tromsø")
    show_fig:       indicate whether to show the figure
    save_fig:       indicate whether to save the figure (filename is determined automatically)
    
    OUTPUT
    figure that is shown and/or saved to disk if requested
    """

    ####################################
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
    node_colors[-1] = "red"    #Verden                
    node_colors[-2] = "green"  #Europa                
    node_colors[-3] = "yellow" #Kontinentalsokkelen   
    #node_colors[-4] = "blue"   #Nord-Sverige         
    #node_colors[-5] = "grey"   #Sør-Sverige          


    ####################
    # b. Build a map

    #import map/plotting tools
    from mpl_toolkits.basemap import Basemap #for creating the background map
    import matplotlib.pyplot as plt #for plotting on top of the background map

    #draw the basic map including country borders
    map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='aeqd', lat_0=63.4, lon_0=10.4) # Azimuthal Equidistant Projection
    # map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0) # mercator projection
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='lightgrey', lake_color='aqua')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries(linewidth=0.3)

    #draw nodes on the map
    node_x, node_y = map(lons, lats)
    map.scatter(node_x, node_y, color=node_colors, zorder=100)


    ##########################
    # c. Plot flow in the map

    #import library for fancy arrows/edges
    import matplotlib.patches as patches

    #arrow settings
    tail_width_base = 8
    head_with = 0.01
    head_length = 0.01
    base_curvature = 0.2
    #arrow settings for the different modes
    color_dict = {"road":"grey", "sea":"blue", "rail":"darkgreen", "total":"black"}
    curvature_fact_dict = {"road":0, "sea":-2, "rail":+1, "total":0}

    #compute maximum and total flows over all edges (for scaling purposes)
    max_flow = max(df_flow["flow"])
    max_flow_road = max(df_flow["flow_road"])
    max_flow_sea = max(df_flow["flow_sea"])
    max_flow_rail = max(df_flow["flow_rail"])
    total_flow = sum(df_flow["flow"])
    total_flow_road = sum(df_flow["flow_road"])
    total_flow_sea = sum(df_flow["flow_sea"])
    total_flow_rail = sum(df_flow["flow_rail"])
    #store in dictionaries
    total_flow_dict = {"road":total_flow_road, "sea":total_flow_sea, "rail":total_flow_rail, "total":total_flow, "all":total_flow}
    max_flow_dict = {"road":max_flow_road, "sea":max_flow_sea, "rail":max_flow_rail, "total":max_flow, "all":max_flow}

    #iterate over egdes
    for index, row in df_flow.iterrows():
        #store current origin and destinations + indices
        cur_orig = row["orig"]
        cur_dest = row["dest"]
        cur_orig_index = N_NODES.index(cur_orig)
        cur_dest_index = N_NODES.index(cur_dest)
        #check if it is a long distance (temporarily don't plot those to avoid cluttering)
        overseas = False
        up_north = False
        if cur_orig in ["Kontinentalsokkelen", "Europa", "Verden"] or cur_dest in ["Kontinentalsokkelen", "Europa", "Verden"]:
            overseas = True
        if cur_orig in ["Bodø", "Tromsø"] or cur_dest in ["Bodø", "Tromsø"]:
            up_north = True
        #check mode variant
        if mode_variant == "all": #we will plot all modes in one figure
            #create dictionary that stores all the flows
            flow_dict = {"road":row["flow_road"], "sea":row["flow_sea"], "rail":row["flow_rail"]}
            #loop over the three modes
            for cur_mode in ["road", "sea", "rail"]:
                #extract information the current mode
                cur_flow = flow_dict[cur_mode]
                cur_total_flow = total_flow_dict[mode_variant]
                cur_max_flow = max_flow_dict[mode_variant]
                curvature_factor = curvature_fact_dict[cur_mode] #indicates in what direction the arc should bend
                #create new arc
                if cur_flow > 0.001*cur_total_flow: #only plot an arc if we have significant flow (at least 0.1% of total flow for the relevant mode)
                    new_arc = patches.FancyArrowPatch(
                        (node_x[cur_orig_index], node_y[cur_orig_index]),  #origin coordinates
                        (node_x[cur_dest_index], node_y[cur_dest_index]),  #destination coordinates
                        connectionstyle=f"arc3,rad={base_curvature * curvature_factor}", #curvature of the edge
                        arrowstyle=f"Simple, tail_width={tail_width_base * cur_flow/cur_max_flow}, head_width={head_with}, head_length={head_length}", #tail width: constant times normalized flow
                        color=color_dict[cur_mode]
                        )    
                    if ((not overseas) or (overseas and plot_overseas)) and ((not up_north) or (up_north and plot_up_north)): #only add the arc if we want to plot it
                        #add the arc to the plot
                        plt.gca().add_patch(new_arc) 
        else: #we only plot one mode
            #get current flow and related information
            cur_flow = 0.0
            if mode_variant == "total":
                cur_flow = row["flow"]
            elif mode_variant == "road":
                cur_flow = row["flow_road"]
            elif mode_variant == "sea":
                cur_flow = row["flow_sea"]
            elif mode_variant == "rail":
                cur_flow = row["flow_rail"]
            cur_total_flow = total_flow_dict[mode_variant]
            cur_max_flow = max_flow_dict[mode_variant]
            curvature_factor = curvature_fact_dict[mode_variant] #indicates in what direction the arc should bend
            #create new arc
            if cur_flow > 0.001*cur_total_flow: #only plot an arc if we have significant flow (at least 0.1% of total flow for the relevant mode)
                new_arc = patches.FancyArrowPatch(
                    (node_x[cur_orig_index], node_y[cur_orig_index]), 
                    (node_x[cur_dest_index], node_y[cur_dest_index]), 
                    connectionstyle=f"arc3,rad={base_curvature * curvature_factor}",
                    arrowstyle=f"Simple, tail_width={tail_width_base * cur_flow/cur_max_flow}, head_width={head_with}, head_length={head_length}", #tail width: constant times normalized flow
                    color=color_dict[mode_variant]
                    )    
                if ((not overseas) or (overseas and plot_overseas)) and ((not up_north) or (up_north and plot_up_north)): #only add the arc if we want to plot it
                    #add the arc to the plot
                    plt.gca().add_patch(new_arc)
        
    ###############################
    # d. Show and save the figure

    #set size
    plt.gcf().set_size_inches(8.5,10.5, forward=True) #TODO: FIND THE RIGH TSIZE
    #save figure
    if save_fig:
        filename = f"flow_plot_{sel_time_period}_{sel_scenario}_{mode_variant}.png"
        plt.savefig(filename)
    #show figure
    if show_fig:
        plt.show()

# select what type of plot to make
flow_variant = "flow" # ["flow", "difference"]
mode_variant = "all" # ["road", "sea", "rail", "all", "total"]

# select whether to plot distant flow
plot_overseas = False 
plot_up_north = False


plot_flow_on_map(df_flow, base_data, flow_variant, mode_variant, plot_overseas, plot_up_north)




##################

#TODO:
"""
- write code that can plot increase/decrease in flow
"""


