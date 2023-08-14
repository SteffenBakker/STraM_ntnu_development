import pandas as pd
import numpy as np
from dijkstar import Graph, find_path

# READ AND PROCESS DATA
df_dist = pd.read_excel("Data/Spatial/spatial_data.xlsx", sheet_name="distances_STRAM", usecols=list(range(5)))
df_zones = pd.read_excel("Data/Spatial/spatial_data.xlsx", sheet_name="zones_STRAM")

edges_with_double_route = []
edges_with_double_route_matrix = df_dist[df_dist['Route']==2]
edges_with_double_route_matrix.shape
for index, row in edges_with_double_route_matrix.iterrows():
    i = row['From']
    j = row['To']
    m = row['Mode']
    edges_with_double_route.append((i,j,m))
df_dist = df_dist[~(df_dist['Route'] ==2)] #delete "additional routes"
df_dist = df_dist.reset_index(drop=True)

df_costs = pd.read_excel("Data/transport_costs_emissions.xlsx", sheet_name="costs_emissions",usecols=list(range(7)))
df_co2_fee = pd.read_excel("Data/transport_costs_emissions_raw.xlsx", sheet_name="CO2_fee")
df_transfer_costs = pd.read_excel("Data/transport_costs_emissions_raw.xlsx", sheet_name="transfer_costs")


colnames_dist = ["orig", "dest", "dist","mode","route"]
colnames_costs = ["mode", "product", "vehicle", "fuel", "year", "costs", "emissions"]
colnames_transfer = ["product", "transfer_from","transfer_to", "transfer_cost"]
colnames_co2_fee = ["year", "co2_fee_base", "co2_fee_scen2", "co2_fee_scen3"]

df_dist.columns = colnames_dist
df_costs.columns = colnames_costs
df_transfer_costs.columns = colnames_transfer
df_co2_fee.columns = colnames_co2_fee

node_dict = {zone: centroid for zone, centroid in zip(df_zones['zone_nr'], df_zones['centroid_name'])}


#define product list
products = df_costs['product'].unique()
num_products = len(products)

#define mode list 
modes = df_costs["mode"].unique()
num_modes = len(modes)
mode_dict = {mode: idx for idx, mode in enumerate(modes)}

#define nodes
nodes = []
nodes.extend(df_dist['orig'].unique())
nodes.extend(df_dist['dest'].unique())

nodes = list(set(nodes))
nodes = [node for node in nodes]
num_nodes = len(nodes)

#make adjacency list 
adj_map = {(i,m):set() for i in nodes for m in modes}
for index, row in df_dist.iterrows():
    mode = row['mode']
    orig = row['orig']
    dest = row['dest']
    adj_map[(orig,mode)].add(dest)
    adj_map[(dest,mode)].add(orig)


#define vehicle lists (UNUSED) and
#define fuel list
vehicles = {m:list() for m in modes}
mode_to_fuels = {m:list() for m in modes}
for m in modes:
    for index,row in df_costs.iterrows(): 
        if row['mode'] == m:
            if row['vehicle'] not in vehicles[m]:
                vehicles[m].append(row['vehicle'])
            if row['fuel'] not in mode_to_fuels[m]:
                mode_to_fuels[m].append(row['fuel'])

#define years
years = [2022, 2026, 2030, 2040, 2050]
num_years = len(years)
year_dict = {year: idx for idx, year in enumerate(years)}

#create cost matrix
costs = {(m,p,f,y):0 for m in modes for p in products for f in mode_to_fuels[m] for y in years}
for index,row in df_costs.iterrows():
    m = row['mode']
    p = row['product']
    f = row['fuel']
    y = row['year']
    costs[(m,p,f,y)] = row['costs']  + + row['emissions']* df_co2_fee['co2_fee_base'][year_dict[y]]  

#create transfer cost matrix 
prod_transfer_cost = {(m,mm,p):0 for m in modes for mm in modes for p in products}
for index,row in df_transfer_costs.iterrows():
    m1 = row['transfer_from']
    m2 = row['transfer_to']
    p = row['product']
    prod_transfer_cost[(m1, m2, p)] = row['transfer_cost']



#distance_mapping
max_dist = 999999.0
distance = {(i,j,m):max_dist for i in nodes for j in nodes for m in modes}
for index,row in df_dist.iterrows():
    i = row["orig"]
    j = row["dest"]
    m = row["mode"]
    dist = row["dist"]
    distance[(i,j,m)] = dist
    distance[(j,i,m)] = dist

####################################
# COMPUTE SHORTEST PATHS
####################################

def gen_paths(mode_cost, transfer_cost, mode_comb_level=2):
    sh_path_uni = {(i,j,m):[] for i in nodes for j in nodes for m in modes}  #unimodal shortest paths
    max_dist = 999999.0
    sh_dist_uni = {(i,j,m):max_dist for i in nodes for j in nodes for m in modes} #unimodal shortest distances

    for m in modes:
        #initialize graph 
        G = Graph()
        #fill graph with nodes and edges (using minimum distance as edge cost)
        for o in nodes:
            for d in adj_map[(o,m)]:
                #compute cost (= shortest distance for mode m)
                cost = distance[(o,d,m)]*mode_cost[m] 
                G.add_edge(o, d, cost)
        #create shortest paths toward destination d BRUTE FORCE (some unneccesary calculations)
        for o in nodes:
            for d in nodes:
                if o != d:
                    try:
                        result = find_path(G, o, d)
                        sh_path_uni[(o,d,m)] = result.nodes
                        sh_dist_uni[(o,d,m)] = result.total_cost
                        if sh_dist_uni[(o,d,m)] >= max_dist:
                            sh_path_uni[(o,d,m)] = [] #remove "fake" paths (that don't exist) 
                    except:
                        pass


    mode_combinations = []
    #single-mode combinations 
    for m in modes:
        mode_combinations.append([m])
    #two-mode combinations 
    if mode_comb_level >= 2:
        for m1 in modes:
            for m2 in modes:
                if m1 != m2:
                    mode_combinations.append([m1, m2])
    #three-mode combinations 
    if mode_comb_level >= 3:
        for m1 in modes:
            for m2 in modes:
                for m3 in modes:
                    if m1 != m2 and m2 != m3:
                        mode_combinations.append([m1, m2, m3])

    mode_combi = []
    mode_combi_dict = {}
    mode_combi_dict_inverse = {}
    num_mode_combinations = len(mode_combinations)
    for i in range(num_mode_combinations):
        mode_combi.append(i)
        mode_combi_dict[i] = mode_combinations[i]
        mode_combi_dict_inverse[tuple(mode_combinations[i])] = i
    

    #DEFINE ALL SHORTEST PATHS (both uni and multimodal)
    
    sh_path = {(i,j,mc):[] for i in nodes for j in nodes for mc in mode_combi}  #unimodal shortest paths
    sh_dist = {(i,j,mc):max_dist for i in nodes for j in nodes for mc in mode_combi} #unimodal shortest distances

    for mc_index in mode_combi:
        mc = mode_combi_dict[mc_index]
        #uni-modal paths 
        if len(mc) == 1:
            m = mc[0]
            for o in nodes:
                for d in nodes:
                    if o != d: #don't make loops
                        path_list = []
                        if len(sh_path_uni[(o,d,m)]) > 1: #not an empty path (not unreachable)
                            for l in range(len(sh_path_uni[(o,d,m)]) - 1): #loop over leg l in the path 
                                cur_o = sh_path_uni[(o,d,m)][l]
                                cur_d = sh_path_uni[(o,d,m)][l + 1]
                                path_list.append((cur_o, cur_d, m, 1)) #HARDCODED: route number 1 (we add number 2 later, only for hamar-trondheim railway)
                        sh_path[(o,d,mc_index)] = path_list
                        sh_dist[(o,d,mc_index)] = sh_dist_uni[(o,d,m)]
                    else:
                        sh_path[(o,d,mc_index)] = []
                        sh_dist[(o,d,mc_index)] = max_dist
        #two-mode paths
        elif len(mc) == 2:
            m1 = mc[0]
            m1_index = mode_combi_dict_inverse[tuple([m1])]
            m2 = mc[1]
            m2_index = mode_combi_dict_inverse[tuple([m2])]
            for o in nodes:
                for d in nodes:
                    if o != d:
                        best_dist = max_dist
                        best_mid_point = 0
                        #find best mid-point 
                        for n in nodes:
                            cur_dist = max_dist + 1
                            if len(sh_path[(o,d,m1_index)]) > 1:
                                if len(sh_path[(o,d,m2_index)]) > 1:
                                    cur_dist = sh_dist[(o,n,m1_index)] + sh_dist[(n,d,m2_index)] + transfer_cost[(m1,m2)]
                                    if cur_dist < best_dist:
                                        best_mid_point = n
                                        best_dist = cur_dist
                        if best_mid_point != 0: #found a best midpoint 
                            sh_path[(o,d,mc_index)] = sh_path[(o,best_mid_point,m1_index)] + sh_path[(best_mid_point,d,m2_index)]
                            sh_dist[(o,d,mc_index)] = sh_dist[(o,best_mid_point,m1_index)] + sh_dist[(best_mid_point,d,m2_index)] + transfer_cost[(m1,m2)]
        #three-mode paths (note that all two-mode paths are already done)
        elif len(mc) == 3:
            m1 = mc[0]
            m1_index = mode_combi_dict_inverse[tuple([m1])]
            m2 = mc[1]
            m2_index = mode_combi_dict_inverse[tuple([m2])]
            m3 = mc[2]
            m3_index = mode_combi_dict_inverse[tuple([m3])]
            mc1 = [m1]       #mode-combination 1: first leg (so equal to m1)
            mc1_index = mode_combi_dict_inverse[tuple([m1])]
            mc2 = [m2,m3]
            mc2_index = mode_combi_dict_inverse[(m2,m3)]
            for o in nodes:
                for d in nodes:
                    if o != d: #don't make loops
                        best_dist = max_dist
                        best_mid_point = 0
                        #find best mid-point 
                        for n in nodes:
                            cur_dist = sh_dist[(o,n,mc1_index)] + sh_dist[(n,d,mc2_index)] + transfer_cost[(m1,m2)]
                            if cur_dist < best_dist:
                                best_mid_point = n
                                best_dist = cur_dist
                        if best_mid_point != 0:
                            sh_path[(o,d,mc_index)] = sh_path[(o,best_mid_point,mc1_index)] + sh_path[(best_mid_point,d,mc2_index)]
                            sh_dist[(o,d,mc_index)] = sh_dist[(o,best_mid_point,mc1_index)] + sh_dist[(best_mid_point,d,mc2_index)] + transfer_cost[(m1,m2)]

    #make list of generated paths   
    generated_paths = []
    generated_path_lengths = []
    for o in nodes:
        for d in nodes:
            for mc_index in mode_combi:
                if len(sh_path[(o,d,mc_index)]) > 0:
                    #add path to list 
                    shortest_path = sh_path[(o,d,mc_index)]
                    generated_paths.append(shortest_path)
                    generated_path_lengths.append(sh_dist[(o,d,mc_index)])

                    #add copy of hamar-trondheim railway if exists 
                    route_no_2 = False #boolean checking if there exists an alternative route
                    which_leg = 0 #keeping track of the leg where it occurs
                    for l in range(len(sh_path[(o,d,mc_index)])):
                        i = sh_path[(o,d,mc_index)][l][0]
                        j = sh_path[(o,d,mc_index)][l][1]
                        m = sh_path[(o,d,mc_index)][l][2]
                        if ((i,j,m) in edges_with_double_route) or ((j,i,m) in edges_with_double_route):
                            route_no_2 = True
                            which_leg = l
                    if route_no_2:
                        #make copy with the other route (use same path length (only minor difference in reality))
                        new_path = sh_path[(o,d,mc_index)].copy()
                        new_path[which_leg] = (new_path[which_leg][0], new_path[which_leg][1],
                                                 new_path[which_leg][2], 2)
                        #add path to list
                        generated_paths.append(new_path)
                        generated_path_lengths.append(sh_dist[(o,d,mc_index)])

    return sh_path, sh_dist, generated_paths, generated_path_lengths

#set max number of different modes per path
mode_comb_level = 2

#Generate all paths (loop over all years, products, fuels)
all_gen_paths = set()
#years=[years[0]]  #to reduce the computational burden. The results should be somewhat similar
#products =[products[0]]

for y in years:
    for p in products:
        transf_costs ={(m1,m2):cc for (m1,m2,pp),cc in prod_transfer_cost.items() if (p == pp)}
        #consider all combinations of mode fuels 
        for f1 in mode_to_fuels[modes[0]]: #road fuel 
            for f2 in mode_to_fuels[modes[1]]: #sea fuel 
                for f3 in mode_to_fuels[modes[2]]: #rail fuel 
                    cur_mode_fuels = [f1, f2, f3] #vector with fuel per mode 
                    #define vector of cost per mode (per Tkm)
                    mode_cost = {m:0 for m in modes}
                    for m_index in range(len(modes)):
                        f = cur_mode_fuels[m_index]
                        m= modes[m_index]
                        mode_cost[m] = costs[(m,p,f,y)]
                    
                    #run the path generation for the current costs 
                    (sh_path, sh_dist, generated_paths, generated_path_lengths) = gen_paths(mode_cost,
                                                                                           transf_costs,
                                                                                           mode_comb_level)
                    for path in generated_paths:
                        all_gen_paths.add(tuple(path))

all_gen_paths = list(all_gen_paths)
print(len(all_gen_paths))

# WRITE TO CSV FILE    UPDATE
file_name = "Data/SPATIAL/generated_paths_"+str(mode_comb_level)+"_modes.csv"
with open(file_name, "w") as f:
    f.write(",paths\n")
    for i in range(len(all_gen_paths)):
        f.write(str(i) + ",\"[")
        for l in range(len(all_gen_paths[i])):
            leg = all_gen_paths[i][l]
            f.write("({}, {}, '{}', {})".format(leg[0], leg[1], leg[2], leg[3]))
            if l < len(all_gen_paths[i]) - 1:
                f.write(", ")
        f.write("]\"\n")
    

