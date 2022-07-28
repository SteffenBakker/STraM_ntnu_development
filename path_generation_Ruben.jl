#########################################
#READ AND PROCESS DATA

#using Pandas
using XLSX
using DataFrames


df_dist_sea = DataFrame(XLSX.readtable("Data/distances.xlsx", "Sea"))[:,1:3]
df_dist_rail = DataFrame(XLSX.readtable("Data/distances.xlsx", "Rail"))[1:16,1:3] #HARDCODED: delete final column
df_dist_road = DataFrame(XLSX.readtable("Data/distances.xlsx", "Road"))[:,1:3]
df_costs = DataFrame(XLSX.readtable("Data/transport_costs_emissions.xlsx", "Costs"))[:,1:11]
df_transfer_costs = DataFrame(XLSX.readtable("Data/transport_costs_emissions.xlsx", "transfer_costs"))
delete!(df_dist_rail, [9]) #HARDCODED: delete (longest) duplicate route

colnames_dist = ["orig", "dest", "dist"]
colnames_costs = ["mode", "product", "vehicle", "fuel", "year", "costs_eur", "costs", "emissions", "co2_fee", "co2_fee_high", "co2_fee_low"]
colnames_transfer = ["product", "transfer_type", "transfer_cost"]
rename!(df_dist_sea, colnames_dist)
rename!(df_dist_rail, colnames_dist)
rename!(df_dist_road, colnames_dist)
rename!(df_costs, colnames_costs)
rename!(df_transfer_costs, colnames_transfer)

#make list of DataFrames
df_list = [df_dist_road, df_dist_sea, df_dist_rail] #index d

#define nodes
nodes = String[]
#add nodes to list from all three date files
for d in 1:length(df_list)
    for i in 1:size(df_list[d],1)
        if !(df_list[d].orig[i] in nodes)
            append!(nodes, [df_list[d].orig[i]])
        end
        if !(df_list[d].dest[i] in nodes)
            append!(nodes, [df_list[d].dest[i]])
        end
    end
end
num_nodes = length(nodes)
#make node dictionary
node_dict = Dict(zip(nodes, 1:length(nodes)))

#make adjacency list 
adj_sets = Array{Any}(undef, num_nodes)
for n in 1:num_nodes
    adj_sets[n] = Set{Int64}()
end

for d in 1:length(df_list)
    for i in 1:size(df_list[d],1)
        orig_index = node_dict[df_list[d].orig[i]]
        dest_index = node_dict[df_list[d].dest[i]]
        adj_sets[orig_index] = union!(adj_sets[orig_index], Set([dest_index]))
        adj_sets[dest_index] = union!(adj_sets[dest_index], [orig_index])
    end
end

#define product list
products = []
for i in 1:size(df_costs,1)
    if !(df_costs.product[i] in products)
        append!(products, [df_costs.product[i]])
    end
end
num_products = length(products)
product_dict = Dict(zip(products,1:num_products))

#define mode list 
modes = []
for i in 1:size(df_costs, 1)
    if !(df_costs.mode[i] in modes)
        append!(modes, [df_costs.mode[i]])
    end
end
num_modes = length(modes)
mode_dict = Dict(zip(modes, 1:length(modes)))

#define vehicle lists (UNUSED) and
#define fuel list
vehicles = Array{Any}(undef, num_modes)
fuels = Array{Any}(undef, num_modes)
for m in 1:num_modes
    vehicles[m] = []
    fuels[m] = []
    for i in 1:size(df_costs, 1)
        if df_costs.mode[i] == modes[m]
            if !(df_costs.vehicle[i] in vehicles[m])
                append!(vehicles[m], [df_costs.vehicle[i]])
            end
            if !(df_costs.fuel[i] in fuels[m])
                append!(fuels[m], [df_costs.fuel[i]])
            end
        end
    end
end

#list all fuels over all modes 
all_fuels = []
for i in 1:length(fuels)
    for j in 1:length(fuels[i])
        all_fuels = union(all_fuels, [fuels[i][j]])
    end
end
num_fuels = length(all_fuels)

#dict from fuel to integer
fuel_dict = Dict(zip(all_fuels, 1:length(all_fuels)))

#define years
years = [2020, 2025, 2030, 2040, 2050] #HARDCODED
num_years = length(years)
year_dict = Dict(zip(years, 1:num_years))

#create cost matrix
costs = zeros(num_modes, num_products, num_fuels, num_years)
for i in 1:size(df_costs, 1)
    m = mode_dict[df_costs.mode[i]]
    p = product_dict[df_costs.product[i]]
    f = fuel_dict[df_costs.fuel[i]]
    y = year_dict[df_costs.year[i]]
    costs[m,p,f,y] = df_costs.costs[i] + df_costs.emissions[i] * df_costs.co2_fee[i]
end


####################################
#COMPUTE SHORTEST PATHS

using Dijkstra

#adjacency_list = Dict(zip(1:num_nodes, adj_sets)) #not used 

#Find shortest paths for each mode 
sh_path_uni = [[] for o in 1:num_nodes, d in 1:num_nodes, m in 1:num_modes]  #unimodal shortest paths
max_dist = 999999.0
sh_dist_uni = max_dist * ones(num_nodes, num_nodes, num_modes)  #unimodal shortest distances

for m in 1:num_modes
    #initialize graph 
    G = Graph()
    #fill graph with nodes and edges (using minimum distance as edge cost)
    for o in 1:num_nodes
        for d in adj_sets[o]
            cost = 999999
            #compute cost (= shortest distance for mode m)
            for i in 1:size(df_list[m],1)
                if (df_list[m].orig[i] == nodes[o] && df_list[m].dest[i] == nodes[d]) || (df_list[m].orig[i] == nodes[d] && df_list[m].dest[i] == nodes[o])
                    cost = min(cost, df_list[m].dist[i])
                end
            end
            add_edge!(G, o, d, cost)
        end
    end
    #create shortest paths toward destination d
    for d in 1:num_nodes #loop over destinations 
        R = ShortestPath(G, d) #find shortest paths towards d
        for o in 1:num_nodes #loop over origins 
            stop = false
            cur_o = o #initialize cur_o
            sh_path_uni[o,d,m] = append!(sh_path_uni[o,d,m], o) #add origin to shortest path 
            while !stop
                cur_d = R.prev[cur_o] #current destination is predecessor of cur_o 
                sh_path_uni[o,d,m] = append!(sh_path_uni[o,d,m], cur_d) #add cur_d to shortest path 
                #stop if we have reached the destination
                if cur_d == d
                    stop = true #stop if we have reached the destination 
                end
                println("o = ", o, ", d = ", d, ", m = ", m,  ", cur_o = ", cur_o, ", cur_d = ", cur_d, ", stop = ", stop)
                cur_o = cur_d #update cur_o (old dest becomes new orig)
            end
            sh_dist_uni[o,d,m] = get_distance(R, o) #update shortest distance 
            if sh_dist_uni[o,d,m] >= max_dist 
                sh_path_uni[o,d,m] = [] #remove "fake" paths (that don't exist)
            end
        end
    end
end

