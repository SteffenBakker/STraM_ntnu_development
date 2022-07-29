#########################################
#READ AND PROCESS DATA

#using Pandas
#using CSV
using XLSX
using DataFrames


df_dist_sea = DataFrame(XLSX.readtable("Data/distances.xlsx", "Sea")...)[:,1:3]
df_dist_rail = DataFrame(XLSX.readtable("Data/distances.xlsx", "Rail")...)[1:16,1:3] #HARDCODED: delete final column
df_dist_road = DataFrame(XLSX.readtable("Data/distances.xlsx", "Road")...)[:,1:3]
df_costs = DataFrame(XLSX.readtable("Data/transport_costs_emissions.xlsx", "Costs")...)[:,1:11]
df_transfer_costs = DataFrame(XLSX.readtable("Data/transport_costs_emissions.xlsx", "transfer_costs")...)
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

#process df_transfer_costs 
transfer_prod_list = Array{Any}(undef, size(df_transfer_costs,1))
transfer_mode_list = Array{Any}(undef, size(df_transfer_costs,1))
for i in 1:length(transfer_mode_list)
    #fix transfer_product
    transfer_prod_list[i] = df_transfer_costs.product[i]
    #fix transfer_mode 
    if mod(i,3) == 1 #sea-rail 
        transfer_mode_list[i] = [mode_dict["Sea"], mode_dict["Rail"]]
    elseif mod(i,3) == 2 #sea-road 
        transfer_mode_list[i] = [mode_dict["Sea"], mode_dict["Road"]]
    elseif mod(i,3) == 0 #rail-road 
        transfer_mode_list[i] = [mode_dict["Rail"], mode_dict["Road"]]
    end        
end


#create transfer cost matrix 
prod_transfer_cost = zeros(num_modes, num_modes, num_products)
for i in 1:length(transfer_mode_list)
    m1 = transfer_mode_list[i][1]
    m2 = transfer_mode_list[i][2]
    p = product_dict[transfer_prod_list[i]]
    prod_transfer_cost[m1,m2,p] = df_transfer_costs.transfer_cost[i]
    prod_transfer_cost[m2,m1,p] = df_transfer_costs.transfer_cost[i]
end


####################################
#COMPUTE SHORTEST PATHS

using Dijkstra

#function that generates shortest paths given a certain cost per mode
function gen_paths(mode_cost, transfer_cost, mode_comb_level=3)
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
                cost = max_dist
                #compute cost (= shortest distance for mode m)
                for i in 1:size(df_list[m],1)
                    if (df_list[m].orig[i] == nodes[o] && df_list[m].dest[i] == nodes[d]) || (df_list[m].orig[i] == nodes[d] && df_list[m].dest[i] == nodes[o])
                        cost = min(cost, df_list[m].dist[i] * mode_cost[m])
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
                    #println("o = ", o, ", d = ", d, ", m = ", m,  ", cur_o = ", cur_o, ", cur_d = ", cur_d, ", stop = ", stop)
                    cur_o = cur_d #update cur_o (old dest becomes new orig)
                end
                sh_dist_uni[o,d,m] = get_distance(R, o) #update shortest distance 
                if sh_dist_uni[o,d,m] >= max_dist 
                    sh_path_uni[o,d,m] = [] #remove "fake" paths (that don't exist)
                end
                #remove cycles (o==d)
                if o == d 
                    sh_path_uni[o,d,m] = []
                    sh_dist_uni[o,d,m] = max_dist #also set distance to +infty so we won't use it 
                end
            end
        end
    end


    #define mode combinations 
    mode_combinations = [] #(index mc)
    #single-mode combinations 
    for m in modes
        append!(mode_combinations, [[m]])
    end
    #two-mode combinations 
    if mode_comb_level >= 2
        for m1 in modes
            for m2 in modes
                if !(m1 == m2)
                    append!(mode_combinations, [[m1, m2]])
                end
            end
        end
    end
    #three-mode combinations 
    if mode_comb_level >= 3
        for m1 in modes
            for m2 in modes
                for m3 in modes
                    if !(m1 == m2) && !(m2 == m3)
                        append!(mode_combinations, [[m1, m2, m3]])
                    end
                end
            end
        end
    end
    num_mode_combinations = length(mode_combinations)

    #define shortest paths (both uni and multimodal)
    sh_path = [[] for o in 1:num_nodes, d in 1:num_nodes, mc in 1:num_mode_combinations]  #uni-/multimodal paths 
    sh_dist = max_dist * ones(num_nodes, num_nodes, num_mode_combinations)

    for mc in 1:num_mode_combinations 
        #uni-modal paths 
        if length(mode_combinations[mc]) == 1
            m = mc 
            for o in 1:num_nodes 
                for d in 1:num_nodes 
                    if o != d #don't make loops
                        path_list = []
                        if length(sh_path_uni[o,d,m]) > 1 #not an empty path (not unreachable)
                            for l in 1:(length(sh_path_uni[o,d,m]) - 1) #loop over leg l in the path 
                                cur_o = sh_path_uni[o,d,m][l]
                                cur_d = sh_path_uni[o,d,m][l + 1]
                                append!(path_list, [(nodes[cur_o], nodes[cur_d], modes[m], 1)]) #HARDCODED: route number 1 (we add number 2 later, only for hamar-trondheim railway)
                            end
                        end
                        sh_path[o,d,m] = path_list 
                        sh_dist[o,d,m] = sh_dist_uni[o,d,m]
                    end
                end
            end
        #two-mode paths
        elseif length(mode_combinations[mc]) == 2 
            m1 = mode_dict[mode_combinations[mc][1]]
            m2 = mode_dict[mode_combinations[mc][2]]
            if transfer_cost[m1,m2] == 0.0
                println("m1 = ", m1, ", m2 = ", m2)
            end
            for o in 1:num_nodes 
                for d in 1:num_nodes 
                    if o != d #don't make loops
                        best_dist = max_dist
                        best_mid_point = 0
                        #find best mid-point 
                        for n in 1:num_nodes 
                            cur_dist = sh_dist[o,n,m1] + sh_dist[n,d,m2] + transfer_cost[m1,m2]
                            if cur_dist < best_dist 
                                best_mid_point = n
                                best_dist = cur_dist
                            end
                        end
                        if best_mid_point != 0 #found a best midpoint 
                            sh_path[o,d,mc] = vcat(sh_path[o, best_mid_point, m1], sh_path[best_mid_point, d, m2])
                            sh_dist[o,d,mc] = sh_dist[o, best_mid_point, m1] + sh_dist[best_mid_point, d, m2] + transfer_cost[m1,m2]
                        end
                    end
                end
            end
        #three-mode paths (note that all two-mode paths are already done)
        elseif length(mode_combinations[mc]) == 3 
            m1 = mode_dict[mode_combinations[mc][1]]
            m2 = mode_dict[mode_combinations[mc][2]]
            m3 = mode_dict[mode_combinations[mc][3]]
            mc1 = m1 #mode-combination 1: first leg (so equal to m1)
            #find mc2: second leg, mc corresponding to [m2, m3]
            mc2 = 0 #initialize
            for mc in 1:num_mode_combinations
                if length(mode_combinations[mc]) == 2
                    if mode_dict[mode_combinations[mc][1]] == m2 && mode_dict[mode_combinations[mc][2]] == m3 
                        mc2 = mc #now mc2 should be correct
                    end
                end
            end
            if mc2 == 0
                error("mc2 is incorrect")
            end
            for o in 1:num_nodes 
                for d in 1:num_nodes 
                    if o != d #don't make loops
                        best_dist = max_dist
                        best_mid_point = 0
                        #find best mid-point 
                        for n in 1:num_nodes 
                            cur_dist = sh_dist[o,n,mc1] + sh_dist[n,d,mc2] + transfer_cost[m1,m2]
                            if cur_dist < best_dist 
                                best_mid_point = n
                                best_dist = cur_dist
                            end
                        end
                        if best_mid_point != 0 #found a best midpoint 
                            sh_path[o,d,mc] = vcat(sh_path[o, best_mid_point, mc1], sh_path[best_mid_point, d, mc2])
                            sh_dist[o,d,mc] = sh_dist[o, best_mid_point, mc1] + sh_dist[best_mid_point, d, mc2] + transfer_cost[m1,m2]
                        end
                    end
                end
            end
        end
    end
    #make list of generated paths     
    generated_paths = []
    generated_path_lengths = []
    for o in 1:num_nodes
        for d in 1:num_nodes 
            for mc in 1:num_mode_combinations 
                if length(sh_path[o,d,mc]) > 0 #if valid path exists
                    #add path to list 
                    generated_paths = append!(generated_paths, [sh_path[o,d,mc]])
                    generated_path_lengths = append!(generated_path_lengths, [sh_dist[o,d,mc]])

                    #add copy of hamar-trondheim railway if exists 
                    ham_trd_rail = false #boolean checking if hamar-trondheim railway is in the route 
                    ham_trd_leg = 0 #boolean checking if hamar-trondheim railway is in the route 
                    for l in 1:length(sh_path[o,d,mc]) #loop over legs in path 
                        if sh_path[o,d,mc][l][3] == "Rail" #leg is a rail leg 
                            if sh_path[o,d,mc][l][1] == "Hamar" && sh_path[o,d,mc][l][2] == "Trondheim" #Ham-Trd
                                ham_trd_rail = true
                                ham_trd_leg = l
                            elseif sh_path[o,d,mc][l][1] == "Trondheim" && sh_path[o,d,mc][l][2] == "Hamar" #Trd-Ham
                                ham_trd_rail = true 
                                ham_trd_leg = l
                            end
                        end                        
                    end
                    if ham_trd_rail 
                        #make copy with the other route (use same path length (only minor difference in reality))
                        new_path = copy(sh_path[o,d,mc])
                        new_path[ham_trd_leg] = (new_path[ham_trd_leg][1], new_path[ham_trd_leg][2], new_path[ham_trd_leg][3], 2)
                        #add path to list
                        generated_paths = append!(generated_paths, [new_path])
                        generated_path_lengths = append!(generated_path_lengths, [sh_dist[o,d,mc]])
                    end
                end
            end
        end
    end

    
    #return the paths and the distances
    return (sh_path, sh_dist, generated_paths, generated_path_lengths)
end


#TEST gen_paths()
#cur_p = 1 #TEMP
#generate paths 
#(sh_path, sh_dist, generated_paths, generated_path_lengths) = gen_paths(mode_cost, prod_transfer_cost[:,:,cur_p])

#set max number of different modes per path
mode_comb_level = 2


#Generate all paths (loop over all years, products, fuels)
all_gen_paths = []
for y in 1:num_years
    for p in 1:num_products
        #consider all combinations of mode fuels 
        for f1 in fuels[1] #road fuel 
            for f2 in fuels[2] #sea fuel 
                for f3 in fuels[3] #rail fuel 
                    cur_mode_fuels = [f1, f2, f3] #vector with fuel per mode 
                    #define vector of cost per mode (per Tkm)
                    mode_cost = zeros(3)
                    for m in 1:num_modes
                        mode_cost[m] = costs[m,p,fuel_dict[cur_mode_fuels[m]],y]
                    end
                    #run the path generation for the current costs 
                    (sh_path, sh_dist, generated_paths, generated_path_lengths) = gen_paths(mode_cost, prod_transfer_cost[:,:,p], mode_comb_level)
                    global all_gen_paths = union(all_gen_paths, generated_paths)
                end
            end
        end
    end
end 

#print all paths
all_gen_paths 
#(approximately 12,000 paths. this is about 15x as many as AIM used)
#(approximately 3,500 paths if we only allow for max two modes per path0)



############################
#WRITE TO CSV FILE 


#Write to CSV file  (in exact same format as AIM originally did)
file_name = "Data/generated_paths_Ruben.csv"
open(file_name, "w") do f
    #header
    write(f, ",paths\n")
    for i in 1:length(all_gen_paths) #loop over paths 
        write(f, string(i-1))
        write(f, ",")
        write(f, "\"[")
        for l in 1:length(all_gen_paths[i]) #loop over legs 
            write(f, string("('", all_gen_paths[i][l][1], "', '", all_gen_paths[i][l][2], "', '", all_gen_paths[i][l][3], "', ", all_gen_paths[i][l][4], ")"))
            if l < length(all_gen_paths[i])
                write(f, ", ")
            end
        end
        write(f,"]\"\n")
    end
end