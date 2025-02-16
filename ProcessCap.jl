"""
This file takes compiled outputs from the GenX case.

To get compiled outputs, run GenX with MGA, then run mga_consolidate.jl to create a Comp_MGA folder, which will contain capacity, network,
emissions, and costs files.

Load those files into the input here.

Functions: 

main() will compile them into one sheet in the proper format for you. There are options to select different levels of aggregation or lack thereof.

rearrange_columns() will sort generator cluster columns into the proper order for fixing capacities and rerunning an operational model later.

collate_budgets() takes a series of compiled csvs, usually for different budgets, and collates them into one csv.

"""

using DataFrames, CSV
file_num = ""
fold_input() = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\Raw_Data\\Final3z\\EvaluatedPoints" #"C:\\Users\\mike_\\Documents\\ZeroLab\\Local_Files\\MGA_Tests\\GenXMGATests"
cap_input() = joinpath(fold_input(),"Raw_Capacity"*file_num*".csv")
net_input() = joinpath(fold_input(),"Raw_Network"*file_num*".csv")
cost_input() = joinpath(fold_input(),"Raw_Costs"*file_num*".csv")
ems_input() = joinpath(fold_input(),"Raw_Emissions"*file_num*".csv")
fold_output() = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults"
cap_output() = joinpath(fold_output(),"Evaluated_PointsFinal3z"*file_num)

function iter_cap_nogroup(df::DataFrame)
    gdf = groupby(df,[:Resource, :MGA_Iteration], sort=true)
    plants = sort(unique(df.Resource))
    its = unique(df.MGA_Iteration)
    primer = zeros((length(its),length(plants)))
    sum_df = DataFrame(primer, plants)
    counter = 0
    key = keys(gdf)
    for i in eachindex(plants)
        for j in eachindex(its)
            counter += 1
            sum_df[j,i] = sum(gdf[counter].EndCap)
            if i == 2
                #println(sum_df[j,i])
                #println(gdf[counter])
            end
        end
    end
    #println(sum_df)
    return sum_df
end

function iter_cap_grouping(df::DataFrame)
    gdf = groupby(df,[:Type, :MGA_Iteration], sort=true)
    types = sort(unique(df.Type))
    its = unique(df.MGA_Iteration)
    primer = zeros((length(its),length(types)))
    sum_df = DataFrame(primer, types)
    counter = 0
    key = keys(gdf)
    for i in eachindex(types)
        for j in eachindex(its)
            counter += 1
            sum_df[j,i] = sum(gdf[counter].EndCap)
            if i == 2
                #println(sum_df[j,i])
                #println(gdf[counter])
            end
        end
    end
    #println(sum_df)
    return sum_df
end

function iter_disagcap_grouping(df::DataFrame)
    gdf = groupby(df,[:Type, :Zone,:MGA_Iteration], sort=true)#
    types = sort(unique(df.Type))
    println(types)
    zones = sort(unique(df.Zone))
    println(zones)
    its = unique(df.MGA_Iteration)
    primer = zeros((length(its),length(types), length(zones)))
    sum_df = Vector{DataFrame}(undef,length(zones))
    for i in eachindex(zones) 
        sum_df[i] = DataFrame(primer[:,:,i], types)
    end
    counter = 1
    key = keys(gdf)

    
    for i in eachindex(types)    
        for k in eachindex(zones)
            for j in eachindex(its)
                if (k == 3 && (i == 7 || i == 8 || i == 10))
                    break
                else
                    sum_df[k][j,i] = sum(gdf[counter].EndCap)
                    counter += 1
                end
            end
        end
    end
    #println(sum_df)
    return sum_df
end

function add_type(df::DataFrame)
    insertcols!(df,2,:Type => "none")
    rows = nrow(df)
    i = 1
    while i <= rows
        type = "none"
        string_vec = split(df.Resource[i],"_")
        if length(string_vec) < 2
            deleteat!(df, i)
            rows = rows-1
            continue
        end
        if all(c->isuppercase(c),string_vec[2])
            popat!(string_vec,2)
        end
        if string_vec[2] == "conventional" || string_vec[2] == "small"
            type = "hydro"
        elseif string_vec[2] == "natural"
            type = "natural_gas"
        elseif string_vec[2] == "offshorewind" || string_vec[2] == "offshore"
            type = "offshore_wind"
        elseif string_vec[2] == "landbasedwind" || string_vec[2] == "onshore"
            type = "onshore_wind"
        elseif string_vec[2] == "hydroelectric"
            type = "pumped"
        elseif string_vec[2] == "utilitypv"
            type = "solar"
        elseif string_vec[2] == "naturalgas"
            if occursin("ccs", string_vec[3])
                type = "natural_gas_CCS"
            else
                type = "natural_gas"
            end
        else
            type = string_vec[2]
        end
        
        df.Type[i] = type
        i = i + 1 
    end
    return df
end

function add_trans(sum_df::DataFrame, net_df::DataFrame)
    gdf = groupby(net_df,[:MGA_Iteration], sort=true)
    insertcols!(sum_df, :transmission => 0.0)
    its = unique(net_df.MGA_Iteration)
    for i in eachindex(its)
        sum_df.transmission[i] = sum(gdf[i].New_Trans_Capacity)
    end
    return sum_df
end
function add_trans_disag(sum_df::DataFrame, net_df::DataFrame)
    gdf = groupby(net_df,[:MGA_Iteration,:Line], sort=true)
    its = unique(net_df.MGA_Iteration)
    println(its)
    lines = unique(net_df.Line)
    for i in eachindex(lines)
        name = "Line_"*string(i)
        insertcols!(sum_df, Symbol(name) => 0.0)
    end
    for i in its
        for j in eachindex(lines)
            name = "Line_"*string(j)
            sum_df[!,Symbol(name)][i+1] = sum(gdf[(i,j)].New_Trans_Capacity)
        end
    end
    return sum_df
end

function add_cost(sum_df::DataFrame, cost_df::DataFrame)
    gdf = groupby(cost_df,[:MGA_Iteration], sort=true)
    insertcols!(sum_df, :cost => 0.0)#deleted position
    its = unique(cost_df.MGA_Iteration)
    println(its)
    for i in eachindex(its)
        sum_df.cost[i] = sum(gdf[i].Total)
    end
    return sum_df
end

function add_cost_disag(sum_df::DataFrame, cost_df::DataFrame, zone::Int64)
    colname = "Zone"*string(zone)
    println(colname)
    gdf = groupby(cost_df,[:MGA_Iteration], sort=true)
    
    its = unique(cost_df.MGA_Iteration)
    insertcols!(sum_df,1, Symbol("cost_"*string(zone)) => gdf[1][:,Symbol(colname)])
    for i in 2:length(its)
        push!(sum_df[!,Symbol("cost_"*string(zone))],sum(gdf[i][:,Symbol(colname)]))
    end
    return sum_df
end

function add_ems(sum_df::DataFrame, ems_df::DataFrame)
    gdf = groupby(ems_df,[:MGA_Iteration], sort=true)
    insertcols!(sum_df, :emissions => 0.0)#deleted_position
    its = unique(ems_df.MGA_Iteration)
    for i in eachindex(its)
        sum_df.emissions[i] = sum(gdf[i].Total)
    end
    return sum_df
end

function add_ems_disag(sum_df::DataFrame, ems_df::DataFrame, zone::Int64)
    colname = string(zone)
    gdf = groupby(ems_df,[:MGA_Iteration], sort=true)
    insertcols!(sum_df,2, Symbol("emissions_"*string(zone)) => 0.0)
    its = unique(ems_df.MGA_Iteration)
    for i in eachindex(its)
        sum_df[i,Symbol("emissions_"*string(zone))] = sum(gdf[i][:,Symbol(colname)])
    end
    return sum_df
end

function add_iteration(df::DataFrame)
    r = collect(1:nrow(df))
    df[!,:iteration] = r
    return df
end

function create_metric_comparison()
    input_df = CSV.read(joinpath(pwd(),"ExportFullPointMulti.csv"), header=1,DataFrame)
    input_names = names(input_df)
    metrics = ["Column1","cost", "emissions"]
    indx = Vector{Vector{Int64}}(undef,length(metrics))
    # Get indices of metrics
    for i in 1:length(metrics)
        indx[i] = findall(x -> occursin(metrics[i],x), input_names)
    end
    indx = reduce(vcat, indx)
    metric_df = input_df[:, indx]
    mgca_metric_names = ["MGCA_".*names(metric_df)]
    #mgca_name_dict = Dict(key for key in names(metric_df), name for name in mgca_metric_names)
    rename!(metric_df, mgca_name_dict)
    act_metric_names = "Actual_".*names(metric_df)
    metric_names = [mgca_metric_names;act_metric_names]
    output_df = DataFrame(columns=metric_names)
    
    path = joinpath(pwd(),"MGAReplicate")
end

function get_actual(col_names::Vector, path::String)
    fold = readdir(path, join=false)
    fold_path = readdir(path, join=true)
    
    output_df = DataFrame(zeros(length(fold),length(col_names)),columns=col_names)
    indx = Vector{Int64}(undef,length(fold))
    counter = 1
    count2 = 1
    for f in fold
        fp = fold_path[counter]
        substrs = split(f, "_")
        indx[counter] = substrs[3]
        counter += 1
        files = readdir(fp, join = false)
        files_path = readdir(fp, join = true)
        for fi in files
            if occursin("cost", fi) || occursin("emissions", fi)
                
            end
        
            
            count2 += 1
        end
    end
end

function main()
    cap_df = CSV.read(cap_input(), DataFrame)
    net_df = CSV.read(net_input(), DataFrame)
    cost_df = CSV.read(cost_input(), DataFrame)
    ems_df = CSV.read(ems_input(), DataFrame)
    zonal = true
    resource = true
    all_zones = true

    if resource ==true 
        sum_df = iter_cap_nogroup(cap_df)
        sum_df = add_trans_disag(sum_df, net_df)
        sum_df = add_cost(sum_df, cost_df)
        sum_df = add_ems(sum_df,ems_df)
        file_path = cap_output()*"nogroup.csv"
        CSV.write(file_path,sum_df)
    end
    cap_df = add_type(cap_df)
    total_df = iter_cap_grouping(cap_df)
    total_df = add_trans_disag(total_df, net_df)
    total_df = add_cost(total_df, cost_df)
    total_df = add_ems(total_df,ems_df)
    file_path = cap_output()*".csv"
    CSV.write(file_path,total_df)
    total_df = add_iteration(total_df)
    #println(total_df)
    

    if zonal == true && resource == false
        sum_df = iter_disagcap_grouping(cap_df)
        sum_df = add_trans_disag(sum_df, net_df)
        for i in eachindex(sum_df)
            sum_df[i] = add_cost_disag(sum_df[i], cost_df,i)
            sum_df[i] = add_ems_disag(sum_df[i],ems_df,i)
            if all_zones == false
                file_path = cap_output()*"zone"*string(i)*".csv"
                CSV.write(file_path,sum_df[i])
            else
                sum_df[i] = add_iteration(sum_df[i])
                total_df = innerjoin(total_df, sum_df[i], on=:iteration, makeunique=true)
            end
        end
        select!(total_df, Not([:iteration]))
        #println(total_df)
        file_path = cap_output()*"AllZ.csv"
        CSV.write(file_path,total_df)
    elseif resource == true && zonal == true
        sum_df = iter_cap_nogroup(cap_df)
        sum_df = add_trans_disag(sum_df, net_df)
        sum_df = add_cost(sum_df, cost_df)
        sum_df = add_ems(sum_df, ems_df)
        sum_df = add_iteration(sum_df)
        counter = 0
        zone_dfs = []
        for name in names(ems_df)
            if tryparse(Int, name) !== nothing
                counter += 1
                push!(zone_dfs, DataFrame())
                zone_dfs[counter] = add_cost_disag(zone_dfs[counter], cost_df,parse(Int,name))
                println(zone_dfs)
                zone_dfs[counter] = add_ems_disag(zone_dfs[counter],ems_df,parse(Int,name))
                zone_dfs[counter] = add_iteration(zone_dfs[counter])
            end
        end
        println(zone_dfs)
        for i in eachindex(zone_dfs)
            sum_df = innerjoin(sum_df, zone_dfs[i], on=:iteration, makeunique=true)
        end
        select!(sum_df, Not([:iteration]))
        println(sum_df[1,:])
        file_path = cap_output()*"AllZnogroup.csv"
        CSV.write(file_path,sum_df)

    end
    
    
end

#main()


function collate_budgets()
    folder_name = raw"C:\Users\mike_\OneDrive\Documents\PhD\ZERO_Lab\MGCA\CapacityResults"
    file_names = ["Compiled_OutputsFinal3z2.csv","Compiled_OutputsFinal3z4.csv","Compiled_OutputsFinal3z6.csv","Compiled_OutputsFinal3z8.csv","Compiled_OutputsFinal3z10.csv"] #

    output_df = CSV.read(joinpath(folder_name,file_names[1]),DataFrame)

    for i in 2:length(file_names)
        df = CSV.read(joinpath(folder_name, file_names[i]), DataFrame)
        deleteat!(df, 1)
        append!(output_df, df)
    end

    CSV.write(joinpath(folder_name, "Compiled_OutputsFinal3z.csv"),output_df)

end

function rearrange_columns()
    folder_name = raw"C:\Users\mike_\OneDrive\Documents\PhD\ZERO_Lab\MGCA\CapacityResults"
    file = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Evaluated_PointsFinal3zAllZnogroup.csv"#joinpath(folder_name, "Compiled_OutputsFinal3zAllZnogroup.csv")
    gen_data = raw"C:\Users\mike_\OneDrive\Documents\PhD\ZERO_Lab\MGCA\Raw_Data\Final3z\Generators_data.csv"
    main_df = CSV.read(file,DataFrame)
    gen_data_df = CSV.read(gen_data, DataFrame)
    main_df = add_iteration(main_df)

    gen_data_df = gen_data_df[:,[:Resource,:R_ID]]
    new_df = DataFrame([[] for i in gen_data_df.Resource], gen_data_df.Resource)

    cap_df = main_df[:,collect(i for i in 1:nrow(gen_data_df))]
    metric_df = main_df[:,collect(i for i in nrow(gen_data_df)+1:end)]

    append!(new_df, cap_df, cols = :setequal)
    
    new_df = add_iteration(new_df)
    new_df = innerjoin(new_df, metric_df, on=:iteration)
    new_df = select(new_df, Not(:iteration))

    CSV.write(file,new_df)

end
rearrange_columns()
#collate_budgets()
#main()