function hsj_obj(new_point::AbstractArray, indic::AbstractArray)
    nrow, ncol = size(new_point)
    for j in 1:ncol, k in 1:nrow
		if new_point[k, j] >= 0.01
			indic[k,j] += 1
		end
    end
    return indic
end

function check_budget_binding(model::Model)
    if dual(model[:budget]) != 0.0
        return true
    else
        return false
    end
end

function print_budgetbind(dists::AbstractVector, outpath::AbstractString)
    file = joinpath(outpath,"MGAbudgetbind.csv")
    its = collect(1:length(dists))
    distdf = DataFrame(Iteration = its, Avg_Dist = dists)
    CSV.write(file,distdf)
end

function print_dists(dists::AbstractVector, outpath::AbstractString)
    file = joinpath(outpath,"MGAdists.csv")
    its = collect(1:length(dists))
    distdf = DataFrame(Iteration = its, Avg_Dist = dists)
    CSV.write(file,distdf)
end

function est_chull_vol(points::AbstractArray)
	(ntt, nz, nit) = size(points)
	clustered = fill(0.0, (ntt, nit))
	clustered = sum(points[:,i,:] for i in 1:nz)
	println(clustered)

	pairs = collect(combinations(1:ntt, 2))
	pairwise_caps = fill(0.0, (nit, 2))
	areas = Vector{Float64}(undef, 0)
	tot_area = 0.0
	for i in eachindex(pairs)
		pairwise_caps[:,1] = view(clustered,pairs[i][1],:)'
		pairwise_caps[:,2] = view(clustered,pairs[i][2],:)'
		uni_pc = uniques(pairwise_caps)
		poly = polyhedron(vrep(Matrix(uni_pc)))
		vol = 0.0
		vol = Polyhedra.volume(poly)
        push!(areas, vol)
	end
	tot_area = sum(areas[i] for i in eachindex(areas))
	println("Volume is: "*string(tot_area))
	return tot_area
end
function uniques(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    placeholder = fill(-1.0,ncol)
    count = 1
    while count < length(placeholder)
        """
        for i in 1:nrow
            if isnan(pointst[i,j])
                pointst[i,j] = 0.0
            end
        end
        """
        if sum(pointst[:,count]) == 0.0
            pointst = pointst[:,1:end .!= count]
            ncol = ncol-1
            pop!(placeholder)
        end
        count += 1
    end
    uniques = fill(-1.0, (nrow, ncol))
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if isapprox(pointst[:,i],uniques[:,k],atol=0.1)
                break
            elseif k == ncol
                counter = counter + 1
                uniques[:,counter] = pointst[:,i]
            end
        end
    end
    uniques = uniques[1:end, 1:counter]
    uniquesT = transpose(uniques)
    println("Done with uniques")
    return uniquesT
end

function hsj_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 1
        # Start MGA Algorithm
	    println("MGA Module")
		println("HSJ Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		# Setup storage
		Indic = fill(0, (length(TechTypes), Z))
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+1))
		point_list[:,:,1] = value.(EP[:vSumvP])
		vols = fill(0.0, setup["ModelingToGenerateAlternativeIterations"])
		"""
	    ### Variables ###

	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type

		
        # Constraint to compute total generation in each zone from a given Technology Type
	    @constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
	    for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
	    ### End Variables ###
		"""

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )


	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_hsj = joinpath(path, "MGAResults_hsj")
	    if !(isdir(outpath_hsj))
	    	mkdir(outpath_hsj)
	    end



	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()

	    print("Starting the first MGA iteration")

	    for i in 1:setup["ModelingToGenerateAlternativeIterations"]

	    	# Create hsj coefficients for the generators that we want to include in the MGA run for the given budget
	    	Indic = hsj_obj(point_list[:,:,i],Indic) #rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])))

	    	### Maximization objective
	    	@objective(EP, Min, sum(Indic[tt,z] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_hsj = joinpath(outpath_hsj, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_hsj, setup, inputs)
			point_list[:,:,i+1] = value.(EP[:vSumvP])
			est_vol = est_chull_vol(point_list[:,:,1:i+1])
			vols[i] = est_vol
	    end

	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_hsj)
	    ### End MGA Iterations ###
	end
	
end





function SPORES_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString, OPTIMIZER)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 8
        # Start MGA Algorithm
	    println("MGA Module")
		println("SPORES Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]
	    
	    

		# Setup storage
		Indic = Vector{Array{Float64, 2}}(undef, 0)
		for i in 1:Threads.nthreads()
		    push!(Indic, fill(0, (length(TechTypes), Z)))
		end
		pSPORES = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),Threads.nthreads()))
        check_it_a_ag!(pRand,setup["ModelingToGenerateAlternativeIterations"])
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+1))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = fill(0.0, setup["ModelingToGenerateAlternativeIterations"])
		a = 0.75
		b = 1-a
		"""
	    ### Variables ###

	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type

		
        # Constraint to compute total generation in each zone from a given Technology Type
	    @constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
	    for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
	    ### End Variables ###
		"""

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )


	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_hsj = joinpath(path, "MGAResults_SPORES")
	    if !(isdir(outpath_hsj))
	    	mkdir(outpath_hsj)
	    end
	    
	    EP_c=Vector{Model}(undef,0)
	    for i in 1:Threads.nthreads()
	        push!(EP_c,copy(EP))
	        set_optimizer(EP_c[i], OPTIMIZER)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()

        
	    print("Starting the first MGA iteration")

	    Threads.@threads :static for i in 1:setup["ModelingToGenerateAlternativeIterations"]
    
	    	# Create hsj coefficients for the generators that we want to include in the MGA run for the given budget
	    	#Indic[k] = hsj_obj(point_list[:,:,i],Indic) #rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])))

	    	### Maximization objective
	    	@objective(EP, Min, sum(a*sum(pSPORES[tt,k]*EP_c[k][:vSumvCap][tt,z] for z in 1:Z) + b*sum(Indic[k][tt,z] * EP_c[k][:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))
            
	    	# Solve Model Iteration
	    	status = optimize!(EP_c[k])

            # Create path for saving MGA iterations
	    	mgaoutpath_hsj = joinpath(outpath_hsj, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c[k], mgaoutpath_hsj, setup, inputs)
			point_list[:,:,i+1] = value.(EP_c[k][:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:i+1])
			vols[i] = est_vol
			
			# Update HSJ obj
			Indic[k] = hsj_obj(point_list[:,:,i+1], Indic[k])
	    end

	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_hsj)
	    ### End MGA Iterations ###
	end
	
end

function heuristic_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 0
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
	    
	    # Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])),setup["ModelingToGenerateAlternativeIterations"])
	    binding = Vector{Bool}(undef, 0)

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)

	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)

	    	### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
			
	    end
	    """
	    
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
	    @expression(EP, eTotEms, sum(EP[:eEmissionsByZone][i,t] for i in 1:Z, t in 1:T))
    	@objective(EP, Min, eTotEms)

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+2))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,2*i+2] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2*i+2])
		push!(vols, est_vol)
	    
	    
	    println(point_list)
        est_vol = est_chull_vol(point_list)
		push!(vols, est_vol)
		"""
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end

function heuristic_combo(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 7
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic/CapMM Combo Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0
		counter = 0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
	    
	    # Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])),Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2)))
	    pBrack = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),setup["ModelingToGenerateAlternativeIterations"]))
        check_it_a_ag!(pBrack,Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2)))
        

	   for i in 1:Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2))
	        #EP_c = copy(EP)
	        #set_optimizer(EP_c, CPLEX.Optimizer)

	    	### Maximization objective
	    	@objective(EP, Max, sum(pRand[tt,z,i] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i])
			#push!(vols, est_vol)

	    	### Minimization objective
	    	@objective(EP, Min, sum(pRand[tt,z,i] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			#push!(vols, est_vol)
			counter += 1
	    end
	    
        for i in counter:setup["ModelingToGenerateAlternativeIterations"]
            
	    	### Maximization objective
	    	@objective(EP, Max, sum(pBrack[tt,i-counter+1] * sum(EP[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i])
			#push!(vols, est_vol)
			
			### Minimization objective
	    	@objective(EP, Min, sum(pBrack[tt,i-counter+1] * sum(EP[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			#push!(vols, est_vol)
	    end

		#println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    #print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function check_it_a_disag!(a::AbstractArray, iterations::Int64)
    (r,c,i) = size(a)
    if iterations < i
        a = a[1:r,1:c,1:iterations]
        return a
    else
        println("Error")
    end
end

function check_it_a_ag!(a::AbstractArray, iterations::Int64)
    (r,i) = size(a)
    if iterations < i
        a = a[1:r,1:iterations]
        return a
    else
        println("Error")
    end
end

function unique_int(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)

    uniques = fill(-2, (nrow, ncol))
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if pointst[:,i]==uniques[:,k]
                break
            elseif k == ncol
                counter = counter + 1
                uniques[:,counter] = pointst[:,i]
            end
        end
    end
    uniques = uniques[1:end, 1:counter]
    uniquesT = transpose(uniques)
    println("Done with uniques")
    return uniquesT
end

function Disag_capminmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 5
        # Start MGA Algorithm
	    println("MGA Module")
		println("Spatially Disaggregated Capacity Min/Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
	    outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
    	pRand = rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])), setup["ModelingToGenerateAlternativeIterations"])

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)
			
			### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2i+2])
		push!(vols, est_vol)
	    """
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function Ag_capminmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 6
        # Start MGA Algorithm
	    println("MGA Module")
		println("Tech Aggregated Capacity Min/Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)
	    println(Least_System_Cost)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")
	    println("G is: $G")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		est_vol = 0.0
		vols = Vector{Float64}(undef, 0)

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###
	   
	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )
	    println(Least_System_Cost*(1+slack))

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
	    outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
        pRand = rand(-1:1,(length(TechTypes),setup["ModelingToGenerateAlternativeIterations"]))
        println(pRand)
        
        # Initialize models 
        	    
    	EP_c = Vector{Model}(undef,Threads.nthreads())
		settings_path = joinpath(path, "Settings")
		OPTIMIZER = configure_solver("gurobi", settings_path)
		for i in 1:length(EP_c)
		    EP_c[i] = copy(EP)
		    set_optimizer(EP_c[i], OPTIMIZER)
		end

        #Initialize Emissions point
        MGCA_Ems = fill(-1.0,(2*setup["ModelingToGenerateAlternativeIterations"]+1, G))
        MGCA_Ems[1,:] = value.(EP[:MGCAEms])'
        
	    println("Starting the first MGA iteration")

	    Threads.@threads :static for i in 1:setup["ModelingToGenerateAlternativeIterations"]#
            k = Threads.threadid()
            println(pRand[:,i])
            println(sum(EP_c[k][:vSumvCap] for z in 1:Z))
            println(TechTypes)
	    	### Maximization objective
	    	@objective(EP_c[k], Max, sum(pRand[tt,i] * sum(EP_c[k][:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c[k])

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c[k], mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[k][:vSumvCap])
			MGCA_Ems[2*i,:] = value.(EP[:MGCAEms])'
			
			### Minimization objective
	    	@objective(EP_c[k], Min, sum(pRand[tt,i] * sum(EP_c[k][:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c[k])

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c[k], mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[k][:vSumvCap])
			MGCA_Ems[2*i+1,:] = value.(EP[:MGCAEms])'
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,2*i+2] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2*i+2])
		push!(vols, est_vol)
	    """
	    total_time = time() - mga_start_time
	    println(MGCA_Ems)
	    println(dfGen[:,:Resource])
	    df_ems = DataFrame(MGCA_Ems, dfGen[:,:Resource])
	    CSV.write(joinpath(path, "Emissions_interp.csv"),df_ems)

	    ### End MGA Iterations ###

	end
end

function Disag_capmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 3
        # Start MGA Algorithm
	    println("MGA Module")
		println("Spatially Disaggregated Capacity Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
    	pRand = rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])), setup["ModelingToGenerateAlternativeIterations"])

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:i])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:i+1])
		push!(vols, est_vol)
		"""
	    
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function Ag_capmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 4
        # Start MGA Algorithm
	    println("MGA Module")
		println("Tech Aggregated Capacity Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
        pRand = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),2*setup["ModelingToGenerateAlternativeIterations"]))
        check_it_a_ag!(pRand,setup["ModelingToGenerateAlternativeIterations"])
        	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,i] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:i])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
	
		est_vol = est_chull_vol(point_list[:,:,1:i+1])
		push!(vols, est_vol)
		"""
	    
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end

function sequential_heuristic_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 2
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+1))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    print("Starting the first MGA iteration")
	    binding = Vector{Bool}(undef,0)

	    for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    	pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])))

	    	### Maximization objective
	    	@objective(EP, Max, sum(pRand[tt,z] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i])
			#push!(vols, est_vol)
			bind = check_budget_binding(EP)
			push!(binding, bind)

	    	### Minimization objective
	    	@objective(EP, Min, sum(pRand[tt,z] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP[:vSumvCap])
			#est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			#push!(vols, est_vol)
			bind = check_budget_binding(EP)
			push!(binding, bind)
	    end
	    println(point_list)
        #est_vol = est_chull_vol(point_list)
		#push!(vols, est_vol)
		#println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    #print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	    println(binding)
	    print_budgetbind(binding, outpath_max)
	end
	
end

function user_specified_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 9
        # Start MGA Algorithm
	    println("MGA Module")
		println("User-Specified Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		#vols = Vector{Float64}(undef, 0)

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    print("Starting the first MGA iteration")
	    binding = Vector{Bool}(undef,0)
	    
	    mgca_df=CSV.read(joinpath(path, "ExportVector.csv"), header=1, DataFrame)
	    mga_caps=CSV.read(joinpath(path, "ExportPoint.csv"), header=1, DataFrame)
	    mga_vec_df = match_tt(mgca_df,TechTypes)
	    mga_vec = Matrix(mga_vec_df)
	    (r,c) = size(mga_vec)
	    println(mga_vec)
	    
        
	    	### Maximization objective
	    	@objective(EP, Max, sum(mga_vec[z,tt] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_max, setup, inputs)

			bind = check_budget_binding(EP)
			push!(binding, bind)

	    	### Minimization objective
	    	@objective(EP, Min, sum(mga_vec[z,tt] * EP[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_min, setup, inputs)


			bind = check_budget_binding(EP)
			push!(binding, bind)


	    total_time = time() - mga_start_time
	    #print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	    println(binding)
	    print_budgetbind(binding, outpath_max)
	end
	
end

function match_point_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 10
        # Start MGA Algorithm
	    println("MGA Module")
		println("Match-Point Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]


	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    #@constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
 
        outpath_min = joinpath(path, "MGAReplicate")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    print("Starting the first MGA iteration")
	    binding = Vector{Bool}(undef,0)
	   
	   
	    println(value.(EP[:vSumvCap]))
	    println(size(EP[:vSumvCap]))
	    mgca_caps_df=CSV.read(joinpath(path, "ExportPointMulti.csv"), header=1, DataFrame)
	    mga_caps = convert_points(mgca_caps_df,Z,TechTypes)
	    if setup["ParameterScale"] == 1
	        mga_caps = mga_caps./ModelScalingFactor
	    end
	    
	    (r,c,its) = size(mga_caps)
	    println(mga_caps)
	    ### Minimization objective
        @objective(EP, Min, EP[:eObj]) # min cost
        
        
	    for i in 1:its
    	    ### Fix zonal cap type values to specified values
    	    for tt in 1:length(TechTypes)
    	        for z in 1:Z
    	            fix(EP[:vSumvCap][tt,z], mga_caps[tt,z,i], force=true)
    	        end
    	    end
    
        	# Solve Model Iteration
        	status = optimize!(EP)
    
            # Create path for saving MGA iterations
        	mgaoutpath_min = joinpath(outpath_min, string("MGA_Fixed_"*string(i)))
    
        	# Write results
        	write_outputs(EP, mgaoutpath_min, setup, inputs)
    	   
    	    total_time = time() - mga_start_time
    	end
	end
end

function match_point_gen_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 11
        # Start MGA Algorithm
	    println("MGA Module")
		println("Match-Point Generator Method")

	    # Objective function value of the least cost problem
	    #Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    #TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    #println("TT is: $TechTypes")
	    #println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    #slack = setup["ModelingtoGenerateAlternativeSlack"]


	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    #@constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
 
        outpath_min = joinpath(path, "MGAReplicate")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    println("Starting the first MGA iteration")
	    println(EP[:eTotalCap])
	    
	    
	    binding = Vector{Bool}(undef,0)
	   
	   
	    
	    mgca_df=CSV.read(joinpath(path, "ExportFullPointMulti.csv"), header=1, DataFrame)
	    mgca_caps_df = mgca_df[:,collect(2:G+1)]
	    println(mgca_caps_df)
	    #mga_caps = convert_points(mgca_caps_df,Z,TechTypes)
	    if setup["ParameterScale"] == 1
	        mgca_caps_df = mgca_caps_df./ModelScalingFactor
	    end
	    
	    println(mgca_caps_df)
	    ### Minimization objective
        @objective(EP, Min, EP[:eObj]) # min cost
        its = nrow(mgca_caps_df)
        mgca_caps_arr = Matrix(mgca_caps_df)
        inputs["solve_time"] = 0.0
        
        
	    for i in 1:its
    	    ### Fix zonal cap type values to specified values
    	    @constraint(EP, cFixing, EP[:eTotalCap] .== mgca_caps_arr[i,:])
    	    println(EP[:cFixing])

    
        	# Solve Model Iteration
        	status = optimize!(EP)
    
            # Create path for saving MGA iterations
        	mgaoutpath_min = joinpath(outpath_min, string("MGA_Fixed_"*string(i)))
            
        	# Write results
        	write_outputs(EP, mgaoutpath_min, setup, inputs)
        	
        	delete.(EP, cFixing)
        	unregister(EP,:cFixing)
    	   
    	    total_time = time() - mga_start_time
    	    
    	
    	    
    	end
	end
end

function check_conflict(EP::Model)
    marker = false
    compute_conflict!(EP)
	list_of_conflicting_constraints = ConstraintRef[]
    for (F, S) in list_of_constraint_types(EP)
        for con in all_constraints(EP, F, S)
            if MOI.get(EP, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                push!(list_of_conflicting_constraints, con)
                marker = true
            end
        end
    end
    if marker == true
        println(list_of_conflicting_constraints)
    end
end

function match_tt(df::DataFrame, TechTypes::Vector)
    new_df = DataFrame()
    for name in TechTypes
        if name in names(df)
            insertcols!(new_df, Symbol(name) => df[:,Symbol(name)]) 
        end            
    end
    return new_df
end

function convert_points(df::DataFrame, Z::Int64, TechTypes::Vector)
    (r,c) = size(df)
    output_arr = Array{Float64,3}(undef,(length(TechTypes),Z,r))
    indx = Vector{Vector{Int64}}(undef,length(TechTypes))
    println(names(df))
    for tt in 1:length(TechTypes)
        indx[tt] = findall(x -> occursin(TechTypes[tt], x), names(df))
        if TechTypes[tt] == "natural_gas" && "natural_gas_CCS" in TechTypes
            deleteat!(indx[tt],2:2:length(indx[tt]))
        end
    end
    println(indx)
    for row in 1:r
        for tt in 1:length(TechTypes)
            for z in 1:Z
                output_arr[tt,z,row] = df[row,indx[tt][z]]
            end
        end
    end
    println(output_arr)
    return output_arr
end



