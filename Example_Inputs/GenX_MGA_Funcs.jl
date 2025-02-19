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

function match_point_gen_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 11
        # Start MGA Algorithm
	    println("MGA Module")
		println("Match-Point Generator Method")

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]


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
    	    ### Fix generator cap values to specified values
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