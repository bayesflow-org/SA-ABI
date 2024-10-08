"Generates a reaction time dataset according to the four models in Wieschen, Voss and Radev (2020)."
function generate_levy_dataset_by_model(model::Int, n_clusters::Int64, n_trials::Int64, power_scaling::Bool)::Tuple{Vector{Float64}, Array{Float64, 3}}

    if power_scaling == true
        scaling_factors = generate_scaling_factors(0.1, 10.0) # Power scaling bounds!
    elseif power_scaling == false
        scaling_factors = [1.0, 1.0] # no scaling applied
    end
    data = fill(0.0, (n_clusters, n_trials, 3))
    hyperpriors = generate_levy_hyperpriors(scaling_factors)

    for k in 1:n_clusters
        a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st = generate_levy_priors(hyperpriors...)
 
        if model == 1 || model == 3 # Gaussian noise
            alpha_l = 2.0
        end

        if model == 1 || model == 2 # Basic instead of full diffusion model
            sz = sv = st = 0.0 
        end

        data[k, :, :] = generate_levy_participant(n_trials, a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st)
    end

    return scaling_factors, data
end

"""
Generates a batch of datasets simulated from a given model.
Uses Threads.@threads for parallelization - assure that number of threads Julia uses is set accordingly (e.g., in the VSCode Julia extension).
Check with println(Threads.nthreads()) or println(ENV["JULIA_NUM_THREADS"]).
"""
function generate_levy_batch(model::Int64, batch_size::Int64, n_clusters::Int64, n_trials::Int64, power_scaling::Bool)::Tuple{Array{Float64, 2}, Vector{Int64}, Array{Float64, 4}}

    scaling_factors = fill(0.0, (batch_size, 2))
    index_list = fill(model, batch_size)
    data = fill(0.0, (batch_size, n_clusters, n_trials, 3))

    Threads.@threads for b in 1:batch_size
        scaling_factors[b, :], data[b, :, :, :] = generate_levy_dataset_by_model(model, n_clusters, n_trials, power_scaling)
    end

    return scaling_factors, index_list, data
end

"""
Generates a batch of datasets simulated from all models, in BayesFlow dict format. 
Uses a flat model prior (equal number of data sets per model).
"""
function multi_generative_model(num_models::Int64, batch_size::Int64, n_clusters::Int64, n_trials::Int64, power_scaling::Bool)
    # Check if batch_size is divisible by num_models
    if batch_size % num_models != 0
        throw(ArgumentError("batch_size must be divisible by num_models"))
    end

    sims_per_model = Int(batch_size / num_models)
    out_dict = Dict("model_outputs" => [], "model_indices" => [])

    for m_idx in 1:num_models
        scaling_factors, index_list, data = generate_levy_batch(m_idx, sims_per_model, n_clusters, n_trials, power_scaling)
        m_idx_outputs = Dict("prior_batchable_context" => scaling_factors, "sim_data" => data)

        push!(out_dict["model_outputs"], m_idx_outputs)
        push!(out_dict["model_indices"], m_idx - 1)
    end

    return out_dict
end