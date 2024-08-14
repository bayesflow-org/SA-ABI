using Test
using Distributions
using AlphaStableDistributions
using Random

# Include your files if necessary
include("01_priors.jl")
include("02_diffusion.jl")
include("03_experiment.jl")
include("04_datasets.jl")


# Test generate_scaling_factors function
@testset "generate_scaling_factors" begin
    function test_generate_scaling_factors()
        test_factor_bounds = [
            [0.1, 10.0],
            [0.5, 2.0]
        ]
        
        num_iterations = 1000
        tolerance = num_iterations / 10
        
        for bounds in test_factor_bounds
            lower_bound = bounds[1]
            upper_bound = bounds[2]
            below_1_count = 0
            above_1_count = 0
            
            for _ in 1:num_iterations
                scaling_factors = generate_scaling_factors(lower_bound, upper_bound)

                @test length(scaling_factors) == 2
                @test all(lower_bound .<= scaling_factors .<= upper_bound)
                
                # Check whether the values are correctly transformed to log space and back
                @test isapprox(exp10(log10(scaling_factors[1])), scaling_factors[1], atol=1e-6)
                @test isapprox(exp10(log10(scaling_factors[2])), scaling_factors[2], atol=1e-6)

                # Count the values below and above 1 in each iteration
                below_1_count += sum(scaling_factors .< 1.0)
                above_1_count += sum(scaling_factors .> 1.0)
            end
            
            # Check whether the total count of values below 1 is roughly equal to above 1
            @test abs(below_1_count - above_1_count) <= tolerance
        end   
    end
    test_generate_scaling_factors()
end


# Test generate_levy_trial function
@testset "generate_levy_trial" begin
    function test_generate_levy_trial()
        a_l = 1.0
        zr_l = 0.5
        v_l = 2.0
        t0_l = 0.5
        alpha_l = 1.5
        sz = 0.3
        sv = 1.6
        st = 0.2
        rt, decision = generate_levy_trial(a_l, zr_l, v_l, t0_l, alpha_l, sz, sv, st)
        @test rt >= 0.0
        @test decision == 0.0 || decision == 1.0
    end

    test_generate_levy_trial()
end

# Test generate_levy_participant function
@testset "generate_levy_participant" begin
    function test_generate_levy_participant()
        n_trials = 900
        a_l = 1.0
        zr_l = 0.5
        v0_l = -2.0
        v1_l = 2.0
        t0_l = 0.5
        alpha_l = 1.5
        sz = 0.3
        sv = 1.6
        st = 0.2

        data = generate_levy_participant(n_trials, a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st)
        @test size(data) == (n_trials, 3)
        @test all(data[:, 3] .== 0.0 .|| data[:, 3] .== 1.0)  # Check decision outcome validity
    end

    test_generate_levy_participant()
end

# Test generate_levy_hyperpriors function
@testset "generate_levy_hyperpriors" begin
    function test_generate_levy_hyperpriors()
        test_factor_bounds = [
            [0.1, 10.0],
            [0.5, 2.0]
        ]

        # Test with powerscaling
        for bounds in test_factor_bounds
            lower_bound = bounds[1]
            upper_bound = bounds[2]
            scaling_factors = generate_scaling_factors(lower_bound, upper_bound)
            hyperpriors = generate_levy_hyperpriors(scaling_factors)
            @test typeof(hyperpriors) == Vector{Float64}
            @test !isempty(hyperpriors)
            @test length(hyperpriors) == 12
        end

        # Test without powerscaling
        scaling_factors = [1.0, 1.0]
        hyperpriors = generate_levy_hyperpriors(scaling_factors)
        @test typeof(hyperpriors) == Vector{Float64}
        @test !isempty(hyperpriors)
        @test length(hyperpriors) == 12
    end

    test_generate_levy_hyperpriors()
end

# Test generate_levy_priors function
@testset "generate_levy_priors" begin
    function test_generate_levy_priors()
        test_factors = [
            [0.1, 10.0],
            [1.0, 1.0]
        ]
        for factors in test_factors
            h_test = generate_levy_hyperpriors(factors)
            priors = generate_levy_priors(h_test...)
            @test length(priors) == 9
        end
    end
    test_generate_levy_priors()
end

# Test generate_levy_dataset_by_model function
@testset "generate_levy_dataset_by_model" begin
    function test_generate_levy_dataset_by_model()
        n_clusters = 40
        n_trials = 900
        power_scalings = [true, false]
        for power_scaling in power_scalings
            for model in 1:4
                scaling_factors, data = generate_levy_dataset_by_model(model, n_clusters, n_trials, power_scaling)
                if power_scaling
                    @test any(scaling_factors .!= 1.0)
                else
                    @test all(scaling_factors .== 1.0)
                end
                @test size(data) == (n_clusters, n_trials, 3)
                @test all(data[:, :, 3] .== 0.0 .|| data[:, :, 3] .== 1.0)  # Check decision outcome validity
            end
        end
    end

    test_generate_levy_dataset_by_model()
end

# Test generate_levy_batch function
@testset "generate_levy_batch" begin
    function test_generate_levy_batch()
        for model in 1:4
            batch_size = 2
            n_clusters = 40
            n_trials = 900
            power_scalings = [true, false]
            for power_scaling in power_scalings
                scaling_factors, index_list, data_batch = generate_levy_batch(model, batch_size, n_clusters, n_trials, power_scaling)
                if power_scaling
                    @test any(scaling_factors .!= 1.0)
                else
                    @test all(scaling_factors .== 1.0)
                end
                @test length(index_list) == batch_size
                @test size(data_batch) == (batch_size, n_clusters, n_trials, 3)
                @test all(data_batch[:, :, :, 3] .== 0.0 .|| data_batch[:, :, :, 3] .== 1.0)  # Check decision outcome validity
            end
        end
    end

    test_generate_levy_batch()
end

@testset "multi_generative_model" begin
    function test_multi_generative_model()
        num_models = 4
        batch_size = 16
        n_clusters = 40
        n_trials = 900
        power_scalings = [true, false]

        for power_scaling in power_scalings
            out_dict = multi_generative_model(num_models, batch_size, n_clusters, n_trials, power_scaling)
            @test haskey(out_dict, "model_outputs")
            @test haskey(out_dict, "model_indices")
            @test length(out_dict["model_outputs"]) == num_models
            @test length(out_dict["model_indices"]) == num_models

            for (idx, model_output) in enumerate(out_dict["model_outputs"])
                @test haskey(model_output, "prior_batchable_context")
                @test haskey(model_output, "sim_data")
                if power_scaling
                    @test any(model_output["prior_batchable_context"] .!= 1.0)
                else
                    @test all(model_output["prior_batchable_context"] .== 1.0)
                end
                @test size(model_output["sim_data"]) == (batch_size ÷ num_models, n_clusters, n_trials, 3)
                @test all(model_output["sim_data"][:, :, :, 3] .== 0.0 .|| model_output["sim_data"][:, :, :, 3] .== 1.0)  # Check decision outcome validity
            end
        end
    end

    test_multi_generative_model()
end

println("All tests passed successfully!");
