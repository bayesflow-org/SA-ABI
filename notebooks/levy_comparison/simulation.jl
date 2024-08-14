using Distributions
using AlphaStableDistributions
using Statistics
using StatsFuns
using StatsBase
using PyCall

parent_folder = dirname(dirname(@__DIR__))
source_path = parent_folder * "\\src\\julia"

include("$source_path/01_priors.jl")
include("$source_path/02_diffusion.jl")
include("$source_path/03_experiment.jl")
include("$source_path/04_datasets.jl")

# Generates 8 data sets, expected to take 2.5h
pickle = pyimport("pickle")
goals = ["pretrain", "finetune", "validate", "test"]
power_scalings = [true, false]
n_clusters = 40

for goal in goals
    for power_scaling in power_scalings
        @time begin
            # Settings
            if goal == "pretrain"
                n_datasets = 40000
                n_trials = 100
            end
            
            if goal == "finetune"
                n_datasets = 8000 
                n_trials = 900
            end
            
            if goal == "validate" 
                n_datasets = 100
                n_trials = 900
            end
            
            if goal == "test" 
                n_datasets = 8000
                n_trials = 900
            end

            path = "$parent_folder" * "/data/levy_comparison/sim_" 
            if power_scaling
                path = "$path" * "powerscaled"
            else
                path = "$path" * "unscaled"
            end
            mkpath(path)
            
            # Simulate and save
            sim_data = multi_generative_model(4, n_datasets, n_clusters, n_trials, power_scaling)
            file = open("$path" * "/$goal.pkl", "w")
            pickle.dump(sim_data, file)
            close(file)

            println("Finished $goal with powerscaling being $power_scaling.")
        end
    end
end