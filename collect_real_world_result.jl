import Optimisers
import Random
import BSON
import CSV
import Lux

include("neural-mixed-effects-paper/new_approach/model.jl");
include("neural-mixed-effects-paper/new_approach/variational_2.jl");
include("src/lib/constraints.jl");

using Dates
using DataFrames

df1 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df1_prophylaxis_imputed.csv"))
df2 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df2_surgery_imputed.csv"))

dfs = [df1, df2, df3]

df_idx = 2
df = dfs[df_idx]
grouper = [:SubjectID, :SubjectID, [:ID, :OCC]][df_idx]
df_group = groupby(df, grouper)

indvs = Vector{Individual}(undef, length(df_group))
for (j, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:FFM, :VWF_final]])
    y = Vector{Float32}(group[group.MDV .== 0, :DV] .- group[.!ismissing.(group.Baseline), :Baseline][1]) 
    t = Vector{Float32}(group[group.MDV .== 0, :Time])
    𝐈 = Matrix{Float32}(group[group.MDV .== 1, [:Time, :Dose, :Rate, :Duration]])
    callback = generate_dosing_callback(𝐈; S1 = 1/1000.f0)
    indvs[j] = Individual(x, y, t, callback; id = "$(df_idx == 3 ? "$(group.ID[1])_$(group.OCC[1]))" : group[1, grouper])")
end

population = Population(filter(indv -> !isempty(indv.y), indvs))
adapt!(population, 2)

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
softplusinv(x::T) where T<:Real = log(exp(x) - one(T))

prob = ODEProblem(two_comp!, zeros(Float32, 2), (-0.01f0, 72.f0))

function mse(ann, prob, population::Population, st::NamedTuple, p::NamedTuple)
    ζ, _ = Lux.apply(ann, population.x, p, st)
    ζ = max.(ζ, 0.001f0) # prevent errors with PK parameters close to 0
    SSE = zero(Float32)
    k = 0
    for i in eachindex(population)
        individual = population[i]
        ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζ[:, i], individual.eta)
        SSE += sum(abs2, individual.y - ŷ)
        k += length(individual.ty)
    end
    return SSE / k
end

error = :prop # [:add, :prop]
ann_type = "inn"
for type in ["VI"]
    println("running for $type")
    name = "$(type == "FO" ? "fo" : (type == "VI" ? "vi" : "foce"))"
    folder = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$df_idx/$name"
    files = readdir(folder)
    filter!(file -> contains(file, "prop") == (error == :prop), files)
    # filter!(file -> contains(file, "fold_1_") || contains(file, "replicate_1"), files)
    result_file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(type)_$(ann_type)_$(String(error)).csv"
    if isfile(result_file)
        println(" ALREADY DONE, MOVING ON.")
        continue
        result = DataFrame(CSV.File(result_file))
    else
        if error == :prop
            result = DataFrame(outer = 0, inner = 0, epoch = 0, rmse_typ = 0.f0, omega_1 = 0.f0, omega_2 = 0.f0, rho = 0.f0, sigma_1 = 0.f0, sigma_2 = 0.f0, LL = 0.f0)
        else
            result = DataFrame(outer = 0, inner = 0, epoch = 0, rmse_typ = 0.f0, omega_1 = 0.f0, omega_2 = 0.f0, rho = 0.f0, sigma = 0.f0, LL = 0.f0)
        end
    end
    for (j, file) in enumerate(files)
        outer_idx = parse(Int64, split(file[findfirst(r"outer_\d+", file)], "_")[2])
        inner_idx = parse(Int64, split(file[findfirst(r"inner_\d+", file)], "_")[2])
        print("Running for file $j, outer fold $outer_idx inner $inner_idx")
        cv = BSON.load("neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$df_idx.bson")
        population_ = population[cv[:outer][outer_idx]] # Reshuffle the population so that data fold
        
        ckpt = BSON.load(joinpath(folder, file));
        parameters = ckpt[:saved_parameters]
        train_idxs = cv[:inner_train][inner_idx]
        test_idxs = cv[:inner_test][inner_idx]
        
        indexes = eachindex(parameters)
        epochs = 0:25:((length(indexes) - 1) * 25)

        for i in indexes
            rmse_typ = sqrt(mse(ckpt[:model], prob, population_[test_idxs], ckpt[:state], parameters[i].weights))
            if error == :prop
                σ = softplus.(parameters[i].theta[1:2])
                ω = softplus.(parameters[i].theta[3:4])
                C = inverse(VecCorrBijector())(parameters[i].theta[5:end])
            else
                σ = softplus(parameters[i].theta[1])
                ω = softplus.(parameters[i].theta[2:3])
                C = inverse(VecCorrBijector())(parameters[i].theta[4:end])
            end
            ρ = C[2]
            if error == :prop
                append!(result, DataFrame(outer = Integer(outer_idx), inner = Integer(inner_idx), epoch = Integer(epochs[i]), rmse_typ = Float32(rmse_typ), omega_1 = Float32(ω[1]), omega_2 = Float32(ω[2]), rho = Float32(ρ), sigma_1 = Float32(σ[1]), sigma_2 = Float32(σ[2]), LL = ckpt[:LL][i]))
            else
                append!(result, DataFrame(outer = Integer(outer_idx), inner = Integer(inner_idx), epoch = Integer(epochs[i]), rmse_typ = Float32(rmse_typ), omega_1 = Float32(ω[1]), omega_2 = Float32(ω[2]), rho = Float32(ρ), sigma = Float32.(σ), LL = ckpt[:LL][i]))
            end
            print(".")
        end
        println(" DONE!")
    end
    sort!(result, ["outer", "inner", "epoch"])
    CSV.write(result_file, result[result.outer .!= 0, :])
end



# For MSE:
type = "mse"
folder = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$df_idx/mse"
files = readdir(folder)
result_file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(type)_$(ann_type).csv"
if !isfile(result_file)
    result = DataFrame(outer = 0, inner = 0, epoch = 0, mse = 0., rmse_typ = 0.f0)
    for (j, file) in enumerate(files)
        outer_idx = parse(Int64, split(file[findfirst(r"outer_\d+", file)], "_")[2])
        inner_idx = parse(Int64, split(file[findfirst(r"inner_\d+", file)], "_")[2])
        print("Running for file $j, outer fold $outer_idx inner $inner_idx")
        cv = BSON.load("neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$df_idx.bson")
        population_ = population[cv[:outer][outer_idx]] # Reshuffle the population so that data fold
        
        ckpt = BSON.load(joinpath(folder, file));
        parameters = ckpt[:saved_parameters]
        train_idxs = cv[:inner_train][inner_idx]
        test_idxs = cv[:inner_test][inner_idx]
        
        indexes = eachindex(parameters)
        epochs = 0:25:((length(indexes) - 1) * 25)

        for i in indexes
            mse_typ = mse(ckpt[:model], prob, population_[train_idxs], ckpt[:state], parameters[i].weights)
            rmse_typ = sqrt(mse(ckpt[:model], prob, population_[test_idxs], ckpt[:state], parameters[i].weights))
            
            append!(result, DataFrame(outer = Integer(outer_idx), inner = Integer(inner_idx), epoch = Integer(epochs[i]), mse = mse_typ, rmse_typ = rmse_typ))
            print(".")
        end
        println(" DONE!")
    end
    sort!(result, ["outer", "inner", "epoch"])
    CSV.write(result_file, result[result.outer .!= 0, :])
end