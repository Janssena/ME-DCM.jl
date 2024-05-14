import Random
import BSON
import CSV
import Lux

include("neural-mixed-effects-paper/new_approach/model.jl");
include("neural-mixed-effects-paper/new_approach/variational_2.jl");
include("src/lib/constraints.jl");

using DataFrames

function skewfit(x)
    opt = Optim.optimize(p -> sum(-logpdf.(SkewNormal(p[1], softplus(p[2]), p[3]), x)), rand(3))
    p_opt = opt.minimizer
    return SkewNormal(p_opt[1], softplus(p_opt[2]), p_opt[3])
end

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
non_zero_relu(x::T) where {T<:Real} = Lux.relu(x) + T(1e-3)

df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/simulation.csv"))
df_group = groupby(df, :id)

indvs = Vector{Individual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:wt, :vwf]])
    y = Vector{Float32}(group.dv[2:end])
    t = Vector{Float32}(group.time[2:end])
    𝐈 = Matrix{Float32}(group[1:1, [:time, :amt, :rate, :duration]])
    callback = generate_dosing_callback(𝐈; S1 = 1/1000.f0)
    indvs[i] = Individual(x, y, t, callback; id = "$(group[1,:id])")
end

population = Population(indvs)
adapt!(population, 2)
prob = ODEProblem(two_comp!, zeros(Float32, 2), (-0.01f0, 1.f0))

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

ann_type = "inn"
# for type in ["FO", "FO-alt", "FOCE1", "FOCE1-slow", "FOCE1-alt", "FOCE2", "FOCE2-slow", "FOCE2-alt", "VI-eta", "VI-eta-single-sample", "VI-full"]
for type in ["VI-eta-single-sample"]
    println("running for $type")
    folder = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/$(ann_type)"
    files = readdir(folder)
    filter!(file -> parse(Int64, split(match(r"replicate_\d+", file).match, "_")[2]) <= 5, files)
    # filter!(file -> contains(file, "fold_1_") || contains(file, "replicate_1"), files)
    result_file = "neural-mixed-effects-paper/new_approach/data/result_$(type)_$(ann_type).csv"
    if isfile(result_file)
        println(" ALREADY DONE, MOVING ON.")
        continue
        result = DataFrame(CSV.File(result_file))
    else
        result = DataFrame(fold = 0, replicate = 0, epoch = 0, rmse_typ = 0.f0, omega_1 = 0.f0, omega_2 = 0.f0, rho = 0.f0, sigma = 0.f0, LL = 0.f0)
    end
    for (j, file) in enumerate(files)
        replicate_idx = parse(Int64, split(file[findfirst(r"replicate_\d+", file)], "_")[2])
        fold_idx = parse(Int64, split(file[findfirst(r"fold_\d+", file)], "_")[2])
        print("Running for file $j, replicate $replicate_idx fold $fold_idx")
        
        ckpt = BSON.load(joinpath(folder, file));
        parameters = ckpt[:saved_parameters]
        train_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(fold_idx).csv")).idxs
        val_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/validation_set_$(fold_idx).csv")).idxs
        test_idxs = (1:500)[Not(unique([train_idxs; val_idxs]))]
        
        indexes = eachindex(parameters)
        epochs = 0:25:((length(indexes) - 1) * 25)

        for i in indexes
            # if any(result.fold .== fold_idx) && any(result.replicate .== replicate_idx) && any(result.epoch .== epochs[i])
            #     println(" ALREADY DONE, MOVING ON.")
            #     continue
            # end
            try
                rmse_typ = sqrt(mse(ckpt[:model], prob, population[test_idxs], ckpt[:state], parameters[i].weights))
            catch
                continue
            end
            rmse_typ = sqrt(mse(ckpt[:model], prob, population[test_idxs], ckpt[:state], parameters[i].weights))

            if contains(type, "full")
                μ = parameters[i].theta[1:4]
                S = softplus.(parameters[i].theta[5:8])
                C = inverse(VecCorrBijector())(parameters[i].theta[9:end])
                Σ = Symmetric(S .* C .* S')
                q = MultivariateNormal(μ, Σ)
                samples = rand(q, 10_000)
                σ = mode(fit(LogNormal, exp.(samples[1, :])))
                ω = zeros(Float32, 2)
                ω[1] = mode(fit(LogNormal, exp.(samples[2, :])))
                ω[2] = mode(fit(LogNormal, exp.(samples[3, :])))
                ρ = mode(skewfit(map(x -> inverse(VecCorrBijector())([x])[2], samples[4, :])))
            else
                if endswith(type, "alt")
                    σ = softplus(parameters[i].theta[1])
                    L = LowerTriangular(vec_to_tril(parameters[i].theta[2:end]))
                    Ω = L * L'
                    ω = sqrt.(diag(Ω))
                    ρ = Ω[2] / prod(ω)
                else
                    # grabs the modes for VI-full
                    σ = softplus(parameters[i].theta[1])
                    ω = softplus.(parameters[i].theta[2:3])
                    C = inverse(VecCorrBijector())(parameters[i].theta[4:end])
                    ρ = C[2]
                end
            end
            append!(result, DataFrame(fold = Integer(fold_idx), replicate = Integer(replicate_idx), epoch = Integer(epochs[i]), rmse_typ = Float32(rmse_typ), omega_1 = Float32(ω[1]), omega_2 = Float32(ω[2]), rho = Float32(ρ), sigma = Float32(σ), LL = ckpt[:LL][i]))
            # append!(result, DataFrame(fold = fold_idx, replicate = replicate_idx, epoch = epochs[i], rmse_typ = rmse_typ, rmse_indv = rmse_indv, omega_1 = ω[1], omega_2 = ω[2], rho = ρ, sigma = σ))
            print(".")
        end
        println(" DONE!")
    end
    sort!(result, ["fold", "replicate", "epoch"])
    CSV.write(result_file, result[result.fold .!= 0, :])
end


##### models trained using MSE:
type = "mse"
folder = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/inn"
files = readdir(folder)
filter!(file -> parse(Int64, split(match(r"replicate_\d+", file).match, "_")[2]) <= 5, files)
result_file = "neural-mixed-effects-paper/new_approach/data/result_$(type)_inn.csv"
if isfile(result_file)
    result = DataFrame(CSV.File(result_file))
else
    result = DataFrame(fold = 0, replicate = 0, epoch = 0, rmse_typ = 0.f0)
end
for (j, file) in enumerate(files)
    print("Running for file $j")
    replicate_idx = parse(Int64, split(file[findfirst(r"replicate_\d+", file)], "_")[2])
    fold_idx = parse(Int64, split(file[findfirst(r"fold_\d+", file)], "_")[2])
    
    ckpt = BSON.parse(joinpath(folder, file));
    delete!(ckpt, :prob);
    ckpt = BSON.raise_recursive(ckpt, Main);
    parameters = :parameters in keys(ckpt) ? ckpt[:parameters] : ckpt[:saved_parameters]
    train_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(fold_idx).csv")).idxs
    val_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/validation_set_$(fold_idx).csv")).idxs
    test_idxs = (1:500)[Not(unique([train_idxs; val_idxs]))]

    indexes = eachindex(parameters)
    if length(parameters) > 102
        indexes = 1:Integer(length(parameters)/25+1)
    end
    epochs = 0:25:((length(indexes) - 1) * 25)

    for i in indexes
        rmse_typ = sqrt(fixed_objective(ckpt[:model], prob, population[test_idxs], ckpt[:state], parameters[i]))
        append!(result, DataFrame(fold = Integer(fold_idx), replicate = Integer(replicate_idx), epoch = Integer(epochs[i]), rmse_typ = Float32(rmse_typ)))
        print(".")
    end
    println(" DONE!")
end
sort!(result, ["fold", "replicate", "epoch"])
CSV.write(result_file, result[result.fold .!= 0, :])