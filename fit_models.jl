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

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
softplusinv(x::T) where T<:Real = log(exp(x) - one(T))

prob = ODEProblem(two_comp!, zeros(Float32, 2), (-0.01f0, 72.f0))

ann = Lux.Chain(
    Normalize([150.f0, 350.f0]),
    Lux.Dense(2, 8, Lux.swish),
    Lux.Dense(8, 8, Lux.swish),
    Lux.Dense(8, 2, Lux.softplus, init_bias=Lux.ones32),
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

inn = Lux.Chain(
    Lux.BranchLayer(
        Lux.Chain(
            Lux.SelectDim(1, 1),
            Lux.ReshapeLayer((1,)),
            Normalize([150.f0]),
            Lux.Dense(1, 12, smooth_relu), # was smooth_relu
            Lux.Parallel(vcat, 
                Lux.Dense(12, 1, Lux.szoftplus, init_bias=Lux.ones32),
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
            )
        ;name="wt"),
        Lux.Chain(
            Lux.SelectDim(1, 2),
            Lux.ReshapeLayer((1,)),
            Normalize([350.f0]),
            Lux.Dense(1, 12, smooth_relu),
            Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
        ;name="vwf")
    ),
    Combine(1 => [1, 2], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

model = ann # [ann, inn]

total_epochs = 2_000
save_every = 25
epochs_to_save = filter(x -> x == 1 || (x % save_every) == 0, 1:total_epochs)
replicate = 1
fold = 1

# for type in ["FO", "VI-eta-single-sample", "VI-eta", "FOCE1", "FOCE2"] #, "FOCE1", "VI-full"]
for type in ["VI-eta-single-sample"] #, "FOCE1", "VI-full"]
    # Select objective function:
    if type == "FO" || type == "FO-alt"
        func = endswith(type, "alt") ? objective_alt : objective
        f(p) = func(FO, model, prob, population[train_idxs], st, p)
    elseif startswith(type, "FOCE1")
        func = endswith(type, "alt") ? objective_alt : objective
        f(p) = func(FOCE1, model, prob, population[train_idxs], st, p)
    elseif startswith(type, "FOCE2")
        func = endswith(type, "alt") ? objective_alt : objective
        f(p) = func(FOCE2, model, prob, population[train_idxs], st, p)
    elseif startswith(type, "VI-eta")
        func = endswith(type, "alt") ? partial_advi_alt : partial_advi
        f(p) = -func(model, prob, population[train_idxs], p, st; num_samples = endswith(type, "single-sample") ? 1 : 3)
    elseif startswith(type, "VI-full")
        if endswith(type, "alt") throw(ErrorException("Not implemented!")) end
        f(p) = -full_advi(model, prob, population[train_idxs], p, st; num_samples = 3)
    end

    for replicate in 1:5
        Threads.@threads for fold in 1:20
            file = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/inn/$(type)_inn_fold_$(fold)_replicate_$(replicate).bson"
            if isfile(file) continue end
            println("Running replicate $(replicate) for fold $fold")
            
            train_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(fold).csv")).idxs
            # initialize parameters:
            ps, st = Lux.setup(Random.default_rng(), model)
            rho = rand(Normal(0., 0.1), 1)
            sigma = softplusinv(rand(truncated(Normal(0.1, 0.025), 0, Inf)))
            omega = softplusinv.(rand(truncated(Normal(0.1, 0.025), 0, Inf), 2))
            if endswith(type, "alt")
                Ω_ = Symmetric(softplus.(omega) .* inverse(VecCorrBijector())(rho) .* softplus.(omega)')
                L_ = _chol_lower(cholesky(Ω_))
                theta_init = Float32[sigma; tril_to_vec(L_)]
            else 
                theta_init = Float32[sigma; omega; rho]
            end

            if startswith(type, "FO")
                parameters = (theta = theta_init, weights = ps)
            else
                if startswith(type, "VI-eta")  # i.e. VI based objective
                    if endswith(type, "alt")
                        var_init = softplus.(ones(2, length(train_idxs)) .* -1)
                        rho_init = rand(Normal(0., 0.1), 1, length(train_idxs))
                        Ω_ = [Symmetric(var_init[:, i] .* inverse(VecCorrBijector())(rho_init[:, i]) .* var_init[:, i]') for i in eachindex(train_idxs)]
                        L_ = _chol_lower.(cholesky.(Ω_))
                        phi_init = Float32.(vcat(zeros(2, length(train_idxs)), hcat(tril_to_vec.(L_)...)))
                    else
                        phi_init = Float32.(vcat(zeros(2, length(train_idxs)), ones(2, length(train_idxs)) .* -1, rand(Normal(0., 0.1), 1, length(train_idxs))))
                    end
                    parameters = (theta = theta_init, phi = phi_init, weights = ps)
                else # VI-full, i.e. add full-rank variance
                    theta_phi = Float32[fill(-1, 4); VecCholeskyBijector(:L)(rand(LKJCholesky(4, 2)))]
                    # theta_init here now is: [4x μ; 4x σ; 6x L_C]
                    parameters = (theta = [theta_init; theta_phi], phi = phi_init, weights = ps)
                end
            end 

            saved_parameters = Vector{typeof(parameters)}(undef, length(epochs_to_save))
            # opt = Optimisers.ADAM(0.1f0)
            opt = Optimisers.ADAM(contains(type, "slow") ? 0.01f0 : 0.1f0) # Slower learning rate
            # opt = Optimisers.ADAM(contains(type, "slower") ? 0.001f0 : 0.1f0) # Slower learning rate
            # We should run all models with lower training rates.
            opt_state = Optimisers.setup(opt, parameters)
            losses = zeros(total_epochs)
            LL = zeros(Float32, length(epochs_to_save))

            if startswith(type, "FOCE")
                [indv.eta .= zero(eltype(indv.eta)) for indv in population[train_idxs]]
            end

            start_time = now()
            try
                for epoch in 1:total_epochs
                    if startswith(type, "FOCE")
                        if endswith(type, "alt")
                            optimize_etas_alt!(population[train_idxs], prob, model, st, parameters)
                        else
                            optimize_etas!(population[train_idxs], prob, model, st, parameters, Optim.BFGS())
                            # display(Plots.histogram(hcat([indv.eta for indv in population[train_idxs]]...)', layout=2))
                        end
                    end

                    loss, back = Zygote.pullback(f, parameters)
                    ∇ = first(back(1))

                    losses[epoch] = loss
                    if epoch == 1 || epoch % (startswith(type, "FOCE") ? 100 : 250) == 0
                        println("($(type), fold $fold: replicate $replicate) Epoch $epoch: loss = $(loss)")
                    end

                    opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
                    if epoch in epochs_to_save
                        LL[Integer(round(epoch / save_every)) + 1] = Float32(loss)
                        if startswith(type, "FO")
                            saved_parameters[Integer(round(epoch / save_every)) + 1] = (theta = copy(parameters.theta), weights = Lux.fmap(copy, parameters.weights))
                        else
                            saved_parameters[Integer(round(epoch / save_every)) + 1] = (theta = copy(parameters.theta), phi = copy(parameters.phi), weights = Lux.fmap(copy, parameters.weights))
                        end
                    end
                end
            catch e
                println("($(type), fold $fold: replicate $replicate): Epoch $(findfirst(x -> x == 0, losses)-1): Stopped early due to error:\n$e")
            end
            end_time = now()
            last_epoch_idx = any(LL .== 0) ? findfirst(x -> x == 0., LL) - 1 : length(LL)
            
            BSON.bson(file, 
                Dict(
                    :prob => :two_comp,
                    :model => model, 
                    :state => st,
                    :saved_parameters => saved_parameters[1:last_epoch_idx], 
                    :LL => LL[1:last_epoch_idx],
                    :opt => opt, 
                    :opt_state => opt_state, # To allow continuation of training.
                    :duration => end_time - start_time,
                    :last_parameters => parameters # Check what parameters resulted in error
                )
            )
            
            if startswith(type, "FOCE")
                [indv.eta .= zero(eltype(indv.eta)) for indv in population[train_idxs]]
            end
        end
    end
end



# Fitting mse models:
type = "mse"
for replicate in 1:5
    Threads.@threads for fold in 1:20
        file = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/inn/$(type)_inn_fold_$(fold)_replicate_$(replicate).bson"
        if isfile(file) continue end
        println("Running replicate $(replicate) for fold $fold")
        
        train_idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(fold).csv")).idxs
        # initialize parameters:
        ps, st = Lux.setup(Random.default_rng(), model)

        parameters = (weights = ps,)
        saved_parameters = Vector{typeof(parameters)}(undef, length(epochs_to_save))
        opt = Optimisers.ADAM(0.1f0)
        opt_state = Optimisers.setup(opt, parameters)
        losses = zeros(total_epochs)
        mse = zeros(Float32, length(epochs_to_save))

        start_time = now()
        for epoch in 1:total_epochs
            loss, back = Zygote.pullback(p -> fixed_objective(model, prob, population[train_idxs], st, p), parameters)
            ∇ = first(back(1))

            losses[epoch] = loss
            if epoch == 1 || epoch % 250 == 0
                println("($(type), fold $fold: replicate $replicate) Epoch $epoch: loss = $(loss)")
            end

            opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
            if epoch in epochs_to_save
                mse[Integer(round(epoch / save_every)) + 1] = Float32(loss)
                saved_parameters[Integer(round(epoch / save_every)) + 1] = (weights = Lux.fmap(copy, parameters.weights),)
            end
        end
        end_time = now()
        
        BSON.bson(file, 
            Dict(
                :prob => :two_comp,
                :model => model, 
                :state => st,
                :saved_parameters => saved_parameters, 
                :mse => mse,
                :opt => opt, 
                :opt_state => opt_state, # To allow continuation of training.
                :duration => end_time - start_time,
            )
        )
    end
end