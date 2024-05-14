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
df3 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df3_surgery_imputed.csv"))
# df3 = df3_[df3_.OCC .== 1, :] # Only use data from occassion 1

dfs = [df1, df2, df3]

i = 2
df = dfs[i]
grouper = [:SubjectID, :SubjectID, [:ID, :OCC]][i]
df_group = groupby(df, grouper)

"""df3 should later on also be divided up into sub groups based on OCC?"""

indvs = Vector{Individual}(undef, length(df_group))
for (j, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:FFM, :VWF_final]])
    y = Vector{Float32}(group[group.MDV .== 0, :DV] .- group[.!ismissing.(group.Baseline), :Baseline][1]) 
    t = Vector{Float32}(group[group.MDV .== 0, :Time])
    𝐈 = Matrix{Float32}(group[group.MDV .== 1, [:Time, :Dose, :Rate, :Duration]])
    callback = generate_dosing_callback(𝐈; S1 = 1/1000.f0)
    indvs[j] = Individual(x, y, t, callback; id = "$(i == 3 ? "$(group.ID[1])_$(group.OCC[1]))" : group[1, grouper])")
end

population = Population(filter(indv -> !isempty(indv.y), indvs))
adapt!(population, 2)

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
softplusinv(x::T) where T<:Real = log(exp(x) - one(T))

prob = ODEProblem(two_comp!, zeros(Float32, 2), (-0.01f0, 72.f0))

model = Lux.Chain(
    Normalize([150.f0, 3.5f0]),
    Lux.BranchLayer(
        Lux.Chain(
            Lux.SelectDim(1, 1),
            Lux.ReshapeLayer((1,)),
            Lux.Dense(1, 12, smooth_relu), # was smooth_relu
            Lux.Parallel(vcat, 
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32),
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
            )
        ;name="wt"),
        Lux.Chain(
            Lux.SelectDim(1, 2),
            Lux.ReshapeLayer((1,)),
            Lux.Dense(1, 12, smooth_relu),
            Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
        ;name="vwf")
    ),
    Combine(1 => [1, 2], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

method = :VI # :FO, :FOCE or :VI
total_epochs = 1_250
save_every = 25
epochs_to_save = filter(x -> x == 1 || (x % save_every) == 0, 1:total_epochs)

for outer_fold in 1:10
    cv = BSON.load("neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$i.bson")
    population_ = population[cv[:outer][outer_fold]] # Reshuffle the population so that data folds are re-aligned
    # We can then use the same inner fold indexes
    
    Threads.@threads for inner_fold in 1:10
        name = "$(method == :FO ? "fo" : (method == :VI ? "vi" : "foce"))"
        # file = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$i/$(name)/outer_$(outer_fold)_inner_$(inner_fold).bson"
        file = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$i/$(name)/prop_error_outer_$(outer_fold)_inner_$(inner_fold).bson"
        if isfile(file) continue end
        println("Running model for outer_fold $(outer_fold) and inner fold $inner_fold")
        train_idxs = cv[:inner_train][inner_fold]
        ps, st = Lux.setup(Random.default_rng(), model)
        
        # obj_fn(p) = objective(FO, model, prob, population_[train_idxs], st, p) # Additive error
        # obj_fn(p) = FO_objective_prop(model, prob, population_[train_idxs], p, st) # Proportional error
        # obj_fn(p) = FOCE2_objective_prop(model, prob, population_[train_idxs], p, st) # Combined error
        # obj_fn(p) = -partial_advi(model, prob, population_[train_idxs], p, st; num_samples = 3) # Additive error
        obj_fn(p) = -partial_advi_prop(model, prob, population_[train_idxs], p, st; num_samples = 3)

        rho = rand(Normal(0., 0.1), 1)
        # sigma = softplusinv(rand(truncated(Normal(0.1, 0.025), 0, Inf))) # Additive error
        sigma = softplusinv.(rand(truncated(Normal(0.1, 0.025), 0, Inf), 2)) # Combined error
        omega = softplusinv.(rand(truncated(Normal(0.1, 0.025), 0, Inf), 2))
        theta_init = Float32[sigma; omega; rho]
        phi_init = Float32.(vcat(zeros(2, length(train_idxs)), ones(2, length(train_idxs)) .* -1, rand(Normal(0., 0.1), 1, length(train_idxs))))
        if method == :FO || method == :FOCE
            parameters = (theta = theta_init, weights = ps)
        else
            parameters = (theta = theta_init, phi = phi_init, weights = ps)
        end

        saved_parameters = Vector{typeof(parameters)}(undef, length(epochs_to_save))
        opt = Optimisers.ADAM(method == :FOCE || contains(name, "slow") ? 0.01f0 : 0.1f0)
        opt_state = Optimisers.setup(opt, parameters)
        losses = zeros(total_epochs)
        LL = zeros(Float32, length(epochs_to_save))
        start_time = now()

        if method == :FOCE
            [indv.eta .= 0 for indv in population_]
        end

        for epoch in 1:total_epochs
            if method == :FOCE
                optimize_etas!(population_[train_idxs], prob, model, st, parameters, Optim.BFGS())
            end

            loss, back = Zygote.pullback(obj_fn, parameters)
            ∇ = first(back(1))
            # Some initializations lead to initial NaN gradients for FO, not sure why. 
            while epoch == 1 && method == :FO && any(isnan.(∇.theta))
                rho = rand(Normal(0., 0.1), 1)
                # sigma = softplusinv(rand(truncated(Normal(0.1, 0.025), 0, Inf))) # Additive error
                sigma = softplusinv.(rand(truncated(Normal(0.1, 0.025), 0, Inf), 2)) # Combined error
                omega = softplusinv.(rand(truncated(Normal(0.1, 0.025), 0, Inf), 2))
                theta_init = Float32[sigma; omega; rho]
                phi_init = Float32.(vcat(zeros(2, length(train_idxs)), ones(2, length(train_idxs)) .* -1, rand(Normal(0., 0.1), 1, length(train_idxs))))
                parameters = (theta = theta_init, weights = ps)
                loss, back = Zygote.pullback(obj_fn, parameters)
                ∇ = first(back(1))
            end

            losses[epoch] = loss
            if epoch == 1 || epoch % 250 == 0
                println("(Outer $(outer_fold), inner $(inner_fold)) Epoch $(epoch): loss = $(loss)")
            end

            opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
            if epoch in epochs_to_save
                LL[Integer(round(epoch / save_every)) + 1] = Float32(loss)
                if method == :FO || method == :FOCE
                    saved_parameters[Integer(round(epoch / save_every)) + 1] = (theta = copy(parameters.theta), weights = Lux.fmap(copy, parameters.weights))
                else
                    saved_parameters[Integer(round(epoch / save_every)) + 1] = (theta = copy(parameters.theta), phi = copy(parameters.phi), weights = Lux.fmap(copy, parameters.weights))
                end
            end
        end
        end_time = now()
        
        BSON.bson(file, 
            Dict(
                :prob => :two_comp,
                :model => model, 
                :state => st,
                :saved_parameters => saved_parameters, 
                :LL => LL,
                :opt => opt, 
                :opt_state => opt_state, # To allow continuation of training.
                :duration => end_time - start_time,
                :last_parameters => parameters # Check what parameters resulted in error
            )
        )

        if method == :FOCE
            [indv.eta .= 0 for indv in population_]
        end
    end
end


# Continue training:
method = :VI # :FO, :FOCE or :VI
total_epochs = 1_250
save_every = 25
epochs_to_save = filter(x -> x == 1 || (x % save_every) == 0, 1:total_epochs)

for outer_fold in 1:10
    cv = BSON.load("neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$i.bson")
    population_ = population[cv[:outer][outer_fold]] # Reshuffle the population so that data folds are re-aligned
    # We can then use the same inner fold indexes
    
    Threads.@threads for inner_fold in 1:10
        train_idxs = cv[:inner_train][inner_fold]
        name = "$(method == :FO ? "fo" : (method == :VI ? "vi" : "foce"))"
        # file = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$i/$(name)/outer_$(outer_fold)_inner_$(inner_fold).bson"
        file = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$i/$(name)/prop_error_outer_$(outer_fold)_inner_$(inner_fold).bson"
        ckpt = BSON.load(file)
        if haskey(ckpt, :duration_second)
            continue 
        end    
        println("Continuing training for outer_fold $(outer_fold) and inner fold $inner_fold")
        # obj_fn(p) = objective(FO, model, prob, population_[train_idxs], st, p) # Additive error
        # obj_fn(p) = FO_objective_prop(model, prob, population_[train_idxs], p, st) # Proportional error
        # obj_fn(p) = FOCE2_objective_prop(model, prob, population_[train_idxs], p, st) # Combined error
        # obj_fn(p) = -partial_advi(model, prob, population_[train_idxs], p, st; num_samples = 3) # Additive error
        obj_fn(p) = -partial_advi_prop(model, prob, population_[train_idxs], p, st; num_samples = 3)
        parameters = ckpt[:saved_parameters][end]
        st = ckpt[:state]
        saved_parameters = ckpt[:saved_parameters]

        opt = ckpt[:opt]
        opt_state = ckpt[:opt_state]
        LL = ckpt[:LL]
        
        if method == :FOCE
            [indv.eta .= 0 for indv in population_]
        end
        
        start_time = now()
        for epoch in (1001:total_epochs)
            if method == :FOCE
                optimize_etas!(population_[train_idxs], prob, model, st, parameters, Optim.BFGS())
            end

            loss, back = Zygote.pullback(obj_fn, parameters)
            ∇ = first(back(1))

            if epoch == 1001 || epoch % 50 == 0
                println("(Outer $(outer_fold), inner $(inner_fold)) Epoch $(epoch): loss = $(loss)")
            end

            opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
            if epoch in epochs_to_save
                push!(LL, Float32(loss))
                if method == :FO || method == :FOCE
                    push!(saved_parameters, (theta = copy(parameters.theta), weights = Lux.fmap(copy, parameters.weights))) 
                else
                    push!(saved_parameters, (theta = copy(parameters.theta), phi = copy(parameters.phi), weights = Lux.fmap(copy, parameters.weights)))
                end
            end
        end
        end_time = now()
        
        BSON.bson(file, 
            Dict(
                :prob => :two_comp,
                :model => model, 
                :state => st,
                :saved_parameters => saved_parameters, 
                :LL => LL,
                :opt => opt, 
                :opt_state => opt_state, # To allow continuation of training.
                :duration => ckpt[:duration],
                :duration_second => end_time - start_time,
                :last_parameters => parameters # Check what parameters resulted in error
            )
        )

        if method == :FOCE
            [indv.eta .= 0 for indv in population_]
        end
    end
end

##### MSE:

method = "mse"
for outer_fold in 1:10
    cv = BSON.load("neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$i.bson")
    population_ = population[cv[:outer][outer_fold]] # Reshuffle the population so that data folds are re-aligned
    
    Threads.@threads for inner_fold in 1:10
        file = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$i/mse/outer_$(outer_fold)_inner_$(inner_fold).bson"
        if isfile(file) continue end
        println("Running model for outer_fold $(outer_fold) and inner fold $inner_fold")
        train_idxs = cv[:inner_train][inner_fold]
        ps, st = Lux.setup(Random.default_rng(), model)

        parameters = (weights = ps,)
        saved_parameters = Vector{typeof(parameters)}(undef, length(epochs_to_save))
        opt = Optimisers.ADAM(0.1f0)
        opt_state = Optimisers.setup(opt, parameters)
        losses = zeros(total_epochs)
        LL = zeros(Float32, length(epochs_to_save))
        start_time = now()

        for epoch in 1:total_epochs
            loss, back = Zygote.pullback(p -> fixed_objective(model, prob, population_[train_idxs], st, p), parameters)
            ∇ = first(back(1))
    
            losses[epoch] = loss
            if epoch == 1 || epoch % 250 == 0
                println("(Outer $(outer_fold), inner $(inner_fold)) Epoch $(epoch): loss = $(loss)")
            end

            opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
            if epoch in epochs_to_save
                LL[Integer(round(epoch / save_every)) + 1] = Float32(loss)
                saved_parameters[Integer(round(epoch / save_every)) + 1] = (weights = Lux.fmap(copy, parameters.weights), )
            end
        end
        end_time = now()
        
        BSON.bson(file, 
            Dict(
                :prob => :two_comp,
                :model => model, 
                :state => st,
                :saved_parameters => saved_parameters, 
                :LL => LL,
                :opt => opt, 
                :opt_state => opt_state, # To allow continuation of training.
                :duration => end_time - start_time,
                :last_parameters => parameters # Check what parameters resulted in error
            )
        )
    end
end

