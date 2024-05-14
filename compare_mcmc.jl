import Optimisers
import Random
import Plots
import BSON
import Flux
import CSV
import Lux

include("neural-mixed-effects-paper/new_approach/model.jl");
include("neural-mixed-effects-paper/new_approach/variational_2.jl");
include("src/lib/constraints.jl");

using Turing
using StatsPlots
using AdvancedVI 
using DataFrames
using KernelDensity

softplus(x::T) where T<:Real = log(exp(x) + one(T))
softplus_inv(x::T) where T<:Real = log(exp(x) - one(T))

Ω_true = [0.037 0.45 * sqrt(0.037) * sqrt(0.017) ; 0.45 * sqrt(0.037) * sqrt(0.017) 0.017]
σ_true = 0.03

df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/simulation.csv"))
df_group = groupby(df, :id)

indvs = Vector{Individual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    x = Vector{Float64}(group[1, [:wt, :vwf]])
    y = Vector{Float64}(group.dv[2:end])
    t = Vector{Float64}(group.time[2:end])
    𝐈 = Matrix{Float64}(group[1:1, [:time, :amt, :rate, :duration]])
    callback = generate_dosing_callback(𝐈; S1 = 1/1000)
    indvs[i] = Individual(x, y, t, callback; id = "$(group[1,:id])")
end

population = Population(indvs)
adapt!(population, 2)

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
non_zero_relu(x::T) where {T<:Real} = Lux.relu(x) + T(1e-3)
leaky_softplus(x::T; α=T(0.05), β=T(0.1)) where {T<:Real} = α * x + (1 - α) * smooth_relu(x; β)

prob = ODEProblem(two_comp!, zeros(2), (-0.01, 72.))

# True PK parameters:
cl = 0.1 .* (population.x[1, :] ./ 70).^0.75 .* (1 / 55 .* leaky_softplus.(-population.x[2, :] .+ 100) .+ 0.9)
v1 = 2. .* (population.x[1, :] ./ 70)
q = 0.15
v2 = 0.75

true_pk = vcat(cl', v1', fill(q, 1, length(cl)), fill(v2, 1, length(cl)))

fold_idx = 1
idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(fold_idx).csv")).idxs


################################################################################
##########                                                            ##########
##########               True PK and pop. params known                ##########
##########                                                            ##########
################################################################################
# This collects data on the accuracy of η posteriors in the best case scenario.

@model function mcmc(prob, individual, ζ, y; Ω=Ω_true, σ=σ_true)
    η ~ MultivariateNormal(zeros(2), Ω)
    
    if any(abs.(η) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

    z = ζ .* exp.([1 0; 0 1; 0 0; 0 0] * η)
    ŷ = predict_adjoint(prob, individual, z)
    y ~ MultivariateNormal(ŷ, σ)
end

Threads.@threads for i in idxs
    m = mcmc(prob, population[i], true_pk[:, i], population[i].y);
    chain = sample(m, NUTS(), 10_000)
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/mcmc/idx_$i.bson", Dict(:chain => chain))
end

# VI approach 1: full-rank using entropy based objective:
function advi_eta_entropy(prob, population, ζ, parameters::NamedTuple; Ω=Ω_true, σ=σ_true, num_samples = 3)
    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))

    for i in eachindex(population)
        μ = 𝜙[1:m, i]
        ω = softplus.(𝜙[m+1:2m, i])
        Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
        Lᵢ = LowerTriangular(Lc .* ω)
        for _ in 1:num_samples
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += (logπ / num_samples)
        end
        ELBO += entropy(MultivariateNormal(μ, Lᵢ * Lᵢ'))
    end
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    parameters = (phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))), )
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -advi_eta_entropy(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-entropy/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end

# VI approach 2: full-rank using p(x, z) - q(z) based objective
function advi_eta_var_post(prob, population, ζ, parameters::NamedTuple; Ω=Ω_true, σ=σ_true, num_samples = 3)
    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))

    for i in eachindex(population)
        μ = 𝜙[1:m, i]
        ω = softplus.(𝜙[m+1:2m, i])
        Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
        Lᵢ = LowerTriangular(Lc .* ω)
        q = MultivariateNormal(μ, Lᵢ * Lᵢ')
        for _ in 1:num_samples
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += ((logπ - logpdf(q, ηᵢ)) / num_samples)
        end
    end
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    parameters = (phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))), )
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -advi_eta_var_post(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-var-post/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end

# VI approach 3: full-rank using p(x, z) - q(z) based objective with path-deriv. gradient estimator
function advi_eta_path_deriv(prob, population, ζ, parameters::NamedTuple; Ω=Ω_true, σ=σ_true, num_samples = 3)
    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))

    for i in eachindex(population)
        μ = 𝜙[1:m, i]
        ω = softplus.(𝜙[m+1:2m, i])
        Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
        Lᵢ = LowerTriangular(Lc .* ω)
        q = Zygote.ChainRules.@ignore_derivatives MultivariateNormal(μ, Lᵢ * Lᵢ')
        for _ in 1:num_samples
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += ((logπ - logpdf(q, ηᵢ)) / num_samples)
        end
    end
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    parameters = (phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))), )
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -advi_eta_path_deriv(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-path-deriv/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end


################################################################################
##########                                                            ##########
##########                   True PK params. known                    ##########
##########                                                            ##########
################################################################################

@model function full_mcmc(prob, population, ζ, y)
    σ ~ LogNormal(-3., 1.)
    ω ~ filldist(LogNormal(-1.5, 1.), 2)
    ρ ~ Bijectors.transformed(Beta(2., 2.), Bijectors.Scale(2.)) # Essentially a LKJ(2, 1)
    Ω = Symmetric(ω .* [1. ρ - 1.; ρ - 1. 1.] .* ω')

    η ~ filldist(MultivariateNormal(zeros(2), Ω), length(population))
    
    if any(abs.(η) .> 4)
        Turing.@addlogprob! -Inf
        return
    end
    
    Threads.@threads for i in eachindex(population)
        individual = population[i]
        z = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * η[:, i])
        ŷ = predict_adjoint(prob, individual, z)
        y[i] ~ MultivariateNormal(ŷ, σ)
    end
end

m = full_mcmc(prob, population[idxs], true_pk[:, idxs], population[idxs].y);
chain = sample(m, NUTS(), 5_000)
BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/mcmc/fold_$(fold_idx)_5000_samples.bson", Dict(:chain => chain))


# VI approach 1: full-rank using entropy based objective:
function full_advi_eta_entropy(prob, population, ζ, parameters::NamedTuple; num_samples = 3)
    b1 = inverse(bijector(LogNormal()))
    b2 = inverse(VecCorrBijector())
    d = 4
    theta_mu = parameters.theta[1:d]
    theta_sigma = softplus.(parameters.theta[d+1:2d])
    theta_Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(parameters.theta[2d+1:end]))
    theta_L = LowerTriangular(theta_Lc .* theta_sigma)
    
    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))
    for j in 1:num_samples
        θ = theta_mu + theta_L * randn(eltype(ζ), 4)
        σ, logabsdet_sigma = with_logabsdet_jacobian(b1, θ[1])
        omega, logabsdet_omega = with_logabsdet_jacobian(b1, θ[2:3])
        C, logabsdet_C = with_logabsdet_jacobian(b2, θ[4:4])
        Ω = Symmetric(omega .* C .* omega')
        
        for i in eachindex(population)
            μ = 𝜙[1:m, i]
            ω = softplus.(𝜙[m+1:2m, i])
            Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
            Lᵢ = LowerTriangular(Lc .* ω)
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += (logπ / num_samples)
            if j == 1 # the entropy should be added only once per individual
                ELBO += entropy(MultivariateNormal(μ, Lᵢ * Lᵢ'))
            end
        end

        ELBO += (logpdf(LogNormal(-3., 1.), σ) + logabsdet_sigma) / num_samples
        ELBO += (logpdf(filldist(LogNormal(-1.5, 1.), 2), omega) + logabsdet_omega) / num_samples
        ELBO += (logpdf(LKJ(2, 1.), C) + logabsdet_C) / num_samples
    end
    
    ELBO += entropy(MultivariateNormal(theta_mu, theta_L * theta_L'))
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    theta_init = [-3; -1.5; -1.5; 0.3; softplus_inv.([0.5, 0.5, 0.5, 0.5]); VecCorrBijector()(collect(I(4)))]
    parameters = (theta = theta_init, phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))))
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -full_advi_eta_entropy(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/vi-entropy/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end

# VI approach 2: full-rank using p(x, z) - q(z) based objective
function full_advi_eta_var_post(prob, population, ζ, parameters::NamedTuple; num_samples = 3)
    b1 = inverse(bijector(LogNormal()))
    b2 = inverse(VecCorrBijector())
    d = 4
    theta_mu = parameters.theta[1:d]
    theta_sigma = softplus.(parameters.theta[d+1:2d])
    theta_Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(parameters.theta[2d+1:end]))
    theta_L = LowerTriangular(theta_Lc .* theta_sigma)
    q_theta = MultivariateNormal(theta_mu, theta_L * theta_L')

    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))

    for _ in 1:num_samples
        θ = theta_mu + theta_L * randn(eltype(ζ), 4)
        σ, logabsdet_sigma = with_logabsdet_jacobian(b1, θ[1])
        omega, logabsdet_omega = with_logabsdet_jacobian(b1, θ[2:3])
        C, logabsdet_C = with_logabsdet_jacobian(b2, θ[4:4])
        Ω = Symmetric(omega .* C .* omega')

        for i in eachindex(population)
            μ = 𝜙[1:m, i]
            ω = softplus.(𝜙[m+1:2m, i])
            Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
            Lᵢ = LowerTriangular(Lc .* ω)
            q = MultivariateNormal(μ, Lᵢ * Lᵢ')
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += ((logπ - logpdf(q, ηᵢ)) / num_samples)
        end

        ELBO += (logpdf(LogNormal(-3., 1.), σ) + logabsdet_sigma) / num_samples
        ELBO += (logpdf(filldist(LogNormal(-1.5, 1.), 2), omega) + logabsdet_omega) / num_samples
        ELBO += (logpdf(LKJ(2, 1.), C) + logabsdet_C) / num_samples

        ELBO -= logpdf(q_theta, θ) / num_samples
    end
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    theta_init = [-3; -1.5; -1.5; 0.3; softplus_inv.([0.5, 0.5, 0.5, 0.5]); VecCorrBijector()(collect(I(4)))]
    parameters = (theta = theta_init, phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))))
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -full_advi_eta_var_post(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/vi-var-post/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end

# VI approach 3: full-rank using p(x, z) - q(z) based objective with path-deriv. gradient estimator
function full_advi_eta_path_deriv(prob, population, ζ, parameters::NamedTuple; num_samples = 3)
    b1 = inverse(bijector(LogNormal()))
    b2 = inverse(VecCorrBijector())
    d = 4
    theta_mu = parameters.theta[1:d]
    theta_sigma = softplus.(parameters.theta[d+1:2d])
    theta_Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(parameters.theta[2d+1:end]))
    theta_L = LowerTriangular(theta_Lc .* theta_sigma)
    q_theta = Zygote.ChainRules.@ignore_derivatives MultivariateNormal(theta_mu, theta_L * theta_L')

    𝜙 = parameters.phi
    m = 2
    ELBO = zero(eltype(ζ))

    for _ in 1:num_samples
        θ = theta_mu + theta_L * randn(eltype(ζ), 4)
        σ, logabsdet_sigma = with_logabsdet_jacobian(b1, θ[1])
        omega, logabsdet_omega = with_logabsdet_jacobian(b1, θ[2:3])
        C, logabsdet_C = with_logabsdet_jacobian(b2, θ[4:4])
        Ω = Symmetric(omega .* C .* omega')

        for i in eachindex(population)
            μ = 𝜙[1:m, i]
            ω = softplus.(𝜙[m+1:2m, i])
            Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(𝜙[2m+1:end, i]))
            Lᵢ = LowerTriangular(Lc .* ω)
            q = Zygote.ChainRules.@ignore_derivatives MultivariateNormal(μ, Lᵢ * Lᵢ')
            ηᵢ = μ + Lᵢ * randn(eltype(ζ), 2)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            ELBO += ((logπ - logpdf(q, ηᵢ)) / num_samples)
        end

        ELBO += (logpdf(LogNormal(-3., 1.), σ) + logabsdet_sigma) / num_samples
        ELBO += (logpdf(filldist(LogNormal(-1.5, 1.), 2), omega) + logabsdet_omega) / num_samples
        ELBO += (logpdf(LKJ(2, 1.), C) + logabsdet_C) / num_samples

        ELBO -= logpdf(q_theta, θ) / num_samples
    end
    
    return ELBO
end

Threads.@threads for replicate in 1:20
    theta_init = [-3; -1.5; -1.5; 0.3; softplus_inv.([0.5, 0.5, 0.5, 0.5]); VecCorrBijector()(collect(I(4)))]
    parameters = (theta = theta_init, phi = (vcat(zeros(2, length(idxs)), ones(2, length(idxs)) .* -1, rand(Normal(0, 0.3), 1, length(idxs)))))
    opt = Optimisers.ADAM(0.1)
    opt_state = Optimisers.setup(opt, parameters)
    for epoch in 1:1_000
        loss, back = Zygote.pullback(p -> -full_advi_eta_path_deriv(prob, population[idxs], true_pk[:, idxs], p), parameters)
        if epoch == 1 || epoch % 100 == 0
            println("(Replicate $replicate) Epoch $epoch:, ELBO = $(-loss)")
        end
        ∇ = first(back(1))
        opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
    end
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/vi-path-deriv/fold_$(fold_idx)_replicate_$replicate.bson", Dict(:parameters => parameters, :idxs => idxs))
end


################################################################################


# Look at results
i = 10
chain = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/mcmc/idx_$(idxs[i]).bson")[:chain];

# Entropy based version:
plt1 = Plots.plot()
for replicate in 1:20
    ckpt_entropy = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-entropy/fold_1_replicate_$replicate.bson")
    p = ckpt_entropy[:parameters]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]))
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=0.5, label=nothing)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=1, label=nothing)
    covellipse!(plt1, q.μ, q.Σ, color=:purple, fillalpha=0.05, n_std=1.96, label=nothing)
end
plt1

Plots.plot!(plt1, kde(chain.value.data[:, 1:2, 1], bandwidth=(0.02, 0.02)), linewidth=2, levels=7)

# version with variational posterior likelihood
plt2 = Plots.plot()
for replicate in 1:20
    ckpt_var_post = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-var-post/fold_1_replicate_$replicate.bson")
    p = ckpt_var_post[:parameters]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]))
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=0.5, label=nothing)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=1, label=nothing)
    covellipse!(plt2, q.μ, q.Σ, color=:orange, fillalpha=0.05, n_std=1.96, label=nothing)
end
plt2

Plots.plot!(plt2, kde(chain.value.data[:, 1:2, 1], bandwidth=(0.02, 0.02)), linewidth=2, levels=7)

# Path. deriv:
plt3 = Plots.plot()
for replicate in 1:20
    ckpt_path_deriv = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-path-deriv/fold_1_replicate_$replicate.bson")
    p = ckpt_path_deriv[:parameters]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]))
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=0.5, label=nothing)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=1, label=nothing)
    covellipse!(plt3, q.μ, q.Σ, color=:green, fillalpha=0.05, n_std=1.96, label=nothing)
end
plt3

Plots.plot!(plt3, kde(chain.value.data[:, 1:2, 1], bandwidth=(0.02, 0.02)), linewidth=2, levels=7)

# Makes it clear that we should use path. deriv estimator:
Plots.plot(plt1, plt2, plt3, layout=(1, 3), size=(800, 200), xlim=0.85 .* Plots.xlims(plt1), ylim=0.65 .* Plots.ylims(plt1), xticks = false, yticks = false, colorbar = false) 



###### We should also look at the posteriors for the population parameters:

#--

################################################################################
#########                                                              #########
#########    Compare posteriors at the end of optimization to MCMC     #########
#########                                                              #########
################################################################################

folder = "neural-mixed-effects-paper/new_approach/checkpoints/VI-eta/inn"
files = readdir(folder)
filter!(contains("fold_$(fold_idx)_"), files)
theta = zeros(4, length(files))
zeta = zeros(4, length(files), length(idxs))
for (i, file) in enumerate(files)
    ckpt = BSON.load(joinpath(folder, file))
    theta[:, i] = mean(hcat(map(x -> x.theta, ckpt[:saved_parameters][40:end])...), dims=2)
    zeta[:, i, :], _ = Lux.apply(ckpt[:model], population[idxs].x, ckpt[:saved_parameters][end].weights, ckpt[:state])
end
theta[1, :] = softplus.(theta[1, :])
theta[2:3, :] = softplus.(theta[2:3, :])
ω = mean(theta[2:3, :], dims=2)[:, 1]
C = inverse(VecCorrBijector())([mean(theta[4:end, :])])
Ω_pred = ω .* C .* ω'
σ_pred = mean(theta[1, :])

pred_zeta = reshape(mean(zeta, dims = 2), (4, length(idxs)))

Threads.@threads for i in eachindex(idxs)
    m = mcmc(prob, population[idxs[i]], pred_zeta[:, i], population[idxs[i]].y; Ω=Ω_pred, σ=σ_pred);
    chain = sample(m, NUTS(), 10_000)
    BSON.bson("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/after_vi_eta/mcmc/idx_$(idxs[i]).bson", Dict(:chain => chain))
end


# Probably we need to look at the actual posteriors (i.e. in PK param level, 
# since the neural networks vary in their predictions, so so will the posteriors):
# Make plot:
i = 1
chain = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/after_vi_eta/mcmc/idx_$(idxs[i]).bson")[:chain]
folder = "neural-mixed-effects-paper/new_approach/checkpoints/VI-eta/inn"
files = readdir(folder)
filter!(contains("fold_$(fold_idx)_"), files)

plt = Plots.plot()
for file in files
    ckpt = BSON.load(joinpath(folder, file))
    p = ckpt[:saved_parameters][end]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]))
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=0.5, label=nothing)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=1, label=nothing)
    covellipse!(plt, q.μ, q.Σ, color=:green, fillalpha=0.1, n_std=1.96, label=nothing)
end
plt

Plots.plot!(plt, kde(chain.value.data[:, 1:2, 1], bandwidth=(0.02, 0.02)), linewidth=2, color=:black, linestyle=:dash, levels=7)


# true post where we transform into z space
true_post = log.(pred_zeta[1:2, i] .* exp.(chain.value.data[:, 1:2, 1])')

plt = Plots.plot()
for file in files
    ckpt = BSON.load(joinpath(folder, file))
    p = ckpt[:saved_parameters][end]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = _chol_lower(inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]))
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    pred, _ = Lux.apply(ckpt[:model], population[idxs].x, ckpt[:saved_parameters][end].weights, ckpt[:state])
    real_q = fit(MultivariateNormal, Float64.(log.(pred[1:2, i].* exp.(rand(q, 100_000)))))
    # Plots.scatter!(plt, result[1, :], result[2, :], alpha=0.1, label=nothing, color=:green)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=0.5, label=nothing)
    # covellipse!(plt, q.μ, q.Σ, color=:purple, fillalpha=0, n_std=1, label=nothing)
    covellipse!(plt, real_q.μ, real_q.Σ, color=:green, fillalpha=0.1, n_std=1.96, label=nothing)
end
plt

Plots.plot!(plt, kde(true_post[1:2, :]', bandwidth=(0.02, 0.02)), linewidth=2, color=:black, linestyle=:dash, levels=7)