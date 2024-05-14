import DifferentialEquations: ODEProblem, remake, solve, DiscreteCallback, Tsit5
import Zygote.ChainRules: @ignore_derivatives
import Zygote.ChainRulesCore
import ForwardDiff
import FiniteDiff
import Zygote
import Random
import Optim
import Lux

include("/home/alexanderjanssen/PhD/Models/DeepCompartmentModels.jl/src/lib/compartment_models.jl");
try
    include("/home/alexanderjanssen/PhD/Models/DeepCompartmentModels.jl/src/lib/population-mvp.jl");
catch e
end
include("/home/alexanderjanssen/PhD/Models/DeepCompartmentModels.jl/src/lib/dataset.jl");

using LinearAlgebra
using Distributions
using SciMLSensitivity

softplus(x::Real) = log(exp(x) + one(x))
invsoftplus(x::Real) = log(exp(x) - one(x))

function indicator(n, a, T=Bool)
    Iₐ = zeros(T, n, length(a))
    Zygote.ignore() do
        for i in eachindex(a)
            Iₐ[a[i], i] = one(T)
        end
    end
    return Iₐ
end

# Get vector of lower triangular, non-diagonal elements in matrix
function vecl(A::AbstractMatrix) 
    d = size(A, 1)
    return A[LowerTriangular(ones(Bool, d, d)) .⊻ I(d)]
end

function vecl_to_correlation_matrix(ρ::AbstractVector)
    d = Int(0.5 * (1 + sqrt(1 + 8 * length(ρ))))
    indexes = vecl(reshape(1:(d*d), (d,d)))
    return Symmetric(reshape(indicator(d*d, indexes) * ρ, (d, d)) + I, :L)
end

# We get a positive definite matrix as follows: Symmetric(ω ⋅ C ⋅ ω'), we need to use Symmetric due to numerical inaccuracy.
covariance_matrix(ρ::AbstractVector, σ::AbstractVector) = Symmetric(σ .* vecl_to_correlation_matrix(ρ) .*  σ')


function predict(prob, individual, p; typical=false, interpolate=false, full=false, tmax=maximum(individual.ty), measurement_idx = 1)
    z = typical ? p : p .* exp.([1 0; 0 1; 0 0; 0 0] * individual.eta)
    prob_ = remake(prob, tspan = (prob.tspan[1], tmax), p = [z; zero(p[1])])
    saveat = interpolate ? empty(individual.ty) : individual.ty
    
    return solve(prob_, Tsit5(), save_idxs=full ? (1:length(prob.u0)) : measurement_idx, saveat=saveat, tstops=individual.callback.condition.times, callback=individual.callback)
end

"""Can infer return type."""
function predict_adjoint(prob, t, callback, ζ, η::AbstractVector{T}; measurement_idx = 1) where T<:Real
    # z = p .* exp.(T[1 0; 0 1; 0 0; 0 0] * η) # Broken because of mutation (udate to LinearAlgebra? Maybe we need to define a custom adjoint here?)
    z = ζ .* exp.([η; 0.f0; 0.f0])
    prob_ = remake(prob, tspan = (prob.tspan[1], maximum(t)), p = [z; zero(ζ[1])])
    # return solve(prob_, Tsit5(), force_dtmin=true, saveat=t, tstops=callback.condition.times, callback=callback, sensealg=ForwardDiffSensitivity(;convert_tspan=true))[1, :]
    return solve(prob_, Tsit5(), dtmin=1e-10, saveat=t, tstops=callback.condition.times, callback=callback, sensealg=ForwardDiffSensitivity(;convert_tspan=true))[measurement_idx, :]
end

function predict_adjoint(prob, individual::Individual, z::AbstractVector{T}; measurement_idx = 1) where T
    prob_ = remake(prob, tspan = (prob.tspan[1], maximum(individual.ty)), p = [z; zero(T)])
    return solve(
        prob_, Tsit5(), dtmin=1e-10, saveat=individual.ty, 
        tstops=individual.callback.condition.times, callback=individual.callback, 
        sensealg=ForwardDiffSensitivity(;convert_tspan=true)
    )[measurement_idx, :]
end


import ForwardDiff
ADjac(func, x) = Zygote.forwarddiff(z -> ForwardDiff.jacobian(func, z), x)

δyδη(prob, t, callback, p, η) = FiniteDiff.finite_difference_jacobian(eta -> predict_adjoint(prob, t, callback, p, eta), η)
# δyδη(prob, t, callback, p, η) = ADjac(eta -> predict_adjoint(prob, t, callback, p, eta), η)

function objective(prob, individual::Individual, ζᵢ::AbstractVector, p::NamedTuple)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    # Σ = Diagonal(one.(ŷ) .* softplus.(p.sigma2))
    Σ = diagm(one.(ŷ) .* softplus.(p.sigma2)) # For alt_2
    Ω = covariance_matrix(tanh.(p.gamma), softplus.(p.omega))
    # In "Derivation of various..." it is: (y - ŷ) + Gᵢ * η == y - (ŷ - Gᵢ * η) == y - (ŷ + Gᵢ * -η)
    residuals = individual.y - (ŷ + Gᵢ * -individual.eta) # This is according to "R-based implementation ..." and definition of Taylor series expansion (ie. f(a) + f'(a) (x - a), where I believe x = 0).
    # residuals = individual.y - ŷ # This is saved under alt_2, also appropriate for FO since η = 0s so Gᵢ * η = 0
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    # return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals # This is for FO
    return log(det(Cᵢ)) + residuals' * inv(Σ) * residuals + individual.eta' * inv(Ω) * individual.eta # This is alt_2 (which is not correct for FO), corresponding to Eq. 21, and seemingly more stable?
end

function FO_objective_prop(ann, prob, population, p, st)
    σ = softplus.(p.theta[1:2])
    ω = softplus.(p.theta[3:4])
    C = inverse(VecCorrBijector())(p.theta[5:end])
    Ω = Symmetric(ω .* C .* ω')

    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ))
    for i in eachindex(population)
        neg_2LL += FO_prop(prob, population[i], ζ[:, i], Ω, σ)
    end
    
    return neg_2LL
end

function FOCE2_objective_prop(ann, prob, population, p, st)
    σ = softplus.(p.theta[1:2])
    ω = softplus.(p.theta[3:4])
    C = inverse(VecCorrBijector())(p.theta[5:end])
    Ω = Symmetric(ω .* C .* ω')

    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ))
    for i in eachindex(population)
        neg_2LL += FOCE2_prop(prob, population[i], ζ[:, i], Ω, σ)
    end
    
    return neg_2LL
end

function FO(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, zero.(individual.eta))
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, zero.(individual.eta))
    Σ = Diagonal(fill(first(σ²), length(ŷ)))
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals
end

function FO_prop(prob, individual::Individual, ζᵢ, Ω, σ)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, zero.(individual.eta))
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, zero.(individual.eta))
    Σ = Diagonal((σ[1] .+ ŷ .* σ[2]).^2)
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals
end

function FOCE1(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = Diagonal(fill(σ², length(ŷ)))
    residuals = individual.y - (ŷ + Gᵢ * -individual.eta)
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals
end

function FOCE2(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = diagm(fill(σ², length(ŷ)))
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Σ) * residuals + individual.eta' * inv(Ω) * individual.eta
end

function FOCE2_prop(prob, individual::Individual, ζᵢ, Ω, σ)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = collect(Diagonal((σ[1] .+ ŷ .* σ[2]).^2))
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Σ) * residuals + individual.eta' * inv(Ω) * individual.eta
end

# Convert a vector to lower-triangular matrix. Taken from ParameterHandling.jl
function vec_to_tril(v::AbstractVector{T}) where {T}
    n_vec = length(v)
    n_tril = Int((sqrt(1 + 8 * n_vec) - 1) / 2) # Infer the size of the matrix from the vector
    L = zeros(T, n_tril, n_tril)
    L[tril!(trues(size(L)))] = v
    return L
end
# Taken from ParameterHandling.jl
function tril_to_vec(X::AbstractMatrix{T}) where {T}
    n, m = size(X)
    n == m || error("Matrix needs to be square")
    return X[tril!(trues(size(X)))]
end
# Taken from ParameterHandling.jl
function ChainRulesCore.rrule(::typeof(vec_to_tril), v::AbstractVector{T}) where {T}
    L = vec_to_tril(v)
    pullback_vec_to_tril(Δ) = ChainRulesCore.NoTangent(), tril_to_vec(ChainRulesCore.unthunk(Δ))
    return L, pullback_vec_to_tril
end


function objective(obj_fn, ann, prob, population, st, p)
    σ² = softplus(p.theta[1])^2
    ω = softplus.(p.theta[2:3])
    C = inverse(VecCorrBijector())(p.theta[4:end])
    Ω = Symmetric(ω .* C .* ω')

    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ²))
    for i in eachindex(population)
        neg_2LL += obj_fn(prob, population[i], ζ[:, i], Ω, σ²)
    end
    
    return neg_2LL
end

function objective_old(objective_fn, ann, prob, population::Population, st, p::NamedTuple)
    ρ = tanh.(p.gamma)
    ω = softplus.(p.omega)
    Ω = covariance_matrix(ρ, ω)
    σ² = softplus.(p.sigma2)
    
    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ²))
    for i in eachindex(population)
        neg_2LL += objective_fn(prob, population[i], ζ[:, i], Ω, σ²)
    end
    
    return neg_2LL
end

function objective_alt(obj_fn, ann, prob, population, st, p)
    σ² = softplus(p.theta[1])^2
    L = vec_to_tril(p.theta[2:end])
    Ω = L * L'

    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ²))
    for i in eachindex(population)
        neg_2LL += obj_fn(prob, population[i], ζ[:, i], Ω, σ²)
    end
    
    return neg_2LL
end


function EBE(prob, individual::Individual, ζᵢ, η, Ω_inv, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, η)
    residuals = individual.y - ŷ
    Σ = Diagonal(fill(first(σ²), length(ŷ)))
    
    return log(det(Σ)) + residuals' * inv(Σ) * residuals + η' * Ω_inv * η
end

function optimize_etas!(population::Population, prob, ann, st::NamedTuple, p::NamedTuple, alg=Optim.NelderMead())
    σ² = softplus(p.theta[1])^2
    ω = softplus.(p.theta[2:3])
    C = inverse(VecCorrBijector())(p.theta[4:end])
    invΩ = inv(Symmetric(ω .* C .* ω'))
    # ρ = tanh.(p.gamma)
    # ω = softplus.(p.omega)
    # Ω_inv = inv(covariance_matrix(ρ, ω))
    # σ² = softplus.(p.sigma2)
    
    ζ, _ = Lux.apply(ann, population.x, p.weights, st)

    for i in eachindex(population)
        # opt = Optim.optimize((eta) -> EBE(prob, population[i], ζ[:, i], eta, invΩ, σ²), zeros(eltype(ζ), 2), alg)
        opt = Optim.optimize(
            eta -> EBE(prob, population[i], ζ[:, i], eta, invΩ, σ²), 
            Float32[-3, -3],
            Float32[3, 3],
            zeros(eltype(ζ), 2), 
            Optim.Fminbox(Optim.BFGS()),
            Optim.Options(outer_iterations = 3, allow_f_increases = true, x_abstol = 1e-3)
        )
        population[i].eta .= Float32.(opt.minimizer)
    end
    
    nothing
end

function optimize_etas_alt!(population::Population, prob, ann, st::NamedTuple, p::NamedTuple, alg=Optim.NelderMead())
    σ² = softplus(p.theta[1])^2
    L = vec_to_tril(p.theta[2:end])
    invΩ = inv(L * L')
    
    ζ, _ = Lux.apply(ann, population.x, p.weights, st)

    for i in eachindex(population)
        opt = Optim.optimize((eta) -> EBE(prob, population[i], ζ[:, i], eta, invΩ, σ²), zeros(eltype(ζ), 2), alg)
        population[i].eta .= opt.minimizer
    end
    
    nothing
end



# function objective(ann, prob, population::Population, st::NamedTuple, p::NamedTuple)
#     ζ, _ = Lux.apply(ann, get_x(population), p.weights, st) # Lux
#     # ζ = ann(p.weights)(get_x(population)) # Flux
#     neg_2LL = zero(eltype(p.sigma2))
#     for i in eachindex(population)
#         neg_2LL += objective(prob, population[i], ζ[:, i], p)
#         # if neg_2LL == Inf return end
#     end
#     return neg_2LL
# end

function fixed_objective(ann, prob, population::Population, st::NamedTuple, p::NamedTuple)
    ζ, _ = Lux.apply(ann, population.x, p.weights, st)
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

# exclude_fn(v) = isempty(v) || Lux.Functors.isleaf(v)

# function get_l2_norm(ann, p)
#     name = filter(key -> typeof(ann.layers[key]) <: Lux.BranchLayer, keys(ann.layers))
#     parameters = first(p[name])
#     squared_cumsum = 0.f0
#     for layer in keys(parameters)
#         for layer_ in keys(parameters[layer])
#             if !isempty(parameters[layer][layer_])
#                 if keys(parameters[layer][layer_])[1] == :layer_1
#                     for layer__ in keys(parameters[layer][layer_])
#                         squared_cumsum += sum(abs2, parameters[layer][layer_][layer__].weight)
#                     end
#                 else
#                     squared_cumsum += sum(abs2, parameters[layer][layer_].weight)
#                 end
#             end
#         end
#     end
#     return squared_cumsum
# end

# function fixed_objective_l2(ann, prob, population::Population, st::NamedTuple, p::NamedTuple)
#     # We want to shrink the weights of the branches, not the biases or the weights of the later layers (i.e. the typical PK parameters)
#     ζ, _ = Lux.apply(ann, get_x(population), p, st)
#     SSE = zero(Float32)
#     ℓ² = get_l2_norm(inn, p)
#     for i in eachindex(population)
#         individual = population[i]
#         ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζ[:, i], individual.eta)
#         SSE += sum(abs2, individual.y - ŷ)
#     end
#     return SSE + ℓ² # is the same as N(0, 1) prior on the weights
# end

################################################################################
##########                                                            ##########
##########                       Normalize layer                      ##########
##########                                                            ##########
################################################################################

struct Normalize{T} <: Lux.AbstractExplicitLayer
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

Normalize(lb::Real, ub::Real) = Normalize([lb], [ub])
Normalize(ub::Real) = Normalize([ub])
Normalize(lb::AbstractVector, ub::AbstractVector) = Normalize{eltype(lb)}(lb, ub)
Normalize(ub::AbstractVector) = Normalize{eltype(ub)}(zero.(ub), ub)

Lux.initialparameters(rng::Random.AbstractRNG, ::Normalize) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Normalize) = (lb=l.lb, ub=l.ub)

Lux.parameterlength(::Normalize) = 0
Lux.statelength(::Normalize) = 2 # is this correct?

function (l::Normalize)(x::AbstractArray, ps, st::NamedTuple)
    # y = (((x .- st.lb) ./ (st.ub - st.lb)) .- 0.5f0) .* 2.f0 # Normalizes between -1 and 1
    y = (x .- st.lb) ./ (st.ub - st.lb) # Normalizes between 0 and 1, seems to work better against overfitting? Maybe because the bias is initialized further away.
    return y, st
end

################################################################################
##########                                                            ##########
##########                 Add global parameter layer                 ##########
##########                                                            ##########
################################################################################


struct AddGlobalParameters{T, F1, F2} <: Lux.AbstractExplicitLayer
    theta_dim::Int
    out_dim::Int
    locations::AbstractVector{Int}
    init_theta::F1
    activation::F2
end

AddGlobalParameters(out_dim, loc, T=Float32; init_theta=Lux.glorot_uniform, activation=softplus) = AddGlobalParameters{T, typeof(init_theta), typeof(activation)}(length(loc), out_dim, loc, init_theta, activation)

Lux.initialparameters(rng::Random.AbstractRNG, l::AddGlobalParameters) = (theta = l.init_theta(rng, l.theta_dim, 1),)
Lux.initialstates(rng::Random.AbstractRNG, l::AddGlobalParameters{T,F1,F2}) where {T,F1,F2} = (indicator_theta = indicator(l.out_dim, l.locations, T), indicator_x = indicator(l.out_dim, (1:l.out_dim)[Not(l.locations)], T))
Lux.parameterlength(l::AddGlobalParameters) = l.theta_dim
Lux.statelength(::AddGlobalParameters) = 2

# the indicators should be in the state!
function (l::AddGlobalParameters)(x::AbstractMatrix, ps, st::NamedTuple)
    if size(st.indicator_x, 2) !== size(x, 1)
        indicator_x = st.indicator_x * st.indicator_x' # Or we simply do not do this, the one might already be in the correct place following the combine function.
    else
        indicator_x = st.indicator_x
    end
    y = indicator_x * x + st.indicator_theta * repeat(l.activation.(ps.theta), 1, size(x, 2))
    return y, st
end


################################################################################
##########                                                            ##########
##########                  Combine parameters layer                  ##########
##########                                                            ##########
################################################################################

struct Combine{T1, T2} <: Lux.AbstractExplicitLayer
    out_dim::Int
    pairs::T2
end

function Combine(pairs::Vararg{Pair}; T=Float32)
    out_dim = maximum([maximum(pairs[i].second) for i in eachindex(pairs)])
    return Combine{T, typeof(pairs)}(out_dim, pairs)
end

function get_state(l::Combine{T1, T2}) where {T1, T2}
    indicators = Vector{Matrix{T1}}(undef, length(l.pairs))
    negatives = Vector{Vector{T1}}(undef, length(l.pairs))
    for pair in l.pairs
        Iₛ = indicator(l.out_dim, pair.second, T1)
        indicators[pair.first] = Iₛ
        negatives[pair.first] = abs.(vec(sum(Iₛ, dims=2)) .- one(T1))
    end
    return (indicators = indicators, negatives = negatives)
end

Lux.initialparameters(rng::Random.AbstractRNG, ::Combine) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Combine) = get_state(l)
Lux.parameterlength(::Combine) = 0
Lux.statelength(::Combine) = 2

function (l::Combine)(x::Tuple, ps, st::NamedTuple) 
    indicators = @ignore_derivatives st.indicators
    negatives = @ignore_derivatives st.negatives
    y = reduce(.*, _combine.(x, indicators, negatives))
    return y, st
end

_combine(x::AbstractMatrix, indicator::AbstractMatrix, negative::AbstractVector) = indicator * x .+ negative