import Bijectors: VecCorrBijector, VecCholeskyBijector, inverse, with_logabsdet_jacobian, bijector
import Turing.DistributionsAD: TuringDenseMvNormal, TuringDiagMvNormal
import Zygote.ChainRules: @ignore_derivatives
import LinearAlgebra: Diagonal
import ForwardDiff

using Distributions

"""
The ELBO we use is:

    ℒ(𝜙) = 𝔼[log p(y | X(T⁻¹(η))) + log p(η)] - 𝔼[q_𝜙(η | y)]
         = 𝔼[log p(y | X(T⁻¹(η))) + log p(η)] + ℍ[q_𝜙] 

This gives an approximation of the posterior from the MCMC procedure. It is 
however possible that the way we code the MCMC model is incorrect (i.e. maybe 
y ~ MultivariateNormal is not correct?).
"""

# We use:
# * MultivariateNormal for p(y | z; σ²), 
# * MultivariateNormal for p(η; Ω), 
# * Full-rank approximation for q(η; 𝜙)

variance(ŷ, σ::T) where T<:Real = Diagonal(fill(σ^2, length(ŷ))) # Additive error (or Proportional)
variance(ŷ, σ::AbstractVector{T}) where T<:Real = Diagonal((σ[1] .+ (ŷ .* σ[2])).^2) # Combined error

function loglikelihood(prob, individual, z, σ)
    ŷ = predict_adjoint(prob, individual, z)
    Σ = variance(ŷ, σ)
    return logpdf(MultivariateNormal(ŷ, Σ), individual.y)
end

logprior(η::AbstractVector, Ω::AbstractMatrix) = logpdf(MultivariateNormal(zero.(η), Ω), η)

logjoint(prob, individual, z, η, Ω, σ) = loglikelihood(prob, individual, z, σ) + logprior(η, Ω)

# More expensive on a population basis (probably due to inversion of Ω)
# function emp_bayes_est!(population, prob, model, p, st)
#     σ = softplus(p.theta[1])
#     ω = softplus.(p.theta[2:3])
#     C = inverse(VecCorrBijector())(p.theta[4:end])
#     Ω = Symmetric(ω .* C .* ω')
#     ζ, _ = Lux.apply(model, population.x, p.weights, st)

#     for i in eachindex(population)
#         opt = Optim.optimize(eta -> -2 * logjoint(prob, population[i], ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * eta), eta, Ω, σ), zero.(population[i].eta))
#         population[i].eta .= Float32.(opt.minimizer)
#     end

#     nothing
# end

# logabsdetjac_z(z::AbstractVector{T}) where {T<:Real} = log(abs(det(Diagonal(-one(T) ./ z))))

function variational_posterior(getq, 𝜙ᵢ::AbstractVector{T}, η; path_deriv_est = true) where {T<:ForwardDiff.Dual}
    q = path_deriv_est ? getq(map(𝜙 -> 𝜙.value, 𝜙ᵢ)) : getq(𝜙ᵢ) # @ignore_derivatives does not work for ForwardDiff.
    return variational_posterior(q, η)
end

function variational_posterior(getq, 𝜙ᵢ, η; path_deriv_est = true) # Evaluates q_𝜙(z)
    q = !path_deriv_est ? getq(𝜙ᵢ) : @ignore_derivatives getq(𝜙ᵢ) # path derivative gradient estimator from "Sticking the landing ... "
    # The path derivative estimator is a bad choice when the variational posterior does not match the true posterior.
    return variational_posterior(q, η)
end

variational_posterior(q::Distribution, η) = logpdf(q, η) # OLD: + logabsdetjac_z(z)

function elbo(prob, individual::Individual, ζᵢ, getq::Function, 𝜙ᵢ, Ω, σ; with_entropy)
    ηᵢ = rand(getq(𝜙ᵢ))
    zᵢ = ζᵢ .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
    logπ = logjoint(prob, individual, zᵢ, ηᵢ, Ω, σ)
    return with_entropy ? logπ + entropy(getq(𝜙ᵢ)) : logπ - variational_posterior(getq, 𝜙ᵢ, ηᵢ)
end

function mean_field_approx(𝜙) # 𝜙 = [μ, σ*]
    d = Integer(length(𝜙) / 2)
    μ = 𝜙[1:d]
    σ = softplus.(𝜙[d+1:d*2])
    return MultivariateNormal(μ, σ)
end

function full_rank(𝜙; d=2) # 𝜙 = [μ, ω*, ρ*], d = num_rand_effects
    μ = 𝜙[1:d]
    σ = softplus.(𝜙[d+1:2d])
    ρ = tanh.(𝜙[2d+1:end])
    Σ = covariance_matrix(ρ, σ)
    return MultivariateNormal(μ, Σ)
end

J_sigmoid(x::T) where {T<:Real} = sigmoid(x) * (one(T) - sigmoid(x))

function getθ(θ_; d) # θ_ = {ω, ρ, σ}
    ω = exp.(θ_[1:d])
    ρ_ = θ_[d+1:d+sum(1:d-1)]
    ρ = sigmoid.(1 .+ 1 .- exp.(ρ_)) # Makes the distribution left skewed between 0 and 1
    σ = exp.(θ_[d+sum(1:d-1)+1:end])
    logabsdet = log(abs(det(Diagonal(ω)))) + log(abs(det(Diagonal(J_sigmoid.(exp.(ρ)) .* exp.(ρ))))) + log(abs(det(Diagonal(σ))))
    return ω, ρ, length(σ) == 1 ? σ[1] : σ, logabsdet
end

"""Calculates the posterior over η"""
function partial_advi(ann, prob, population, 𝜙::NamedTuple, st; num_samples = 1, with_entropy = false)
    m = 2 # num_rand_effects
    ζ, _ = Lux.apply(ann, population.x, 𝜙.weights, st) # Lux
    σ_ω = softplus.(𝜙.theta[1:3])
    σ = σ_ω[1]
    ω = σ_ω[2:3]
    C = inverse(VecCorrBijector())(𝜙.theta[4:end])
    Ω = Symmetric(ω .* C .* ω')

    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        for i in eachindex(population)
            # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
            Lᵢ = LowerTriangular(_chol_lower(inverse(VecCholeskyBijector(:L))(𝜙.phi[2m+1:end, i])) .* softplus.(𝜙.phi[m+1:2m, i]))
            ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:2, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
    end
    
    return ELBO
end

function partial_advi_prop(ann, prob, population, 𝜙::NamedTuple, st; num_samples = 1, with_entropy = false)
    m = 2 # num_rand_effects
    ζ, _ = Lux.apply(ann, Zygote.ignore_derivatives(population.x), 𝜙.weights, st) # Lux
    σ_ω = softplus.(𝜙.theta[1:4])
    σ = σ_ω[1:2]
    ω = σ_ω[3:4]
    C = inverse(VecCorrBijector())(𝜙.theta[5:end])
    Ω = Symmetric(ω .* C .* ω')

    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        for i in eachindex(population)
            # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
            Lᵢ = LowerTriangular(_chol_lower(inverse(VecCholeskyBijector(:L))(𝜙.phi[2m+1:end, i])) .* softplus.(𝜙.phi[m+1:2m, i]))
            ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:2, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
    end
    
    return ELBO
end

"""This formulation uses cholesky decompositions instead of ω and C"""
function partial_advi_alt(ann, prob, population, 𝜙::NamedTuple, st; num_samples = 3)
    m = 2 # num_rand_effects
    ζ, _ = Lux.apply(ann, population.x, 𝜙.weights, st) # Lux
    σ = softplus.(𝜙.theta[1])
    L = vec_to_tril(𝜙.theta[2:end])
    Ω = L * L'

    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        for i in eachindex(population)
            # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
            Lᵢ = vec_to_tril(𝜙.phi[m+1:end, i])
            ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:2, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
    end
    
    return ELBO
    
    # m = 2 # num_rand_effects
    # ζ, _ = Lux.apply(ann, population.x, 𝜙.weights, st) # Lux
    # σ = softplus(𝜙.theta[1])
    # L = LowerTriangular(vec_to_tril(𝜙.theta[2:end]))
    # Ω = L * L'

    # ELBO = eltype(ζ)
    # for _ in 1:num_samples
    #     for i in eachindex(population)
    #         # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
    #         Lᵢ = LowerTriangular(vec_to_tril(𝜙.phi[m+1:end, i]))
    #         ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
    #         zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
    #         logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
    #         qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:m, i], Lᵢ * Lᵢ')
    #         ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
    #     end
    # end
    
    # return ELBO
end

# Lc ~ LKJCholesky() ← use Bijector and set LKJCholesky prior on Lc
# ω ~ LogNormal
# L = LowerTriangular(Lc .* ω) == cholesky(Ω)
_chol_lower(a::Cholesky) = a.uplo === 'L' ? a.L : a.U'

"""Calculates the posterior over η, as well as over ω, ρ (reasonable approximation), and σ"""
function full_advi(ann, prob, population, 𝜙::NamedTuple, st; num_samples = 3, with_entropy = false)
    ζ, _ = ann(population.x, 𝜙.weights, st)
    b1 = inverse(VecCorrBijector())
    b2 = inverse(VecCholeskyBijector(:L))
    b3 = inverse(bijector(LogNormal()))
    d = 4 # num_theta
    m = 2 # num_rand_effects
    
    θ_μ = 𝜙.theta[1:d]
    θ_Lc = LowerTriangular(_chol_lower(b2(𝜙.theta[2d+1:end])) .* softplus.(𝜙.theta[d+1:2d]))
    q_θ = @ignore_derivatives MultivariateNormal(θ_μ, θ_Lc * θ_Lc')
    
    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        θ = θ_μ + θ_Lc * randn(Float32, d)
        σ_ω, logabsdetjac1 = with_logabsdet_jacobian(b3, θ[1:3]) # exp
        σ = σ_ω[1]
        ω = σ_ω[2:3]
        C, logabsdetjac2 = with_logabsdet_jacobian(b1, θ[4:end]) # vec → C
        logabsdetjac_theta = logabsdetjac1 + logabsdetjac2

        Ω = Symmetric(ω .* C .* ω')
        
        for i in eachindex(population)
            # Calculates p(y | η) + p(η) - q_𝜙(η)
            Lᵢ = LowerTriangular(_chol_lower(b2(𝜙.phi[2m+1:end, i])) .* softplus.(𝜙.phi[m+1:2m, i]))
            ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:2, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
        
        priors = logpdf(LogNormal(-3.f0, 1.f0), σ) + sum(logpdf.(LogNormal(-1.5f0, 1.f0), ω)) + logpdf(LKJ(2, 2.f0), C) + logabsdetjac_theta
        ELBO += (priors / num_samples) # p(ω)
        ELBO -= (logpdf(q_θ, θ) / num_samples) # log(abs(det(J_θ))) + { ℍ[q_𝜙] or -q_𝜙(θ) }
    end
    
    return ELBO
end
