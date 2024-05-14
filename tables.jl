import Distributions.cdf
import QuadGK: quadgk
import CSV

using Dates
using Statistics
using DataFrames
using Distributions
using LinearAlgebra

"""
Distance between two Gaussians (𝒲₂):
||m₁ - m₂||₂² + tr(C₁ + C₂ - 2sqrt(sqrt(C₂) * C₁ * sqrt(C₂)))
"""
function wasserstein_2p(P::T, Q::T) where T<: MultivariateNormal
    m₁, m₂, C₁, C₂ = P.μ, Q.μ, P.Σ, Q.Σ
    return sum(abs2, m₁ - m₂) + tr(C₁ + C₂ - 2 * sqrt(sqrt(C₂) * C₁ * sqrt(C₂)))
end

cdf(dist::SkewNormal, x::Real) = quadgk(t -> pdf(dist,t), -Inf, x)[1]

"""
Distance between two one-dim distributions (𝒲₁):
𝒲₁(P, Q) = (∫₀¹ |F₁⁻¹(x) - F₂⁻¹(x)|ᵖ dx)^¹/p = ∫₀¹ |F₁(x) - F₂(x)| dx if p = 1
"""
wasserstein_1p(P::T, Q::T) where T<:Distribution{Univariate, Continuous} = quadgk(x -> abs(cdf(P, x) - cdf(Q, x)), 0, 1)[1]

ρ_true = 0.45
Ω_true = [0.037 ρ_true * sqrt(0.037) * sqrt(0.017) ; ρ_true * sqrt(0.037) * sqrt(0.017) 0.017]
σ_true = 0.03

ms_to_min = 1 / convert(Millisecond, Minute(1)).value

################################################################################
##########                                                            ##########
##########                          Table 1                           ##########
##########                                                            ##########
################################################################################
# Final objective function value and KL divergences for each of the models
cov_mat(vec) = Symmetric(vec[1:2] .* [1 vec[3]; vec[3] 1] .* vec[1:2]')

types = ["FO", "FOCE2-slow", "VI-eta", "VI-eta-single-sample"]
p = MultivariateNormal(zeros(2), Ω_true)

mae(x::AbstractVector, y::AbstractVector) = mean(abs.(x - y))
mae(x::Real, y::AbstractVector) = mean(abs.(x .- y))

result = DataFrame()
cols = [:LL, :rmse_typ, :omega_1, :omega_2, :rho, :sigma]
for type in types
    file = "neural-mixed-effects-paper/new_approach/data/result_$(type)_inn.csv"
    df = DataFrame(CSV.File(file))
    df_group = groupby(filter(x -> x.replicate <= 5, df), [:replicate, :fold])

    res = DataFrame(vcat([mean(Matrix(group[end-20:end, cols]), dims=1) for group in df_group]...), :auto)
    rename!(res, cols)

    kls = [kldivergence(p, MultivariateNormal(zeros(2), cov_mat(Vector(res[i, 3:5])))) for i in 1:nrow(res)]
    
    append!(result, 
        DataFrame(
            type = type, 
            LL = median(res.LL),
            LL_std = std(res.LL),
            rmse = median(res.rmse_typ) * 100,
            rmse_std = std(res.rmse_typ) * 100,
            KL = median(kls),
            KL_std = std(kls),
            omega_1_mae = mae(sqrt(Ω_true[1]), res.omega_1),
            omega_1_mae_std = std(abs2.(sqrt(Ω_true[1]) .- res.omega_1)),
            omega_2_mae = mae(sqrt(Ω_true[4]), res.omega_2),
            omega_2_mae_std = std(abs.(sqrt(Ω_true[4]) .- res.omega_2)),
            sigma_mae = mae(σ_true, res.sigma) * 100,
            sigma_mae_std = std(abs.(σ_true .- res.sigma)) * 100,
        )
    )
end

file = "neural-mixed-effects-paper/new_approach/data/result_mse_inn.csv"
df = DataFrame(CSV.File(file))
res_mse = [mean(group[end-20:end, :rmse_typ].^2) for group in groupby(df, [:replicate, :fold])]
mean(res_mse)
std(res_mse)

result


# Training time:
types = ["mse", "FO", "FOCE2-slow/bfgs", "VI-eta", "VI-eta-single-sample"]
ann_type = "inn"
for type in types
    folder = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/$(ann_type)"
    files = readdir(folder)
    result = Vector{Dates.Millisecond}(undef, length(files))
    for (j, file) in enumerate(files)
        result[j] = BSON.raise_recursive(BSON.parse(joinpath(folder, file))[:duration], Main)
    end
    result = map(x -> x.value * ms_to_min, result)
    println("($type) Median training time: $(median(result)) +- $(std(result))")
end

################################################################################
##########                                                            ##########
##########                          Table 2                           ##########
##########                                                            ##########
################################################################################
# Results from the real world experiment
to_CV(ω) = sqrt(exp(ω^2) - 1) * 100

df_idx = 1
type = "FOCE2"
ann_type = "inn"
error = :prop

if type == "mse"
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(type)_$(ann_type).csv"))
else
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(type)_$(ann_type)_$(String(error)).csv"))
end
result = zeros(error == :prop ? 7 : 6, 10, 10)
for (j, outer_group) in enumerate(groupby(df, :outer))
    for (k, inner_group) in enumerate(groupby(DataFrame(outer_group), :inner))
        last_epoch = contains(type, "FOCE") ? 2000 : (contains(type, "FO") ? 1000 : (df_idx == 1 ? 1000 : 1250))
        last_idx = findfirst(==(last_epoch), inner_group.epoch)
        last_idx = last_idx == nothing ? findfirst(==(1750), inner_group.epoch) : last_idx
        idxs = last_idx-10:last_idx
        # result[1, j, k] = mean(inner_group[end-10:end, :mse])
        result[1, j, k] = mean(inner_group[idxs, :LL])
        result[2, j, k] = mean(inner_group[idxs, :rmse_typ] .* 100) # To IU/dL
        result[3, j, k] = mean(inner_group[idxs, :omega_1])
        result[4, j, k] = mean(inner_group[idxs, :omega_2])
        result[5, j, k] = mean(inner_group[idxs, :rho])
        if error == :add
            result[6, j, k] = mean(inner_group[idxs, :sigma] .* 100) # To IU/dL
        else
            result[6, j, k] = mean(inner_group[idxs, :sigma_1] .* 100) # To IU/dL
            result[7, j, k] = mean(inner_group[idxs, :sigma_2])
        end
    end
end

outer_result = hcat([mean(result[:, j, :], dims=2) for j in 1:10]...)

# df1
# MSE
# mse = 0.017 +- 0.0002
# rmse: 14.1 +- 0.236

# FO additive error
# minimal OBJF: -727.3
# FO proportional error
# minimal OBJF: -764 +- 13.3 <<<<<<
# rmse: 14.3 +- 0.77
# ω₁: 0.289 +- 0.044
# ω₂: 0.127 +- 0.020
# ρ: 0.657 +- 0.13
# add: 3.09 +- 0.43
# prop: 0.105 +- 0.013

# FOCE2 proportional error
# minimal OBJF: -683 +- 11.87
# rmse: 19.0 +- 4.29
# ω₁: 0.240 +- 0.0185
# ω₂: 0.465 +- 0.052
# ρ: 0.165 +- 0.069
# add: 3.695 +- 0.0488
# prop: 0.1075 +- 0.0042

# VI additive error
# minimal OBJF: -187.2 +- 1.4
# VI proportional error
# minimal OBJF: -210.4 +- 8.5 <<<<<<
# rmse: 14.3 +- 0.69
# ω₁: 0.282 +- 0.012
# ω₂: 0.160 +- 0.004
# ρ: 0.718 +- 0.034
# add: 2.89 +- 0.077
# prop: 0.094 +- 0.017

# df2
# MSE
# mse = 0.0864 +- 0.0025
# rmse: 27.559 +- 1.125

# FO add:
# final OBJF: -1114.0
# FO prop:
# final OBJF: -1318 +- 72.4 <<<<<
# rmse: 32.0 +- 1.66
# ω₁: 0.300 +- 0.012
# ω₂: 0.211 +- 0.018
# ρ: 0.526 +- 0.038
# add: 2.89 +- 1.63
# prop: 0.151 +- 0.012

# FOCE2 proportional error
# minimal OBJF: -1180 +- 14
# rmse: 31.5 +- 1.66
# ω₁: 0.321 +- 0.0140
# ω₂: 0.326 +- 0.0196
# ρ: 0.0882 +- 0.0177
# add: 4.53 +- 0.3666
# prop: 0.152 +- 0.00455


# VI add: 
# final OBJF: -83.9
# VI prop:
# final OBJF: -198 +- 1.53
# rmse: 30.0 +- 1.17
# ω₁: 0.3155 +- 0.0052
# ω₂: 0.179 +- 0.0007
# ρ: 0.579 +- 0.0071
# add: 2.46 +- 0.0238
# prop: 0.165 +- 0.00059

df_idx = 2
for type in ["mse", "fo", "foce", "vi-new"]
    folder = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$df_idx/$type"
    files = readdir(folder)
    result = Vector{Dates.Millisecond}(undef, length(files))
    for (j, file) in enumerate(files)
        if type == "foce"
            ckpt = BSON.load(joinpath(folder, file))
            result[j] = ckpt[:duration] + ckpt[:duration_second]
        else
            result[j] = BSON.raise_recursive(BSON.parse(joinpath(folder, file))[:duration], Main)
        end
    end
    times = map(x -> x.value * ms_to_min, result)
    println("($type) Median training time: $(median(times)) +- $(std(times))")
end


################################################################################
##########                                                            ##########
##########                       Supp. table 1                        ##########
##########                                                            ##########
################################################################################
# Results of the comparison between the variational approximation and MCMC posteriors.
idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_1.csv")).idxs
folder = "neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known"
folder2 = "neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known"

function getq(𝜙) 
    μ = 𝜙[1:2]
    σ = softplus.(𝜙[3:4])
    L = LowerTriangular(inverse(VecCholeskyBijector(:L))(𝜙[3:4]).L .* σ)
    Σ = L * L'
    return MultivariateNormal(μ, Σ)
end

function getq_theta(θ; d = 4)
    μ = θ[1:d]
    σ = softplus.(θ[d+1:2d])
    Lc = inverse(VecCholeskyBijector(:L))(θ[2d+1:end]).L
    L = Lc .* σ
    return MultivariateNormal(μ, L * L')
end

function skewfit(x)
    opt = Optim.optimize(p -> sum(-logpdf.(SkewNormal(p[1], softplus(p[2]), p[3]), x)), rand(3))
    p_opt = opt.minimizer
    return SkewNormal(p_opt[1], softplus(p_opt[2]), p_opt[3])
end


full_chain = BSON.load(joinpath(folder2, "mcmc", "fold_1_5000_samples.bson"))[:chain]
P_sigma = fit(LogNormal, group(full_chain, :σ).value.data[:, 1, 1])
P_omega1 = fit(LogNormal, group(full_chain, :ω).value.data[:, 1, 1])
P_omega2 = fit(LogNormal, group(full_chain, :ω).value.data[:, 2, 1])
P_rho = skewfit(group(full_chain, :ρ).value.data[:, 1, 1] .- 1)

# indexes are -> k, d, n, j 
# k = experiment [true_pk_pop_params_known, true_pk_known]
# d = replicate
# n = individuals
# (j) = [μ₁, μ₂]
entropy_ = zeros(2, 20, length(idxs))
var_post = zeros(2, 20, length(idxs))
path_deriv = zeros(2, 20, length(idxs))

theta_entropy = zeros(20, 4)
theta_var_post = zeros(20, 4)
theta_path_deriv = zeros(20, 4)
for k in 1:2
    for (i, idx) in enumerate(idxs)
        # When PK and pop params are known:
        chain = k == 1 ? BSON.load(joinpath(folder, "mcmc", "idx_$idx.bson"))[:chain] : full_chain[[Symbol("η[1,$i]"), Symbol("η[2,$i]")]]
        P = fit(MultivariateNormal, transpose(chain.value.data[:, 1:2, 1]))
        for replicate in 1:20
            println("$k, running for i = $idx, replicate = $replicate")
            # When PK and pop params are known:
            params_entropy = BSON.load(joinpath(k == 1 ? folder : folder2, "vi-entropy", "fold_1_replicate_$replicate.bson"))[:parameters]
            if entropy_[k, replicate, i] == 0 || entropy_bias[k, replicate, i, :] == 0
                phi_entropy = params_entropy.phi[:, i]
                entropy_[k, replicate, i] = wasserstein_2p(P, getq(phi_entropy))
            end
            
            params_var_post = BSON.load(joinpath(k == 1 ? folder : folder2, "vi-var-post", "fold_1_replicate_$replicate.bson"))[:parameters]
            if var_post[k, replicate, i] == 0 || var_post_bias[k, replicate, i, :] == 0
                phi_var_post = params_var_post.phi[:, i]
                var_post[k, replicate, i] = wasserstein_2p(P, getq(phi_var_post))
            end
            
            params_path_deriv = BSON.load(joinpath(k == 1 ? folder : folder2, "vi-path-deriv", "fold_1_replicate_$replicate.bson"))[:parameters]
            if path_deriv[k, replicate, i] == 0 || path_deriv_bias[k, replicate, i, :] == 0
                phi_path_deriv = params_path_deriv.phi[:, i]
                path_deriv[k, replicate, i] = wasserstein_2p(P, getq(phi_path_deriv))
            end

            if k == 2 && i == 1
                # Add acc of theta
                if all(theta_entropy[replicate, :] .== 0)
                    q_entropy = getq_theta(params_entropy.theta)
                    samples_entropy = rand(q_entropy, 10_000)
                    theta_entropy[replicate, 1] = wasserstein_1p(P_sigma, fit(LogNormal, exp.(samples_entropy[1, :])))
                    theta_entropy[replicate, 2] = wasserstein_1p(P_omega1, fit(LogNormal, exp.(samples_entropy[2, :])))
                    theta_entropy[replicate, 3] = wasserstein_1p(P_omega2, fit(LogNormal, exp.(samples_entropy[3, :])))
                    theta_entropy[replicate, 4] = wasserstein_1p(P_rho, skewfit(map(x -> inverse(VecCorrBijector())([x])[2], samples_entropy[4, :])))
                end

                if all(theta_var_post[replicate, :] .== 0)
                    q_var_post = getq_theta(params_var_post.theta)
                    samples_var_post = rand(q_var_post, 10_000)
                    theta_var_post[replicate, 1] = wasserstein_1p(P_sigma, fit(LogNormal, exp.(samples_var_post[1, :])))
                    theta_var_post[replicate, 2] = wasserstein_1p(P_omega1, fit(LogNormal, exp.(samples_var_post[2, :])))
                    theta_var_post[replicate, 3] = wasserstein_1p(P_omega2, fit(LogNormal, exp.(samples_var_post[3, :])))
                    theta_var_post[replicate, 4] = wasserstein_1p(P_rho, skewfit(map(x -> inverse(VecCorrBijector())([x])[2], samples_var_post[4, :])))
                end

                if all(theta_path_deriv[replicate, :] .== 0)
                    q_path_deriv = getq_theta(params_path_deriv.theta)
                    samples_path_deriv = rand(q_path_deriv, 10_000)
                    theta_path_deriv[replicate, 1] = wasserstein_1p(P_sigma, fit(LogNormal, exp.(samples_path_deriv[1, :])))
                    theta_path_deriv[replicate, 2] = wasserstein_1p(P_omega1, fit(LogNormal, exp.(samples_path_deriv[2, :])))
                    theta_path_deriv[replicate, 3] = wasserstein_1p(P_omega2, fit(LogNormal, exp.(samples_path_deriv[3, :])))
                    theta_path_deriv[replicate, 4] = wasserstein_1p(P_rho, skewfit(map(x -> inverse(VecCorrBijector())([x])[2], samples_path_deriv[4, :])))
                end
            end
        end
    end
end


# Wasserstein distances:
##### ENTROPY BASED VI
# eta
mean(mean(entropy_[1, :, :], dims=1)) # 9.0e-3
std(mean(entropy_[1, :, :], dims=1)) # 3.6e-3
# eta
mean(mean(entropy_[2, :, :], dims=1)) # 8.5e-3
std(mean(entropy_[2, :, :], dims=1)) # 3.1e-3
# theta
mean(theta_entropy, dims=1) # 4.6e-3, 9.4e-3, 9.2e-3, 46.7e-3
std(theta_entropy, dims=1) # 0.8e-3, 5.3e-3, 3.6e-3, 14.4e-3

##### VARIATIONAL POSTERIOR BASED VI
# eta
mean(mean(var_post[1, :, :], dims=1)) # 9.0e-3
std(mean(var_post[1, :, :], dims=1)) # 3.5e-3
# eta
mean(mean(var_post[2, :, :], dims=1)) # 8.8e-3
std(mean(var_post[2, :, :], dims=1)) # 3.3e-3
# theta
mean(theta_var_post, dims=1) # 5.0e-3, 7.2e-3, 10.9e-3, 46.9e-3 
std(theta_var_post, dims=1) # 0.6e-3, 3.1e-3, 3.2e-3, 23.9e-3

##### PATH DERIVATIVE ESTIMATOR BASED VI
# eta
mean(mean(path_deriv[1, :, :], dims=1)) # 5.5e-3
std(mean(path_deriv[1, :, :], dims=1)) # 2.9e-3
# eta
mean(mean(path_deriv[2, :, :], dims=1)) # 4.5e-3
std(mean(path_deriv[2, :, :], dims=1)) # 2.3e-3
# theta
mean(theta_path_deriv, dims=1) # 0.9e-3, 6.7e-3, 7.2e-3, 43.4e-3
std(theta_path_deriv, dims=1) # 0.3e-3, 3.5e-3, 4.0e-3, 14.3e-3