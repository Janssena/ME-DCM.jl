import Bijectors: VecCholeskyBijector, inverse, VecCorrBijector
import LinearAlgebra: LowerTriangular, I, Symmetric
import Plots
import BSON
import CSV

include("neural-mixed-effects-paper/new_approach/model.jl");

using Turing
using Bijectors
using Statistics
using StatsPlots
using DataFrames
using KernelDensity
using Distributions
using Plots.Measures

colorscheme = :default
font = "Computer Modern"
Plots.default(fontfamily=font, titlefontsize=12, framestyle=:box, grid=false, tickdirection=:out)

Ω_true = [0.037 0.45 * sqrt(0.037) * sqrt(0.017) ; 0.45 * sqrt(0.037) * sqrt(0.017) 0.017]

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
leaky_softplus(x::Real; α=0.05, β=10.f0) = α * x + (1 - α) * smooth_relu(x; β)



################################################################################
##########                                                            ##########
##########                         Figure 1                           ##########
##########                                                            ##########
################################################################################
"""Overview p(y)"""

df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/simulation.csv"))

id = "virtual_2"
subject = df[df.id .== id, :]
cl = 0.1 .* (subject.wt[1] ./ 70).^0.75 .* (1 / 55 .* leaky_softplus.(-subject.vwf[1] .+ 100; α=0.05, β=0.1) .+ 0.9)
v1 = 2. .* (subject.wt[1] ./ 70)
q = 0.15
v2 = 0.75
p_typ = [cl, v1, q, v2] 
cb = generate_dosing_callback(Matrix{Float64}(subject[.!ismissing.(subject.amt), [:time, :amt, :rate, :duration]]); S1=1/1000)
individual = Individual(p_typ, Float64.(subject[.!ismissing.(subject.dv), :dv]), Float64.(subject[.!ismissing.(subject.dv), :time]), cb; id=id)

Ω_true = [0.037 0.45 * sqrt(0.037) * sqrt(0.017) ; 0.45 * sqrt(0.037) * sqrt(0.017) 0.017]
σ_true = 3. / 100

prob = ODEProblem(two_comp!, zeros(2), (-0.1, 1.))

@model function bayesian_model(prob, individual, y; Ω = Ω_true, σ = σ_true)
    eta ~ Normal(0, sqrt(Ω[1])) # MultivariateNormal(zeros(2), Ω)
    if any(abs.(eta) .> 3) return Turing.@addlogprob! -Inf end
    z = individual.x .* exp.([1 0; 0 1; 0 0; 0 0] * [eta, 0.004119878281702493])
    ŷ = predict_adjoint(prob, individual, z)
    y ~ MultivariateNormal(ŷ, σ)
end

m = bayesian_model(prob, individual, individual.y);
chain = sample(m, NUTS(), 10_000)

mode = Optim.optimize(m, MAP()).optim_result.minimizer

function logjoint_(prob, individual, eta; Ω = Ω_true, σ = σ_true) 
    # prior = MultivariateNormal(zeros(2), Ω)
    prior = Normal(0, sqrt(Ω[1]))
    z = individual.x .* exp.([1 0; 0 1; 0 0; 0 0] * [eta[1], 0.004119878281702493])
    ŷ = predict_adjoint(prob, individual, z)
    return logpdf(MultivariateNormal(ŷ, σ), individual.y) + logpdf(prior, eta[1])
end

H(η₀) = ForwardDiff.hessian(eta -> logjoint_(prob, individual, eta), η₀)

# laplace_approx = MultivariateNormal(mode, -(H(mode))^(-1))
laplace_approx = Normal(mode[1], sqrt(inv(-H(mode))[1]))
FO_approx = MultivariateNormal(zeros(2), Symmetric(-(H(zeros(2)))^(-1)))

mcmc_posterior = fit(Normal, chain.value.data[:, 1, 1]')

x = sort(chain.value.data[:, 1, 1])
p_y = [exp(logjoint_(prob, individual, eta)) for eta in x]
Plots.plot(x, p_y)

covellipse(FO_approx.μ, FO_approx.Σ)
covellipse!(laplace_approx.μ, laplace_approx.Σ)
covellipse!(mcmc_posterior.μ, mcmc_posterior.Σ, color=:black, fillalpha=0, alpha=1, linewidth=2, linestyle=:dash)


Plots.plot(chain)

predict_adjoint(prob, individual, individual.x .* exp.([1 0; 0 1; 0 0; 0 0] * eta))




################################################################################
##########                                                            ##########
##########                         Figure 1                           ##########
##########                                                            ##########
################################################################################
# Comparison of MCMC posteriors to VI posteriors
folder = "neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known"
# folder2 = "neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known"
idxs = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_1.csv")).idxs

i = 1
chain = BSON.load(joinpath(folder, "mcmc", "idx_$(idxs[i]).bson"))[:chain];

# Standard VI algorithm:
plt1 = Plots.plot()
for replicate in 1:20
    ckpt_var_post = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-var-post/fold_1_replicate_$replicate.bson")
    p = ckpt_var_post[:parameters]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]).L
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    covellipse!(plt1, q.μ, q.Σ, color=:orange, fillalpha=0.1, linewidth=1.4, n_std=1.96, label=nothing)
end
plt1

P = fit(MultivariateNormal, chain.value.data[:, 1:2, 1]')
covellipse!(plt1, P.μ, P.Σ, color=:black, linestyle=:dash, linewidth=2, alpha=1, fillalpha=0, n_std=1.96, label=nothing)

# Path. deriv:
plt2 = Plots.plot()
for replicate in 1:20
    ckpt_path_deriv = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_pop_params_known/vi-path-deriv/fold_1_replicate_$replicate.bson")
    p = ckpt_path_deriv[:parameters]
    μ = p.phi[1:2, i]
    ω = softplus.(p.phi[3:4, i])
    Lc = inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]).L
    Lᵢ = LowerTriangular(Lc .* ω)
    q = MultivariateNormal(μ, Lᵢ * Lᵢ')
    covellipse!(plt2, q.μ, q.Σ, color=:green, fillalpha=0.05, linewidth=1.4, n_std=1.96, label=replicate == 1 ? "Variational approximation" : nothing)
end
plt2

covellipse!(plt2, P.μ, P.Σ, color=:black, linestyle=:dash, linewidth=2, alpha=1, fillalpha=0, n_std=1.96, label=nothing)

Plots.plot!(plt2, [-1, 1], [-1, -1], linewidth=2, color=:black, linestyle=:dash, label="MCMC posterior")
# Makes it clear that we should use path. deriv estimator:
Plots.plot(plt1, plt2, layout=(1, 2), size=(450, 225), xlim=(-0.45, 0.11), ylim=(-0.35, 0.05), xticks = false, yticks = false, colorbar = false, title=["Standard" "Path deriv. estimator"], xlabel="Eta 1", ylabel=["Eta 2" ""], leftmargin=6mm, bottommargin=3mm)
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_1.svg")


################################################################################
##########                                                            ##########
##########                         Figure 2                           ##########
##########                                                            ##########
################################################################################
# Model Obj. fn value, rmse, and KL divergence during optimization over fold

models = ["FO", "FOCE2-slow", "VI-eta"]

plta = Plots.plot(ylabel="Obj. function value")
pltb = Plots.plot()
pltc = Plots.plot()
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    LL = [group.LL for group in groupby(df, [:fold, :replicate])]
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    for j in 1:length(LL)
        x = 0:25:Integer((length(LL[j])-1)*25)
    #     Plots.plot!(plt, x, LL[j], alpha=0.2, label=nothing, color=i == 4 ? (:grey) : Plots.palette(colorscheme)[i])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [LL[j][end]], markershape=:x, color=Plots.palette(colorscheme)[i], markerstrokewidth=1.4, label=nothing)
        end
    end
    res = hcat(filter(ll -> length(ll) == maximum(length.(LL)), LL)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(LL[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
Plots.plot!(plta, ylim=(-1100, 0))
Plots.plot!(pltb, ylim=(-1100, 0))
Plots.plot!(pltc, ylim=(-345, -30))
obj = Plots.plot(plta, pltb, pltc, layout=(1, 3), size =(800, 200), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000))

# KL DIVERGENCE
plta = Plots.plot()
pltb = Plots.plot()
pltc = Plots.plot()
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    df_group = groupby(df, [:replicate, :fold])
    kl = Vector{Vector{Union{Float64, Missing}}}(undef, length(df_group))
    for (j, group) in enumerate(df_group)
        res = zeros(Union{Float64, Missing}, nrow(group))
        for k in 1:length(res)
            Ω_pred = covariance_matrix([group[k, :rho]], Vector(group[k, [:omega_1, :omega_2]]))
            try 
                res[k] = log(kldivergence(MultivariateNormal(zeros(2), Ω_true), MultivariateNormal(zeros(2), Ω_pred)))
            catch e
                res[k] = missing
            end
        end
        kl[j] = res
    end

    for j in 1:length(kl)
        x = 0:25:Integer((length(kl[j])-1)*25)
        # if contains(model_type, "FOCE")
        # Plots.plot!(plt, x, kl[j], alpha=0.2, label=nothing, color=Plots.palette(colorscheme)[i])
        # end
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [kl[j][end]], markershape=:x, color=Plots.palette(colorscheme)[i], markerstrokewidth=1.4, label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(kl)), kl)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(kl[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
kl_omega = Plots.plot(plta, pltb, pltc, layout=(1, 3), size = (800, 200), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000), ylim=(-8, 5), ylabel=["log KL(P || Q)" "" ""])


plta = Plots.hline([3], color=:black, linestyle=:dash, label=nothing, linewidth=1.4, ylabel="Residual error\nestimate")
pltb = Plots.hline([3], color=:black, linestyle=:dash, label=nothing, linewidth=1.4)
pltc = Plots.hline([3], color=:black, linestyle=:dash, label=nothing, linewidth=1.4)
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    sigma = [group.sigma .* 100 for group in groupby(df, [:replicate, :fold])]
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end

    for j in 1:length(sigma)
        x = 0:25:Integer((length(sigma[j])-1)*25)
        # if contains(model_type, "FOCE")
        #     Plots.plot!(plt, x, sigma[j], alpha=0.2, label=nothing, linestyle=:dash, color=i == 4 ? (:grey) : Plots.palette(colorscheme)[i])
        # end
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [sigma[j][end]], markershape=:x, color=Plots.palette(colorscheme)[i], markerstrokewidth=1.4, label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(sigma)), sigma)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(sigma[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
sigma = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(1, 3), size = (800, 200), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000), ylim=(0, 16.8), ylabel=["Sigma (IU/dL)" "" ""])

figure_2 = Plots.plot(obj, kl_omega, sigma, layout=(3, 1), size=(600, 500), yticks=false, bottommargin=0mm)
# Plots.plot(obj, kl_omega, sigma, layout=(3, 1), size=(600, 500), bottommargin=0mm)
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_2.svg")


################################################################################
##########                                                            ##########
##########                         Figure 3                           ##########
##########                                                            ##########
################################################################################
# Learned functions during real world experiment.
smooth_relu(x::Real; β=10.f0) = 1 / β * softplus(β * x)
leaky_softplus(x::Real; α=0.05, β=10.f0) = α * x + (1 - α) * smooth_relu(x; β)

df_idx = 1

# get covariate data
df1 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df1_prophylaxis_imputed.csv"))
df2 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df2_surgery_imputed.csv"))
df = [df1, df2][df_idx]
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

models = ["mse", "fo", "foce", "vi-slow"] # "variational-partial"]
epoch_idxs = Integer.([0, 250, 500, 1000, 1500] ./ 25 .+ 1) # epoch / save_every + 1

epoch_1000_ffm_cl = typeof(Plots.plot())[]
epoch_1000_ffm_v1 = typeof(Plots.plot())[]
epoch_1000_vwf_cl = typeof(Plots.plot())[]
color_idxs = (mse = 4, fo = 1, foce = 2, vi_slow = 3)

for (k, type) in enumerate(models)
    # Gather data:
    color_idx = color_idxs[Symbol(replace(type, "-" => "_"))] 
    dummy = vcat(collect.([range(0, 1, 40)', range(0, 1, 40)'])...)
    result_file = "neural-mixed-effects-paper/new_approach/data/learned_functions_real_world_df$(df_idx)_$(type == "mse" ? type : "$(type)_prop").bson"
    if !isfile(result_file)
        folder = "neural-mixed-effects-paper/new_approach/checkpoints/real_world/df$(df_idx)/$(type)"
        files = readdir(folder)
        if type !== "mse"
            filter!(contains("prop"), files)
        end
        res = zeros(length(epoch_idxs), length(files), 3, size(dummy, 2))
        for (i, epoch_idx) in enumerate(epoch_idxs)
            for (j, file) in enumerate(files)
                ckpt = BSON.load(joinpath(folder, file));
                weights = ckpt[:saved_parameters][epoch_idx].weights

                preds_ffm, _ = ckpt[:model][2].layers[1](dummy, weights[2][1], ckpt[:state][2][1])
                norm_ffm, _ = ckpt[:model][2].layers[1](Float32[60/150;;], weights[2][1], ckpt[:state][2][1])
                res[i, j, 1:2, :] = preds_ffm ./ norm_ffm

                preds_vwf, _ = ckpt[:model][2].layers[2](dummy, weights[2][2], ckpt[:state][2][2])
                norm_vwf, _ = ckpt[:model][2].layers[2](Float32[0; 1 / 3.5;;], weights[2][2], ckpt[:state][2][2])
                res[i, j, 3, :] = preds_vwf ./ norm_vwf[1]
            end
        end
        BSON.bson(result_file, Dict(:matrix => res))
    end

    res = BSON.load(result_file)[:matrix]

    # Make plot:
    # wt on cl
    # plts = typeof(Plots.plot())[]
    # for i in eachindex(epoch_idxs)        
    #     plt = Plots.plot(ylabel=i == 1 ? "Weight on\nclearance" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[1, :] .* 150, res[i, j, 1, :], color=Plots.palette(colorscheme)[color_idx], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     push!(plts, plt)
    # end
    # plta = Plots.plot(plts..., layout=(1, 5), ylim=(0, 2.3), title=["Before training" "Epoch 250" "Epoch 500" "Epoch 1000" "Epoch 2000"], xlim=(0, 100));
    
    ci = hcat([quantile(res[end, :, 1, j], [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(res[end, :, 1, j]) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[1, :] .* 150, med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, ylim=(0, 2.3), xlabel="Fat-free mass (kg)", color=Plots.palette(colorscheme)[color_idx])
    Plots.histogram!(Plots.twinx(), population.x[1, :], linecolor=:lightgrey, color=:lightgrey, legend=false, xaxis=false, yaxis=false, ylim=(0, 40), bins=40)
    push!(epoch_1000_ffm_cl, plt)
    
    # wt on v1
    # plts = typeof(Plots.plot())[]
    # for i in eachindex(epoch_idxs)
    #     plt = Plots.plot(ylabel=i == 1 ? "Weight on\nvolume of distribution" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[1, :] .* 150, res[i, j, 2, :], color=Plots.palette(colorscheme)[color_idx], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     # Plots.plot!(plt, dummy[1, :], median(res[i, :, 2, :], dims=1)[1, :], linewidth=3, color=Plots.palette(colorscheme)[k], label=nothing)
    #     push!(plts, plt)
    # end
    # pltb = Plots.plot(plts..., layout=(1, 5), ylim=(0, 2.3), xlim=(0, 100));

    ci = hcat([quantile(res[end, :, 2, j], [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(res[end, :, 2, j]) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[1, :] .* 150, med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, ylim=(0, 2.3), xlabel="Fat-free mass (kg)", color=Plots.palette(colorscheme)[color_idx])
    Plots.histogram!(Plots.twinx(), population.x[1, :], linecolor=:lightgrey, color=:lightgrey, legend=false, xaxis=false, yaxis=false, ylim=(0, 40), bins=40)
    push!(epoch_1000_ffm_v1, plt)
    
    # vwf on cl
    # plts = typeof(Plots.plot())[]
    # for i in eachindex(epoch_idxs)
    #     plt = Plots.plot(ylabel=i == 1 ? "VWF on\nclearance" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[2, :] .* 350, res[i, j, 3, :], color=Plots.palette(colorscheme)[color_idx], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     push!(plts, plt)
    # end
    # pltc = Plots.plot(plts..., layout=(1, 5), ylim=(0, 4));

    ci = hcat([quantile(res[end, :, 3, j], [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(res[end, :, 3, j]) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[2, :] .* 350, med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, ylim=(0, 3.5), xlabel="VWF:Ag (%)", color=Plots.palette(colorscheme)[color_idx])
    Plots.histogram!(Plots.twinx(), population.x[2, :] .* 100, linecolor=:lightgrey, color=:lightgrey, legend=false, xaxis=false, yaxis=false, ylim=(0, 35), bins=20)
    push!(epoch_1000_vwf_cl, plt)

    # plt_ = Plots.plot(plta, pltb, pltc, layout=(3, 1), size=(950, 475), xticks=false, yticks=false, leftmargin=8mm)
    # Plots.savefig("neural-mixed-effects-paper/new_approach/plots/learned_functions_$(replace(type, "/" => "_")).svg")

    # display(plt_)
end

top = Plots.plot(epoch_1000_ffm_cl..., layout=(1, 4), linewidth=3, size=(1050, 200), xlim=(0, 100), ylabel=["Fold change\nin clearance" "" "" ""], legend=false)
middle = Plots.plot(epoch_1000_ffm_v1..., layout=(1, 4), linewidth=3, size=(1050, 200), xlim=(0, 100), ylabel=["Fold change in\nvolume of distribution" "" "" ""], legend=false)
bottom = Plots.plot(epoch_1000_vwf_cl..., layout=(1, 4), linewidth=3, size=(1050, 200), xlim=(0, 350), ylabel=["Fold change\nin clearance" "" "" ""], legend=false)

Plots.plot(top, middle, bottom, layout = (3, 1), size=(1050, 600), leftmargin=4mm, bottommargin=3mm, legend=false, framestyle=:box)
# Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_4_df$(df_idx).svg")
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s6_df$(df_idx).svg")