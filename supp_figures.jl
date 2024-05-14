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

################################################################################
##########                                                            ##########
##########                      Supp. Figure 1                        ##########
##########                                                            ##########
################################################################################
# VI posteriors in the full advi case (MCMC comparison)

## Accuracy of population parameter posteriors:
full_chain = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/mcmc/fold_1_5000_samples.bson")[:chain]

function getq_theta(θ; d = 4)
    μ = θ[1:d]
    σ = softplus.(θ[d+1:2d])
    Lc = inverse(VecCholeskyBijector(:L))(θ[2d+1:end]).L
    L = Lc .* σ
    return MultivariateNormal(μ, L * L')
end

plt1 = Plots.plot(xlabel = "Sigma")
plt2 = Plots.plot(xlabel = "Omega 1")
plt3 = Plots.plot(xlabel = "Omega 2")
plt4 = Plots.plot(xlabel = "Correlation")
folder = "neural-mixed-effects-paper/new_approach/checkpoints/mcmc/true_pk_known/vi-path-deriv"
for (j, file) in enumerate(readdir(folder))
    parameters = BSON.load(joinpath(folder, file))[:parameters]
    q = getq_theta(parameters.theta)
    for i in 1:4
        plt = [plt1, plt2, plt3, plt4][i]
        if i < 4
            samples = exp.(rand(q, 100_000)[i, :])
        else
            samples = map(x -> inverse(VecCorrBijector())([x])[2], rand(q, 10_000)[4, :]) # For rho
        end
        U = kde(samples)
        Plots.plot!(plt, U, ribbon=(U.density, zero.(U.density)), color=:green, alpha=0.5, fillalpha=0.05, label=nothing)
        if i == 1
            deeter = group(full_chain, :σ).value.data[:, 1, 1]
        elseif i == 2
            deeter = group(full_chain, :ω).value.data[:, 1, 1]
        elseif i == 3
            deeter = group(full_chain, :ω).value.data[:, 2, 1]
        else
            deeter = group(full_chain, :ρ).value.data[:, 1, 1] .- 1
        end
        Plots.plot!(Plots.twinx(), yaxis=false, kde(deeter), linewidth=2, linestyle=:dash, color=:black, label=nothing)
    end
end

top = Plots.plot(plt1, plt2, plt3, plt4, layout=(1, 4), size=(1200, 250), yticks = false, bottommargin=5mm)
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s1_top.svg")

files = readdir(folder)
ckpts = map(file -> BSON.load(joinpath(folder, file)), files)
parameters = BSON.load(joinpath(folder, files[1]))[:parameters]
plts = Vector{typeof(Plots.plot())}(undef, length(idxs))
for (i, idx) in enumerate(idxs)
    mcmc = full_chain[[Symbol("η[1,$i]"), Symbol("η[2,$i]")]]
    plt = Plots.plot()
    for replicate in 1:20
        p = ckpts[replicate][:parameters]
        μ = p.phi[1:2, i]
        ω = softplus.(p.phi[3:4, i])
        Lc = inverse(VecCholeskyBijector(:L))(p.phi[5:end, i]).L
        Lᵢ = LowerTriangular(Lc .* ω)
        q = MultivariateNormal(μ, Lᵢ * Lᵢ')
        covellipse!(plt, q.μ, q.Σ, color=:green, fillalpha=0.05, n_std=1.96, label=replicate == 1 ? "Variational approximation" : nothing)
    end
    P = fit(MultivariateNormal, mcmc.value.data[:, :, 1]')
    covellipse!(plt, P.μ, P.Σ, color=:black, linestyle=:dash, linewidth=2, alpha=1, fillalpha=0, n_std=1.96, label=nothing)
    # Plots.plot!(plt, kde(mcmc.value.data[:, 1:2, 1], bandwidth=(0.03, 0.03)), color=:black, linewidth=2, levels=3, colorbar = false)
    plts[i] = plt
end

bottom = Plots.plot(plts..., legend=false, xticks = false, yticks = false, size=(1200, 1200))
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s1_bottom.svg")


################################################################################
##########                                                            ##########
##########                      Supp. Figure 2                        ##########
##########                                                            ##########
################################################################################
# Comparison of FOCE1 and FOCE2 objectives

models = ["FOCE1-slow", "FOCE2-slow"]

plta = Plots.plot(ylabel="Objective function value", title = "FOCE (Eq. 1)")
pltb = Plots.plot(title = "FOCE (Eq. 4b) lower learning rate")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type)_bfgs_inn.csv"))
    LL = [group.LL for group in groupby(df, [:replicate, :fold])]
    if startswith(model_type, "FOCE1")
        plt = plta
    else
        plt = pltb
    end
    for j in 1:length(LL)
        x = 0:25:Integer((length(LL[j])-1)*25)
        Plots.plot!(plt, x, LL[j], alpha=0.1, label=nothing, linewidth=1.4, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [LL[j][end]], markershape=:x, markerstrokewidth=2, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2], label=nothing)
        end
    end
    # Plots.plot!(0:25:Integer(25*(length(LL[1])-1)), median(hcat(LL...), dims=2)[:, 1], label=nothing, linewidth=2, color=i == 4 ? (:grey) : Plots.palette(colorscheme)[i])
end
obj = Plots.plot(plta, pltb, xlabel="Epoch", layout=(2, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000), ylim=(-1100, 1000), ylabel="Neg. loglikelihood")
# Plots.plot(plta, pltb, pltc, plt1, xlabel="Epoch", layout=(1, 4), size = (800, 200), leftmargin=5mm, bottommargin=5mm, ylim=(5., 10), xlim=(0, 3000))

# KL DIVERGENCE
plta = Plots.plot(ylabel="KL(P || Q)", title = "FOCE (Eq. 1)")
pltb = Plots.plot(title = "FOCE (Eq. 4b) lower learning rate")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type)_bfgs_inn.csv"))
    if startswith(model_type, "FOCE1")
        plt = plta
    else
        plt = pltb
    end
    kl = Vector{Vector{Union{Float64, Missing}}}(undef, length(groupby(df, [:replicate, :fold])))
    for (j, group) in enumerate(groupby(df, [:replicate, :fold]))
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
        Plots.plot!(plt, x, kl[j], alpha=0.1, label=nothing, linewidth=1.4, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [kl[j][end]], markershape=:x, markerstrokewidth=2, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2], label=nothing)
        end
    end
    # Plots.plot!(0:25:2000, median(hcat(kl...), dims=2)[:, 1], label=nothing, linewidth=2, color=Plots.palette(colorscheme)[i])
end
kl_omega = Plots.plot(plta, pltb, xlabel="Epoch", layout=(2, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000), ylim=(-8, 5), ylabel="log KL(P || Q)")

plta = Plots.hline([3], color=:black, linestyle=:dash, label=nothing, ylabel="Residual error\nestimate", title = "FOCE (Eq. 1)")
pltb = Plots.hline([3], color=:black, linestyle=:dash, label=nothing, title = "FOCE (Eq. 4b) reduced learning rate")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type)_bfgs_inn.csv"))
    sigma = [group.sigma .* 100 for group in groupby(df, [:replicate, :fold])]
    if startswith(model_type, "FOCE1")
        plt = plta
    else
        plt = pltb
    end
    for j in 1:length(sigma)
        x = 0:25:Integer((length(sigma[j])-1)*25)
        Plots.plot!(plt, x, sigma[j], alpha=0.1, label=nothing, linewidth=1.4, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [sigma[j][end]], markershape=:x, markerstrokewidth=2, color=Plots.palette(colorscheme)[startswith(model_type, "FOCE1") ? 4 : 2], label=nothing)
        end
    end
    # Plots.plot!(0:25:2000, median(hcat(sigma...), dims=2)[:, 1], label=nothing, linewidth=2, color=i == 4 ? (:grey) : Plots.palette(colorscheme)[i])
end
sigma = Plots.plot(plta, pltb, xlabel="Epoch\n", layout=(2, 1), size = (300, 800), xlim=(-50, 2000), ylim=(0, 16.8), ylabel="Sigma (IU/dL)")

Plots.plot(obj, kl_omega, sigma, layout=(1, 3), size=(800, 400), title = "")
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s2.svg")

################################################################################
##########                                                            ##########
##########                      Supp. Figure 3                        ##########
##########                                                            ##########
################################################################################
# Accuracy of subcomponents of Omega matrix
models = ["FOCE2-slow"]

plt = Plots.plot(ylabel="log KL(P || Q)")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
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
        # Plots.plot!(plt, x, kl[j], alpha=0.2, linewidth=1.4, label=nothing, color=Plots.palette(colorscheme)[2])
        # end
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [kl[j][end]], markershape=:x, markerstrokewidth=3, color=Plots.palette(colorscheme)[2], label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(kl)), kl)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(kl[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i+1])
end
kl_omega = Plots.plot(plt, ylim=(-8, 5))
# kl_omega = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=(-50, 2000), ylim=(-0.25, 2), ylabel="KL(P || Q)")

plt = Plots.plot(ylabel="Omega 1")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    omega_1 = [group.omega_1 for group in groupby(df, [:replicate, :fold])]
    for j in 1:length(omega_1)
        x = 0:25:Integer((length(omega_1[j])-1)*25)
        # Plots.plot!(plt, x, omega_1[j], alpha=0.2, label=nothing, color=Plots.palette(colorscheme)[2])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [omega_1[j][end]], markershape=:x, markerstrokewidth=3, color=Plots.palette(colorscheme)[i+1], label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(omega_1)), omega_1)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(omega_1[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i+1])
end
Plots.hline!(plt, [sqrt(Ω_true[1])], color=:black, linestyle=:dash, ylim=(0, Plots.ylims(plt)[2]), linewidth=1.4, label=nothing)
omega1 = plt

plt = Plots.plot(ylabel="Omega 2")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    omega_2 = [group.omega_2 for group in groupby(df, [:replicate, :fold])]
    for j in 1:length(omega_2)
        x = 0:25:Integer((length(omega_2[j])-1)*25)
        # Plots.plot!(plt, x, omega_2[j], alpha=0.2, label=nothing, color=Plots.palette(colorscheme)[i+1])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [omega_2[j][end]], markershape=:x, markerstrokewidth=3, color=Plots.palette(colorscheme)[i+1], label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(omega_2)), omega_2)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(omega_2[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i+1])
end
Plots.hline!(plt, [sqrt(Ω_true[4])], color=:black, linestyle=:dash, ylim=(0, Plots.ylims(plt)[2]), linewidth=1.4, label=nothing)
omega2 = plt

plt = Plots.plot(ylabel="Correlation")
for (i, model_type) in enumerate(models)
    df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/result_$(model_type == "FOCE2-slow" ? "FOCE2-slow_bfgs" : model_type )_inn.csv"))
    rho = [group.rho for group in groupby(df, [:replicate, :fold])]
    for j in 1:length(rho)
        x = 0:25:Integer((length(rho[j])-1)*25)
        # Plots.plot!(plt, x, rho[j], alpha=0.3, label=nothing, color=Plots.palette(colorscheme)[i+1])
        if x[end] !== 2000
            Plots.scatter!(plt, [x[end]], [rho[j][end]], markershape=:x, markerstrokewidth=3, color=Plots.palette(colorscheme)[i+1], label=nothing)
        end
    end
    res = hcat(filter(x -> length(x) == maximum(length.(rho)), rho)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(rho[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i+1])
end
Plots.hline!(plt, [0.45], color=:black, linestyle=:dash, linewidth=1.4, label=nothing)
rho = Plots.plot(plt, ylim=(-1, 1))

supp_figure_3 = Plots.plot(kl_omega, omega1, omega2, rho, layout=(1, 4), size = (950, 200), leftmargin=5mm, bottommargin=5mm, xlabel="Epoch")
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s3.svg")

################################################################################
##########                                                            ##########
##########                    Supplementary figure 4                   ##########
##########                                                            ##########
################################################################################
# Learned functions on synthetic data sets
smooth_relu(x::Real; β=10.f0) = 1 / β * softplus(β * x)
leaky_softplus(x::Real; α=0.05, β=10.f0) = α * x + (1 - α) * smooth_relu(x; β)

true_wt_cl_eff(x) = (x / 70)^0.75
true_wt_v1_eff(x) = x / 70
true_vwf_eff(x) = 1 / 55 * leaky_softplus(-x + 100; α=0.05, β=0.1) + 0.9

models = ["FO", "FOCE2-slow/bfgs", "VI-eta", "mse"] # "variational-partial"]
epoch_idxs = Integer.([0, 250, 500, 1000, 2000] ./ 25 .+ 1) # epoch / save_every + 1

epoch_2000_wt_cl = typeof(Plots.plot())[]
epoch_2000_wt_v1 = typeof(Plots.plot())[]
epoch_2000_vwf_cl = typeof(Plots.plot())[]
for (k, type) in enumerate(models)
    # Gather data:
    dummy = vcat(collect(range(0, 100.f0, 40))', collect(range(0, 350.f0, 40))')
    result_file = "neural-mixed-effects-paper/new_approach/data/learned_functions_$(replace(type, "/" => "_")).bson"
    if !isfile(result_file)
        folder = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/inn"
        files = readdir(folder)
        filter!(file -> parse(Int64, split(file[findfirst(r"replicate_\d+", file)], "_")[2]) <= 5, files)
        res = zeros(length(epoch_idxs), length(files), 3, size(dummy, 2))
        for (i, epoch_idx) in enumerate(epoch_idxs)
            for (j, file) in enumerate(files)
                ckpt = BSON.load(joinpath(folder, file));
                if length(ckpt[:saved_parameters]) < epoch_idx
                    res[i, j, :, :] .= -1
                    continue # this removes learned functions for models that have
                    # errored before this epoch_idx
                end
                weights = ckpt[:saved_parameters][epoch_idx].weights

                preds_wt, _ = ckpt[:model][1].layers[1](dummy, weights[1][1], ckpt[:state][1][1])
                norm_wt, _ = ckpt[:model][1].layers[1]([70.f0;;], weights[1][1], ckpt[:state][1][1])
                res[i, j, 1:2, :] = preds_wt ./ norm_wt

                preds_vwf, _ = ckpt[:model][1].layers[2](dummy, weights[1][2], ckpt[:state][1][2])
                norm_vwf, _ = ckpt[:model][1].layers[2]([0.f0; 100.f0;;], weights[1][2], ckpt[:state][1][2])
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
    #     plt = Plots.plot(1:100, true_wt_cl_eff.(1:100), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "Weight on\nclearance" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[1, :], res[i, j, 1, :], color=Plots.palette(colorscheme)[k], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     # # Plots.plot!(plt, dummy[1, :], median(res[i, :, 1, :], dims=1)[1, :], linewidth=3, color=Plots.palette(colorscheme)[k], label=nothing)
    #     push!(plts, plt)
    # end
    # plta = Plots.plot(plts..., layout=(1, 5), ylim=(0, 1.6), title=["Before training" "Epoch 250" "Epoch 500" "Epoch 1000" "Epoch 2000"]);
    
    # The filter removes iterations that failed before the end of training.
    ci = hcat([quantile(filter(>(0), res[end, :, 1, j]), [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(filter(>(0), res[end, :, 1, j])) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[1, :], med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, color=Plots.palette(colorscheme)[k])
    Plots.plot!(plt, 1:100, true_wt_cl_eff.(1:100), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "Weight on\nclearance" : "↓", label = nothing) # epoch = 1
    push!(epoch_2000_wt_cl, plt)
    
    # wt on v1
    # plts = typeof(Plots.plot())[]
    # for i in eachindex(epoch_idxs)
    #     plt = Plots.plot(1:100, true_wt_v1_eff.(1:100), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "Weight on\nvolume of distribution" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[1, :], res[i, j, 2, :], color=Plots.palette(colorscheme)[k], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     # Plots.plot!(plt, dummy[1, :], median(res[i, :, 2, :], dims=1)[1, :], linewidth=3, color=Plots.palette(colorscheme)[k], label=nothing)
    #     push!(plts, plt)
    #     if i == 5
    #         push!(epoch_2000_wt_v1, plt)
    #     end
    # end
    # pltb = Plots.plot(plts..., layout=(1, 5), ylim=(0, 1.6));

    ci = hcat([quantile(filter(>(0), res[end, :, 2, j]), [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(filter(>(0), res[end, :, 2, j])) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[1, :], med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, color=Plots.palette(colorscheme)[k])
    Plots.plot!(plt, 1:100, true_wt_v1_eff.(1:100), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "Weight on\nclearance" : "↓", label = nothing) # epoch = 1
    push!(epoch_2000_wt_v1, plt)
    
    # vwf on cl
    # plts = typeof(Plots.plot())[]
    # for i in eachindex(epoch_idxs)
    #     plt = Plots.plot(1:350, true_vwf_eff.(1:350), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "VWF on\nclearance" : "↓", label = nothing) # epoch = 1
    #     [Plots.plot!(plt, dummy[2, :], res[i, j, 3, :], color=Plots.palette(colorscheme)[k], alpha=0.3, label=nothing) for j in 1:size(res, 2)]
    #     # Plots.plot!(plt, dummy[2, :], median(res[i, :, 3, :], dims=1)[1, :], linewidth=3, color=Plots.palette(colorscheme)[k], label=nothing)
    #     push!(plts, plt)
    #     if i == 5
    #         push!(epoch_2000_vwf_cl, plt)
    #     end
    # end
    # pltc = Plots.plot(plts..., layout=(1, 5), ylim=(0, 4));

    ci = hcat([quantile(filter(>(0), res[end, :, 3, j]), [0.025, 0.975]) for j in 1:size(dummy, 2)]...)
    med = [median(filter(>(0), res[end, :, 3, j])) for j in 1:size(dummy, 2)]
    plt = Plots.plot(dummy[2, :], med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, color=Plots.palette(colorscheme)[k])
    Plots.plot!(plt, 1:350, true_vwf_eff.(1:350), linewidth=2, color=:black, linestyle=:dash, ylabel=i == 1 ? "Weight on\nclearance" : "↓", label = nothing) # epoch = 1
    push!(epoch_2000_vwf_cl, plt)

    # plt_ = Plots.plot(plta, pltb, pltc, layout=(3, 1), size=(950, 475), xticks=false, yticks=false, leftmargin=8mm)
    # Plots.savefig("neural-mixed-effects-paper/new_approach/plots/learned_functions_$(replace(type, "/" => "_")).svg")

    # display(plt_)
end

top = Plots.plot(epoch_2000_wt_cl..., layout=(1, 4), ylim = (0, 1.5), size=(800, 200), xlabel="Weight (kg)", ylabel=["Fold change\nin clearance" "" "" ""])
middle = Plots.plot(epoch_2000_wt_v1..., layout=(1, 4), ylim = (0, 1.5), size=(800, 200), xlabel="Weight (kg)", ylabel=["Fold change in\nvolume of distribution" "" "" ""])
bottom = Plots.plot(epoch_2000_vwf_cl..., layout=(1, 4), ylim = (0, 5), size=(800, 200), xlabel="Von Willebrand factor (%)", ylabel=["Fold change\nin clearance" "" "" ""])

Plots.plot(top, middle, bottom, layout = (3, 1), size=(1050, 700), leftmargin=4mm, bottommargin=3mm, legend=false)
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s4.svg")

################################################################################
##########                                                            ##########
##########                  Supplementary Figure 5                    ##########
##########                                                            ##########
################################################################################
# Parameters over training on the real world data sets.

# models = ["FO", "VI-eta"]
models = ["FO", "FOCE2", "VI-eta"]
df_idx = 1
vi_length = df_idx == 1 ? 1000 : 1250

plta = Plots.plot(ylabel="Obj. function\nvalue", title = "FO")
pltb = Plots.plot(title = "FOCE (Eq. 4b)")
pltc = Plots.plot(title = "VI (three samples)")
for (i, model_type) in enumerate(models)
    file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(model_type)_inn_prop.csv"
    df = DataFrame(CSV.File(file))
    LL = [group.LL for group in groupby(df, [:outer, :inner])]
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    res = hcat(filter(ll -> length(ll) == maximum(length.(LL)), LL)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(LL[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
Plots.plot!(plta, ylim=(df_idx == 1 ? -900 : -1600, 0), ylabel="Neg. loglikelihood")
Plots.plot!(pltb, ylim=(df_idx == 1 ? -800 : -1350, 0), xlim=(-50, 2000), ylabel="Neg. loglikelihood")
Plots.plot!(pltc, ylim=(df_idx == 1 ? -280 : -250, 0), ylabel="Neg. ELBO")
obj = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (400, 800), xlim=[(-50, 1000) (-50, 2000) (-50, vi_length)], leftmargin=5mm, bottommargin=5mm)

# Omega 1
plta = Plots.plot(ylabel="Omega 1", title = "FO")
pltb = Plots.plot(title = "FOCE (Eq. 4b)")
pltc = Plots.plot(title = "VI (three samples)")
for (i, model_type) in enumerate(models)
    file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(model_type)_inn_prop.csv"
    df = DataFrame(CSV.File(file))
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    omega_1 = [group.omega_1 for group in groupby(df, [:outer, :inner])]
    res = hcat(filter(omega -> length(omega) == maximum(length.(omega_1)), omega_1)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(omega_1[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
omega_1 = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=[(-50, 1000) (-50, 2000) (-50, vi_length)], ylim=(0, 0.6))


# Omega 2
plta = Plots.plot(ylabel="Omega 2", title = "FO")
pltb = Plots.plot(title = "FOCE (Eq. 4b)")
pltc = Plots.plot(title = "VI (three samples)")
for (i, model_type) in enumerate(models)
    file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(model_type)_inn_prop.csv"
    df = DataFrame(CSV.File(file))
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    omega_2 = [group.omega_2 for group in groupby(df, [:outer, :inner])]
    res = hcat(filter(omega -> length(omega) == maximum(length.(omega_2)), omega_2)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(omega_2[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
omega_2 = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=[(-50, 1000) (-50, 2000) (-50, vi_length)], ylim=(0, 0.6))


plta = Plots.plot(ylabel="Additive error", title = "FO")
pltb = Plots.plot(title = "FOCE (Eq. 4b)")
pltc = Plots.plot(title = "VI (three samples)")
for (i, model_type) in enumerate(models)
    file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(model_type)_inn_prop.csv"
    df = DataFrame(CSV.File(file))
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    sigma_1 = [group.sigma_1 for group in groupby(df, [:outer, :inner])]
    res = hcat(filter(sigma -> length(sigma) == maximum(length.(sigma_1)), sigma_1)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(sigma_1[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
sigma_1 = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=[(-50, 1000) (-50, 2000) (-50, vi_length)], ylim=(0, 0.26))

plta = Plots.plot(ylabel="Proportional error", title = "FO")
pltb = Plots.plot(title = "FOCE (Eq. 4b)")
pltc = Plots.plot(title = "VI (three samples)")
for (i, model_type) in enumerate(models)
    file = "neural-mixed-effects-paper/new_approach/data/result_df$(df_idx)_real_world_$(model_type)_inn_prop.csv"
    df = DataFrame(CSV.File(file))
    if model_type == "FO"
        plt = plta
    elseif startswith(model_type, "FOCE")
        plt = pltb
    else
        plt = pltc
    end
    sigma_2 = [group.sigma_2 for group in groupby(df, [:outer, :inner])]
    res = hcat(filter(sigma -> length(sigma) == maximum(length.(sigma_2)), sigma_2)...)
    ci = hcat([quantile(res[k, :], [0.025, 0.975]) for k in 1:size(res)[1]]...)
    med = median(res, dims=2)[:, 1]
    Plots.plot!(plt, 0:25:Integer(25*(length(sigma_2[1])-1)), med, ribbon=(med - ci[1, :], ci[2, :] - med), label=nothing, linewidth=3, color=Plots.palette(colorscheme)[i])
end
sigma_2 = Plots.plot(plta, pltb, pltc, xlabel="Epoch", layout=(3, 1), size = (300, 800), leftmargin=5mm, bottommargin=5mm, xlim=[(-50, 1000) (-50, 2000) (-50, vi_length)], ylim=(0, 0.5))

figure_s5 = Plots.plot(obj, omega_1, omega_2, sigma_1, sigma_2, layout=(1, 5), size=(1200, 600), title="", xlabel=["" "" "Epoch"])
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s5_df$(df_idx).svg")


################################################################################
##########                                                            ##########
##########                  Supplementary Figure 7                    ##########
##########                                                            ##########
################################################################################
# Parameters over training on the real world data sets.

smooth_relu(x::Real; β=10.f0) = 1 / β * softplus(β * x)
leaky_softplus(x::Real; α=0.05, β=10.f0) = α * x + (1 - α) * smooth_relu(x; β)

df_idx = 2

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

models = ["vi", "vi-slow"] # "variational-partial"]
epoch_idxs = Integer.([0, 250, 500, 1000, 1500] ./ 25 .+ 1) # epoch / save_every + 1

epoch_1000_ffm_cl = typeof(Plots.plot())[]
epoch_1000_ffm_v1 = typeof(Plots.plot())[]
epoch_1000_vwf_cl = typeof(Plots.plot())[]
color_idxs = (mse = 4, fo = 1, foce = 2, vi = 3, vi_slow = 3)

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
    plt = Plots.plot(dummy[2, :] .* 350, med, ribbon = (med - ci[1, :], ci[2, :] - med), linewidth=3, ylim=(0, 3.7), xlabel="VWF:Ag (%)", color=Plots.palette(colorscheme)[color_idx])
    Plots.histogram!(Plots.twinx(), population.x[2, :] .* 100, linecolor=:lightgrey, color=:lightgrey, legend=false, xaxis=false, yaxis=false, ylim=(0, 35), bins=20)
    push!(epoch_1000_vwf_cl, plt)

    # plt_ = Plots.plot(plta, pltb, pltc, layout=(3, 1), size=(950, 475), xticks=false, yticks=false, leftmargin=8mm)
    # Plots.savefig("neural-mixed-effects-paper/new_approach/plots/learned_functions_$(replace(type, "/" => "_")).svg")

    # display(plt_)
end

top = Plots.plot(epoch_1000_ffm_cl..., layout=(1, 2), linewidth=3, size=(1050, 200), xlim=(0, 100), ylabel=["Fold change\nin clearance" "" "" ""], legend=false)
middle = Plots.plot(epoch_1000_ffm_v1..., layout=(1, 2), linewidth=3, size=(1050, 200), xlim=(0, 100), ylabel=["Fold change in\nvolume of distribution" "" "" ""], legend=false)
bottom = Plots.plot(epoch_1000_vwf_cl..., layout=(1, 2), linewidth=3, size=(1050, 200), xlim=(0, 350), ylabel=["Fold change\nin clearance" "" "" ""], legend=false)

Plots.plot(top, middle, bottom, layout = (3, 1), size=(500, 600), leftmargin=4mm, bottommargin=3mm, legend=false, framestyle=:box)
Plots.savefig("neural-mixed-effects-paper/new_approach/plots/figure_s7_df$(df_idx).svg")