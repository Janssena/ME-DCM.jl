import Optimisers
import Random
import Plots
import BSON
import Flux
import CSV
import Lux

include("neural-mixed-effects-paper/new_approach/model.jl");
include("src/lib/constraints.jl");

using DataFrames

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
non_zero_relu(x::T) where {T<:Real} = Lux.relu(x) + T(1e-3)

type = "variational-partial" # [FO, FOCE1, FOCE2, variational, variational-partial] 
folder = "neural-mixed-effects-paper/new_approach/checkpoints/$(type)/inn"
files = readdir(folder)
filter!(file -> contains(file, "fold_1_"), files)

xs = collect(hcat(range(0, 150, 100), range(0, 350, 100))')

effect_weight = zeros(length(files), 2, 100)
effect_vwf = zeros(length(files), 100)

for (i, file) in enumerate(files)
    ckpt = BSON.parse(joinpath(folder, file))
    delete!(ckpt, :prob)
    ckpt = BSON.raise_recursive(ckpt, Main)

    if startswith(type, "variational")
        opt_params = ckpt[:saved_parameters][end]
    else
        opt_params = :opt_p in keys(ckpt) ? ckpt[:opt_p] : ckpt[:parameters][ckpt[:best_val_epoch]]
    end

    effs, _ = ckpt[:model][1](xs, opt_params.weights[1], ckpt[:state][1])
    typ_eff, _ = ckpt[:model][1]([70.f0; 100.f0;;], opt_params.weights[1], ckpt[:state][1])

    effect_weight[i, :, :] = effs[1] ./ typ_eff[1]
    effect_vwf[i, :] = vwf_eff = effs[2] ./ typ_eff[2]
end

plt1 = Plots.plot()
plt2 = Plots.plot()
plt3 = Plots.plot()
[Plots.plot!(plt1, xs[1, :], effect_weight[i, 1, :], color=:black, label=nothing) for i in 1:length(files)]
[Plots.plot!(plt2, xs[1, :], effect_weight[i, 2, :], color=:black, label=nothing) for i in 1:length(files)]
[Plots.plot!(plt3, xs[2, :], effect_vwf[i, :], color=:black, label=nothing) for i in 1:length(files)]

Plots.plot(plt1, plt2, plt3, layout = (1, 3), size=(1200, 300))

# FO method seems very stable. Learned functions are likely similar to deep ensemble
# FOCE1 method also seems to result in nice predicitons in function space.
# FOCE2 method also seems to result in nice predicitons in function space.

# Variational method seems quite unstable. This might be from bad typical predictions?