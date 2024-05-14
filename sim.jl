import DifferentialEquations: ODEProblem, solve, remake, DiscreteCallback
import BSON
import Flux
import CSV

include("src/lib/dataset.jl");
include("src/lib/compartment_models.jl");
include("../DeepFVIII.jl/src/generative/neural_spline_flows.jl");

using Bijectors
using DataFrames
using Distributions

"""
Simulation:

CL = θ₁ ⋅ (wt / 70)^0.75 ⋅ leaky_softplus(vwf) ⋅ exp(η₁)
V1 = θ₂ ⋅ (wt / 70) ⋅ exp(η₂)
Q = θ₃
V2 = θ₄

Ω = [0.037 0.0113 ; 0.0113 0.017] # ρ = 0.45, CV_cl ≈ 30%
σ = [3.] # in IU/dL

dose = 25 IU/kg normalized to closest 250 IU.
simulate age, weight, height, bgo and vwf from generative model.
"""

n = 500
# AGE
age = rand(Uniform(1, 85), n)
# HEIGHT
ckpt = BSON.load("../DeepFVIII.jl/checkpoints/age_to_ht.bson")
mean_fn = ckpt[:re_mean](ckpt[:w_mean])
sigma_fn = ckpt[:re_sigma](ckpt[:w_sigma])
height = round.([rand(Normal(first(mean_fn([a])), first(sigma_fn([a])))) for a in age])
# WEIGHT
ckpt = BSON.load("../DeepFVIII.jl/checkpoints/ht_to_wt.bson")
model = ckpt[:re](ckpt[:w])

weight = zeros(Float32, length(height))
for (i, htᵢ) in enumerate(height)
    out_ = model([htᵢ])
    b₁ = NeuralSpline(out_; order=Quadratic(), B=10.f0)
    b₂ = inverse(bijector(LogNormal()))
    Y = transformed(Normal(), b₂ ∘ b₁)
    weight[i] = rand(Y)
end

bgo = rand(Bernoulli(0.45), n) # p(bg = O | country = nl)

f(x::AbstractVector, θ) = exp(θ[1] + max((x[1] / 45), 40 / 45) * θ[2]) * (0.7008693880000001 ^ x[2])
f(x::AbstractMatrix, θ) = exp.(θ[1] .+ max.((x[:, 1] ./ 45.), 40 / 45) .* θ[2]) .* (0.7008693880000001 .^ x[:, 2])

function sample_vwf(age, bgo)
    X = LogNormal(0.1578878428424249, 0.3478189783243864)
    b = Bijectors.Scale(f([age, bgo], [4.11, 0.644 * 0.8]))
    Y = transformed(X, b)
    return rand(Y)
end

vwfag = sample_vwf.(age, bgo)

smooth_relu(x::Real; β=10.f0) = 1 / β * softplus(β * x)
leaky_softplus(x::Real; α=0.05, β=10.f0) = α * x + (1 - α) * smooth_relu(x; β)

# Good approximation of learned function by neural network.
Plots.scatter(vwfag, 1 / 55 .* leaky_softplus.(-vwfag .+ 100; α=0.05, β=0.1) .+ 0.9)
Plots.hline!([1], color=:lightgrey, linestyle=:dash, linewidth=2, ylim=(0.4, 2.5))

cl = 0.1 .* (weight ./ 70).^0.75 .* (1 / 55 .* leaky_softplus.(-vwfag .+ 100; α=0.05, β=0.1) .+ 0.9)
v1 = 2. .* (weight ./ 70)
q = 0.15
v2 = 0.75

Ω = [0.037 0.45 * sqrt(0.037) * sqrt(0.017) ; 0.45 * sqrt(0.037) * sqrt(0.017) 0.017]
etas = rand(MultivariateNormal(zeros(2), Ω), n)

prob = ODEProblem(two_comp!, zeros(2), (-0.1, 72))
𝐈 = [[0. ceil((wt * 25) / 250) * 250 round((wt * 25) / 250) * 250 * 60 1/60] for wt in weight]
callbacks = generate_dosing_callback.(𝐈; S1 = 1/1000)

t = zeros(3, n)
preds = zeros(3, n)
ipreds = zeros(3, n)

for i in 1:n
    t[:, i] = rand.([truncated(Normal(4., 1.5), 0., Inf), Normal(24, 4), Normal(48, 4.)])
    sol_typical = solve(remake(prob, p = [cl[i], v1[i], q, v2, 0.]), saveat=t[:, i], save_idxs=1, tstops=callbacks[i].condition.times, callback=callbacks[i])
    sol_individual = solve(remake(prob, p = [cl[i] * exp(etas[1, i]), v1[i] * exp(etas[2, i]), q, v2, 0.]), saveat=t[:, i], save_idxs=1, tstops=callbacks[i].condition.times, callback=callbacks[i])
    preds[:, i] = sol_typical.u
    ipreds[:, i] = sol_individual.u
end

Plots.histogram(ipreds[1, :])
Plots.histogram(ipreds[2, :])
Plots.histogram(ipreds[3, :])

y = max.(ipreds + rand(Normal(0., 0.03), size(ipreds)), 0.)

df = DataFrame()
for i in 1:n
    tmp = DataFrame(id = "virtual_$i", time = [0.; t[:, i]], dv = [missing; y[:, i]], mdv = [1, 0, 0, 0], amt = [𝐈[i][2], missing, missing, missing], rate = [𝐈[i][3], missing, missing, missing], duration = [𝐈[i][4], missing, missing, missing], age = age[i], ht = height[i], wt = weight[i], bgo = bgo[i] + 0, vwf = vwfag[i], pred = [missing; preds[:, i]], ipred = [missing; ipreds[:, i]])
    append!(df, tmp)
end

CSV.write("neural-mixed-effects-paper/new_approach/data/simulation.csv", df)


for i in 1:20
    idxs = rand(1:500, 60)
    CSV.write("neural-mixed-effects-paper/new_approach/data/train_set_$(i).csv", DataFrame(idxs = idxs))
end


for i in 1:20
    train = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/train_set_$(i).csv")).idxs
    validation = rand(filter(idx -> !(idx ∈ train), 1:500), 60)
    CSV.write("neural-mixed-effects-paper/new_approach/data/validation_set_$(i).csv", DataFrame(idxs = validation))
end