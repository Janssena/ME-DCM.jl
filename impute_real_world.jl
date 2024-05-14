import Flux.NNlib
import Bijectors
import BSON
import Flux
import CSV

include("../DeepFVIII.jl/src/generative/neural_spline_flows.jl");

using DataFrames
using Distributions

ckpt_age_ht = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/generative_model/age_to_ht_manuscript.bson")
ckpt_ht_wt = BSON.load("neural-mixed-effects-paper/new_approach/checkpoints/generative_model/ht_to_wt_manuscript.bson")

function get_median_wt(ht)
    model = ckpt_ht_wt[:re](ckpt_ht_wt[:w])
    out_ = model([ht])
    b₁ = NeuralSpline(out_; order=Quadratic(), B=10.f0)
    b₂ = inverse(Bijectors.bijector(LogNormal()))
    Y = Bijectors.transformed(ckpt_ht_wt[:dist], b₂ ∘ b₁)
    return median(rand(Y, 1_000_000))
end

f(x::AbstractVector, θ) = exp(θ[1] + max((x[1] / 45), 40 / 45) * θ[2]) * (0.7008693880000001 ^ x[2])
f(x::AbstractMatrix, θ) = exp.(θ[1] .+ max.((x[:, 1] ./ 45.), 40 / 45) .* θ[2]) .* (0.7008693880000001 .^ x[:, 2])

function get_mode_vwf(age, bgo)
    x = mode(LogNormal(0.1578878428424249, 0.3478189783243864))
    return x * f([age, bgo], [4.11, 0.644 * 0.8])
end

function calc_ffm(age, wt, ht)
    bmi = wt / (ht / 100.).^2
    return (0.88 + ((1-0.88) / (1 + (age / 13.4)^-12.7))) * ((9270 * wt) / (6680 + (216 * bmi)))
end

"""
Impute missing data with the mode of the prior distributions in the generative 
model.
"""

################################################################################
##########                                                            ##########
##########              Data set 1: Prophylactic data                 ##########
##########                                                            ##########
################################################################################

df = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/complete_opticlot_data.csv"))
filter!(row -> row.SubjectID != "RAD-C-001", df)
# There is something strange with this PK profile
df_pk = df[df.OCC .== 1, :] # get prophylactic data
df_pk[df_pk.MDV .== 1, :Rate] = df_pk[df_pk.MDV .== 1, :Dose] * 60.
df_pk[df_pk.MDV .== 1, :Duration] .= 1 / 60.
# impute missing product:
df_pk[df_pk.SubjectID .== "LUMC-A-003", :FVIII_product] .= "6= NovoEight"
delete!(df_pk, (1:nrow(df_pk))[(df_pk.SubjectID .== "RAD-C-001") .& (df_pk.Time .< 8000)])
df_group = groupby(df_pk, :SubjectID)
df_pk[!, :FFM] .= 0.
df_pk[!, :VWF_impute] .= 0.
df_pk[!, :VWF_final] .= 0.

# Impute missing values
for group in df_group
    age = group.Age[1]
    bgo = startswith(group.Blood_group[1], "4") + 0
    if all(ismissing.(group.Height))
        # impute height based on age
        model_μ = ckpt_age_ht[:re_mean](ckpt_age_ht[:w_mean])
        group.Height .= first(model_μ([age])) # here μ is the mode
    end

    ht = group.Height[1]

    if all(ismissing.(group.Weight))
        # impute weight based on height
        group.Weight .= get_median_wt(ht)
    end

    wt = group.Weight[1]
    vwf_imputation = get_mode_vwf(age, bgo) / 100
    group.VWF_impute .= vwf_imputation
    if all(ismissing.(group.VWFAg))
        group.VWF_final .= vwf_imputation
    else
        non_missing_vwf = first(group[.!ismissing.(group.VWFAg), :VWFAg])
        group.VWF_final .= group.HistoricVWF[1] == 1 ? mean([non_missing_vwf, vwf_imputation]) : non_missing_vwf

    end
    group.FFM .= calc_ffm(age, wt, ht)
end

df_pk_subset = df_pk[:, [:SubjectID, :OCC, :Time, :Dose, :Rate, :Duration, :DV, :MDV, :Blood_group, :Hemophilia_severity, :Baseline, :Age, :Weight, :Height, :FFM, :FVIII_product, :VWFAg, :HistoricVWF, :VWF_impute, :VWF_final]]
# save imputed dataset
CSV.write("neural-mixed-effects-paper/new_approach/data/df1_prophylaxis_imputed.csv", df_pk_subset)

################################################################################
##########                                                            ##########
##########          Data set 2: Prospective Surgical data             ##########
##########                                                            ##########
################################################################################

df_surg = df[df.OCC .== 2, :]
allowmissing!(df_surg, :Dose)
df_surg[.!ismissing.(df_surg.Dose) .& (df_surg.Dose .== 0), [:Dose, :Rate, :Duration]] .= missing
df_surg[(df_surg.MDV .== 1) .& ismissing.(df_surg.Dose), :MDV] .= 2 # these were empty rows, unsure what we added here.

bolus = (df_surg.MDV .== 1) .& ((.!ismissing.(df_surg.Rate)) .& (df_surg.Rate .== 0.))
df_surg[bolus, :Rate] = df_surg[bolus, :Dose] * 60.
df_surg[bolus, :Duration] .= 1 / 60.

df_surg[df_surg.SubjectID .== "LUMC-A-003", :FVIII_product] .= "6= NovoEight"
# No missings for Height or Weight
df_surg[ismissing.(df_surg.FVIII_product), :]

df_surg.FFM .= 0
df_surg.VWF_impute .= 0
df_surg.VWF_final .= 0
# take the mean VWF per group with vwf measurements:
df_surg_group = groupby(df_surg, :SubjectID)
for group in df_surg_group
    non_missing_vwf_idxs = .!ismissing.(group.VWFAg)
    if any(non_missing_vwf_idxs)
        group.VWF_final .= mean(group[non_missing_vwf_idxs, :VWFAg])
    end
    group.FFM .= calc_ffm(group[1, :Age], group[1, :Weight], group[1, :Height])
end

surg_vwf = DataFrame([group[1, [:SubjectID, :VWF_final]] for group in df_surg_group])
rename!(surg_vwf, :VWF_final => :VWF_surg)
df_vwfs = unique(leftjoin(surg_vwf, df_pk_subset[:, [:SubjectID, :VWF_final]], on=:SubjectID))
rename!(df_vwfs, :VWF_final => :VWF_pk)
ratios = df_vwfs.VWF_surg ./ df_vwfs.VWF_pk
filter!(>(0), ratios) # Remove the ones with missing surg vwf (i.e. = 0)
ratio = mean(ratios) # surgical VWF levels are on average 30% higher

last_idxs = df_vwfs[df_vwfs.VWF_surg .== 0, :] # These are idxs with missing VWF in the surgical data set.
for row in eachrow(last_idxs)
    df_surg[df_surg.SubjectID .== row.SubjectID, :VWF_final] .= row.VWF_pk * ratio
end

df_surg_subset = df_surg[:, [:SubjectID, :OCC, :Time, :Dose, :Rate, :Duration, :DV, :MDV, :Expected_duration_surgery, :Blood_group, :Hemophilia_severity, :Baseline, :Surgical_risk_score, :Expected_blood_loss, :Age, :Height, :Weight, :FFM, :VWFAg, :VWF_impute, :VWF_final]]

CSV.write("neural-mixed-effects-paper/new_approach/data/df2_surgery_imputed.csv", df_surg_subset)

################################################################################
##########                                                            ##########
##########          Data set 3: Retrospective Surgical data           ##########
##########                                                            ##########
################################################################################

df_retro = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/FVIII_01040_orignal.csv"))

df_retro_subset = df_retro[:, [Symbol("NEW ID"), :OCC, :CENTER, :TIME4, :DOSE, :RATE, :DV, Symbol("NEW MDV"), Symbol("OK-TIME"), :BASELINE, :PROD, :ZWOK, :INHIBITOR, :SEVERITY, :BG, :AGE, :WT, :LNGT, Symbol("VWF AG")]]
rename!(df_retro_subset, [:ID, :OCC, :CENTER, :Time, :Dose, :Rate, :DV, :MDV, :OK_Time, :Baseline, :FVIII_product, :Surgical_risk_score, :Inhibitor, :Hemophilia_severity, :Blood_group, :Age, :Weight, :Height, :VWFAg])
df_retro_subset[!, :Duration] .= 0.
df_retro_subset[(df_retro_subset.Dose .== 0) .& (df_retro_subset.MDV .== 1), :MDV] .= 2
infusions = (df_retro_subset.MDV .== 1) .& (df_retro_subset.Rate .!= 0) 
df_retro_subset[infusions, :Duration] = df_retro_subset[infusions, :Dose] ./ df_retro_subset[infusions, :Rate]
bolus = (df_retro_subset.MDV .== 1) .& (df_retro_subset.Rate .== 0) 
df_retro_subset[bolus, :Duration] .= 1 / 60
df_retro_subset[bolus, :Rate] = df_retro_subset[bolus, :Dose] * 60

allowmissing!(df_retro_subset, [:Dose, :Rate, :Duration, :DV])
df_retro_subset[df_retro_subset.Dose .== 0, :Duration] .= missing
df_retro_subset[df_retro_subset.Dose .== 0, :Rate] .= missing
df_retro_subset[df_retro_subset.Dose .== 0, :Dose] .= missing
df_retro_subset[df_retro_subset.MDV .== 1, :DV] .= missing

for column in [:FVIII_product, :Surgical_risk_score, :Inhibitor, :Hemophilia_severity, :Blood_group, :Age, :Weight, :Height, :VWFAg]
    fake_missing = df_retro_subset[:, column] .== 999
    if any(fake_missing)
        allowmissing!(df_retro_subset, column)
        df_retro_subset[fake_missing, column] .= missing # Make these actually missing
    end
end


df_retro_subset[!, :FFM] .= 0
df_retro_subset[!, :VWF_impute] .= 0
df_retro_subset[!, :VWF_final] .= 0
for group in groupby(df_retro_subset, :ID)
    age = group.Age[1]
    bgo = (!ismissing(group.Blood_group[1]) && group.Blood_group[1] > 6) + 0
    if all(ismissing.(group.Height))
        # impute height based on age
        model_μ = ckpt_age_ht[:re_mean](ckpt_age_ht[:w_mean])
        group.Height .= first(model_μ([age])) # here μ is the mode
    else any(ismissing.(group.Height))
        group[ismissing.(group.Height), :Height] .= first(group[.!ismissing.(group.Height), :Height])
    end

    ht = group.Height[1]
    wt = group.Weight[1]
    vwf_imputation = get_mode_vwf(age, bgo) / 100
    group.VWF_impute .= vwf_imputation
    if all(ismissing.(group.VWFAg))
        group.VWF_final .= vwf_imputation
    else
        non_missing_vwf = first(group[.!ismissing.(group.VWFAg), :VWFAg])
        group.VWF_final .= mean([non_missing_vwf, vwf_imputation]) # We assume that all VWF measurements are historic here.
    end
    group.FFM .= calc_ffm(age, wt, ht)
end

CSV.write("neural-mixed-effects-paper/new_approach/data/df3_surgery_imputed.csv", df_retro_subset)