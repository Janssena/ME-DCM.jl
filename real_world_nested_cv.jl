import Random
import BSON
import CSV

using DataFrames

function create_cv(n, k)
    idxs = collect(1:n)
    n = length(idxs)
    shuffle!(idxs)
    train = Vector{Vector{Int64}}(undef, k)
    test = Vector{Vector{Int64}}(undef, k)
    fold_size = Integer(round(n / k))
    for i in 1:k
        test[i] = (1 + (i - 1) * fold_size):(i == k ? n : i * fold_size)
        tmp = collect(1:n)
        deleteat!(tmp, test[i])
        train[i] = tmp
        @assert length(train[i]) + length(test[i]) == n "The length of the train and test sets for fold $i does not match."
    end

    println("train lengths: $(map(length, train))")
    println("test lengths: $(map(length, test))")

    @assert sum(map(length, test)) == n "The sum of the fold lengths is not equal to the original data set."

    return ixds[train], idxs[test]
end

df1 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df1_prophylaxis_imputed.csv"))
df2 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df2_surgery_imputed.csv"))
df3 = DataFrame(CSV.File("neural-mixed-effects-paper/new_approach/data/df3_surgery_imputed.csv"))

dfs = [df1, df2, df3]

outer_folds = 10
inner_folds = 10

for (i, df) in enumerate(dfs)
    n = length(groupby(df, [:SubjectID, :SubjectID, :ID][i]))
    outer_reshuffle = Vector{Vector{Int64}}(undef, outer_folds)
    for j in 1:outer_folds
        outer_reshuffle[j] = Random.shuffle(collect(1:n))
    end
    train, test = create_cv(n, inner_folds)
    BSON.bson(
        "neural-mixed-effects-paper/new_approach/data/real_world_nested_cv_df$i.bson", 
        Dict(:outer => outer_reshuffle, :inner_train => train, :inner_test => test)
    )
end
    