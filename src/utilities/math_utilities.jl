using LinearAlgebra
using Base

"""
a struct that represents a confusion matrix with all the necessary fields.
Fields:
    true_positives::Array{Int, 1}: An array representing the true positives
    true_negatives::Array{Int, 1}: An array representing the true negatives
    false_positives::Array{Int, 1}: An array representing the false positives
    false_negatives::Array{Int, 1}: An array representing the false negatives
    matrix::Array{Int, 2}:: The confusion matrix
    Labels::Array{Union{Int,AbstractString},1}:: confusion matrix labels either numerical or string

"""
struct confusion_matrix
    true_positives::Array{Int, 1}
    true_negatives::Array{Int, 1}
    false_positives::Array{Int, 1}
    false_negatives::Array{Int, 1}
    matrix::Array{Int, 2}
    Labels::Array{Union{Int,AbstractString},1}
end


condition_positive(c::confusion_matrix,i) = c.true_positives[i] + c.false_negatives[i]
condition_negative(c::confusion_matrix,i) = c.true_negatives[i] + c.false_positives[i]
predicted_positive(c::confusion_matrix,i) = c.true_positives[i] + c.false_positives[i]
predicted_negative(c::confusion_matrix,i) = c.true_negatives[i] + c.false_negatives[i]
correctly_classified(c::confusion_matrix,i) = c.true_positives[i] + c.true_negatives[i]
incorrectly_classified(c::confusion_matrix,i) = c.false_positives[i] + c.false_negatives[i]
sensitivity(c::confusion_matrix,i) = c.true_positives[i] / condition_positive(c,i)
specificity(c::confusion_matrix,i) = c.true_negatives[i] / condition_negative(c,i)
precision(c::confusion_matrix,i) = c.true_positives[i] / (c.true_positives[i] + c.false_positives[i])
accuracy(c::confusion_matrix,i) = (c.true_positives[i] + c.true_negatives[i] ) / (condition_positive(c,i) + condition_negative(c,i))
balanced_accuracy(c::confusion_matrix,i) = (sensitivity(c,i) +  specificity(c,i)) / 2
negative_predictive_value(c::confusion_matrix,i)  = c.true_negatives[i] / (c.true_negatives[i] + c.false_negatives[i])
false_negative_rate(c::confusion_matrix,i) = c.false_negatives[i] / condition_positive(c,i)
false_positive_rate(c::confusion_matrix,i) = c.false_positives[i] / condition_negative(c,i)
false_discovery_rate(c::confusion_matrix,i) = c.false_positives[i] / ( c.false_positives[i]  + c.true_negatives[i])
false_omission_rate(c::confusion_matrix,i) = 1 - negative_predictive_value(c,i)
f1_score(c::confusion_matrix,i) = (2* c.true_positives[i] ) / (2* c.true_positives[i] + c.false_positives[i] + c.false_negatives[i])




function Base.show(io::IO, ::MIME"text/plain", c::confusion_matrix)
    println(io, summary(c), ":")
    tp = string(c.true_positives)
    fp = string(c.false_positives)
    fn = string(c.false_negatives)
    tn = string(c.true_negatives)
    len_p = max(maximum(map(length, (tp, fp))), 3)
    len_n = max(maximum(map(length, (fn, tn))), 3)
    tp = lpad(tp, len_p); fp = lpad(fp, len_p)
    fn = lpad(fn, len_n); tn = lpad(tn, len_n)
    pad = "  "
    println(io, pad, " ", "Predicted")
    println(io, pad, "  ", lpad("+",len_p), "   ", lpad("-",len_n))
    println(io, pad, "┌", repeat("─",len_p+2), "┬", repeat("─",len_n+2), "┐")
    println(io, pad, "│ ", tp, " │ ", fn, " │ +")
    println(io, pad, "├", repeat("─",len_p+2), "┼", repeat("─",len_n+2), "┤   Actual")
    println(io, pad, "│ ", fp, " │ ", tn, " │ -")
    println(io, pad, "└", repeat("─",len_p+2), "┴", repeat("─",len_n+2), "┘")
end

"""
function init_confusion_params(matrix)
Arguments:
    matrix: an n x n confusion matrix

calculates the necessary fields true_positives, true_negatives, false_positives, false_negatives and returns them as a tuple containing the results
in the following order: true_positives, true_negatives, false_positives, false_negatives
"""
function init_confusion_params(matrix)
    tp = []; tn = []; fp = []; fn = []
    for i in 1:size(matrix)[1]-1
        push!(tp, matrix[i,i])
        push!(fn, tp[i] - sum(matrix[i,:]))
        push!(fp, tp[i] - sum(matrix[:,i]))
        push!(tn, (sum(matrix) - tp[i] - fn[i] - fp[i]))
    end
    return tp, tn, fp, fn
end

"""
function create_confusion_matrix(expected, predicted; labels = nothing, normalize = false)

    This function creates a confusion matrix and returns it
    For more information @confusion_matrix

Arguments:
    expected - an n x 1 array containing the true classification results
    predicted- an n x 1 array containg the predicted classification results
    labels- if the input is passed as strings then labels must be provided, for integers it is optional
    normalize- if normalize is true, matrix normalization will be applied to the confusion matrix



"""

function create_confusion_matrix(expected, predicted; labels = nothing, normalize = false)
    @assert size(expected) == size(predicted) "Expected and predicted vectors must be the same size"
    @assert sort(unique(expected)) == sort(unique(predicted))  "Expected and predicted vectors must have the same labels"
    if labels == nothing
        labels = sort(unique(expected))
    end
    @assert length(unique(typeof.(labels))) == 1 "Labels must be of the same data types"
    matrix = zeros(Int, size(labels)[1],size(labels)[1])
    T_values = copy(expected)
    P_values = copy(predicted)
    for i in 1:size(T_values)[1]
        T_values[i] = findfirst(x-> x == T_values[i],labels)
        P_values[i] = findfirst(x-> x == P_values[i],labels)
    end
    for i in 1:size(T_values)[1]
        matrix[P_values[i],T_values[i]] += 1
    end
    if normalize
        normalize!(matrix)
    end
    tp, tn, fp, fn = init_confusion_params(matrix)
    return confusion_matrix(tp,tn,fp,fn,matrix,labels)
end

"""
create_confusion_matrix(matrix; normalize = false)

returns a confusion matrix object

Arguments:
    matrix- confusion matrix
    normalize- if true, matrix normalization will be applied

For more information @confusion_matrix
"""

function create_confusion_matrix(matrix; normalize = false)
    @assert size(matrix)[1] == size(matrix)[2]
    if normalize
        normalize!(matrix)
    end
    tp, tn, fp, fn = init_confusion_params(matrix)
    return confusion_matrix(tp,tn,fp,fn,matrix,labels)
end

function PCA(matrix)
    sigma = (1/ size(matrix,1) * transpose(matrix) * matrix)
    return svd(sigma)
end
