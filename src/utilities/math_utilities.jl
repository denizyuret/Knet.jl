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
    true_positives::Array{Int}
    true_negatives::Array{Int}
    false_positives::Array{Int}
    false_negatives::Array{Int}
    matrix::Array{Number,2}
    Labels::Array{Union{Int,AbstractString}}
end


condition_positive(c::confusion_matrix,i = 1) = c.true_positives[i] + c.false_negatives[i]
condition_negative(c::confusion_matrix,i = 1) = c.true_negatives[i] + c.false_positives[i]
predicted_positive(c::confusion_matrix,i = 1) = c.true_positives[i] + c.false_positives[i]
predicted_negative(c::confusion_matrix,i = 1) = c.true_negatives[i] + c.false_negatives[i]
correctly_classified(c::confusion_matrix,i = 1) = c.true_positives[i] + c.true_negatives[i]
incorrectly_classified(c::confusion_matrix,i = 1) = c.false_positives[i] + c.false_negatives[i]
sensitivity(c::confusion_matrix,i = 1) = c.true_positives[i] / condition_positive(c,i)
specificity(c::confusion_matrix,i = 1) = c.true_negatives[i] / condition_negative(c,i)
precision(c::confusion_matrix,i = 1) = c.true_positives[i] / (c.true_positives[i] + c.false_positives[i])
accuracy(c::confusion_matrix,i = 1) = (c.true_positives[i] + c.true_negatives[i] ) / (condition_positive(c,i) + condition_negative(c,i))
balanced_accuracy(c::confusion_matrix,i = 1) = (sensitivity(c,i) +  specificity(c,i)) / 2
negative_predictive_value(c::confusion_matrix,i = 1)  = c.true_negatives[i] / (c.true_negatives[i] + c.false_negatives[i])
false_negative_rate(c::confusion_matrix,i = 1) = c.false_negatives[i] / condition_positive(c,i)
false_positive_rate(c::confusion_matrix,i = 1) = c.false_positives[i] / condition_negative(c,i)
false_discovery_rate(c::confusion_matrix,i = 1) = c.false_positives[i] / ( c.false_positives[i]  + c.true_negatives[i])
false_omission_rate(c::confusion_matrix,i = 1) = 1 - negative_predictive_value(c,i)
f1_score(c::confusion_matrix,i = 1) = (2* c.true_positives[i] ) / (2* c.true_positives[i] + c.false_positives[i] + c.false_negatives[i])


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
    println(io, pad, " ", labels)
    println(io, pad, "  ", lpad("+",len_p), "   ", lpad("-",len_n))
    println(io, pad, "┌", repeat("─",len_p+2), "┬", repeat("─",len_n+2), "┐")
    println(io, pad, "│ ", tp, " │ ", fn, " │ +")
    println(io, pad, "├", repeat("─",len_p+2), "┼", repeat("─",len_n+2), "┤   labels")
    println(io, pad, "│ ", fp, " │ ", tn, " │ -")
    println(io, pad, "└", repeat("─",len_p+2), "┴", repeat("─",len_n+2), "┘")
    println(io, pad, "matrix: ")
    println(io,pad, c.matrix)
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

function create_confusion_matrix(expected, predicted, labels)
    @assert size(expected) == size(predicted) "Sizes do not match"
    @assert size(expected)[1] == 1 && size(predicted)[1] == 1 && size(labels)[1] == 1 "Sizes of the expected, predicted and labels arrays must be 1 x n"
    dictionary = Dict()
    j = 1
    for i in labels
        dictionary[i] = j
        j += 1
    end
    println(dictionary)
    matrix = zeros(Number, size(labels)[2], size(labels)[2])
    for i in 1:size(expected)[2]
       println(dictionary[predicted[i]],dictionary[expected[i]])
       matrix[dictionary[predicted[i]],dictionary[expected[i]]] += 1
    end
    tp, tn, fp, fn = init_confusion_params(matrix)
    return confusion_matrix(tp,tn,fp,fn,matrix,labels)
end
