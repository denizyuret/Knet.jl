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
sensitivity(c::confusion_matrix,i = 1) = x = c.true_positives[i] / condition_positive(c,i); return isnan(x) ? 0 : x
specificity(c::confusion_matrix,i = 1) = x = c.true_negatives[i] / condition_negative(c,i); return isnan(x) ? 0 : x
precision(c::confusion_matrix,i = 1) = x = c.true_positives[i] / (c.true_positives[i] + c.false_positives[i]); return isnan(x) ? 0 : x
accuracy(c::confusion_matrix,i = 1) = x = (c.true_positives[i] + c.true_negatives[i] ) / (condition_positive(c,i) + condition_negative(c,i)); return isnan(x) ? 0 : x
balanced_accuracy(c::confusion_matrix,i = 1) = x = (sensitivity(c,i) +  specificity(c,i)) / 2; return isnan(x) ? 0 : x
negative_predictive_value(c::confusion_matrix,i = 1)  = x = c.true_negatives[i] / (c.true_negatives[i] + c.false_negatives[i]); return isnan(x) ? 0 : x
false_negative_rate(c::confusion_matrix,i = 1) = x = c.false_negatives[i] / condition_positive(c,i); return isnan(x) ? 0 : x
false_positive_rate(c::confusion_matrix,i = 1) = x = c.false_positives[i] / condition_negative(c,i); return isnan(x) ? 0 : x
false_discovery_rate(c::confusion_matrix,i = 1) = x = c.false_positives[i] / ( c.false_positives[i]  + c.true_negatives[i]); return isnan(x) ? 0 : x
false_omission_rate(c::confusion_matrix,i = 1) = x = 1 - negative_predictive_value(c,i); return isnan(x) ? 0 : x
f1_score(c::confusion_matrix,i = 1) = x = (2* c.true_positives[i] ) / (2* c.true_positives[i] + c.false_positives[i] + c.false_negatives[i]); return isnan(x) ? 0 : x


function Base.show(io::IO, ::MIME"text/plain", c::confusion_matrix)
    len = findmax([length(string(i)) for i in c.Labels])[1]
    label_size = size(c.Labels)[2]
    label_padding = lpad(" ", label_size * len *2)

    println(io, [lpad(i,len * 2) for i in c.Labels]...)
    println(io, repeat("_", size(c.Labels)[2] * len * 2 + len * 2))
    for i in 1:size(c.matrix)[1]
        println(io,  [lpad(string(i),len * 2) for i in c.matrix[i,:]]..., "   │", c.Labels[i] )
    end
    println(io, "Summary:\n", summary(c))
    println(io, "True Positives: ", c.true_positives)
    println(io, "False Positives: ", c.false_positives)
    println(io, "True Negatives: ", c.true_negatives)
    println(io, "False Negatives: ", c.false_negatives)
    println(io, "\n",lpad("Overall Statistics", 50, "\n"))
    println(io, lpad(" ", 30), [lpad(i, 7) for i in c.Labels]...)
    println(lpad("Condition Positive:", 30), [lpad(round(condition_positive(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Condition Negative:", 30), [lpad(round(condition_negative(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Predicted Positive:", 30), [lpad(round(predicted_positive(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Predicted Negative:", 30), [lpad(round(predicted_negative(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Correctly Classified:", 30), [lpad(round(correctly_classified(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Incorrectly Classified:", 30), [lpad(round(incorrectly_classified(c,i), digits = 3), 7) for i in 1: (label_size-1)]...)
    println(lpad("Sensitivity:", 30), [lpad(round(sensitivity(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("Specificity:", 30), [lpad(round(specificity(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("Precision:", 30) ,  [lpad(round(precision(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("Accuracy:", 30 ) ,  [lpad(round(accuracy(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("Balanced Accuracy:", 30),    [lpad(round(balanced_accuracy(c,i),digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("Negative Predictive Value:", 30),    [lpad(round(negative_predictive_value(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("False Negative Rate:", 30), [lpad(round(false_negative_rate(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("False Positive Rate:", 30), [lpad(round(false_positive_rate(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("False Discovery Rate:", 30), [lpad(round(false_discovery_rate(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("False Omission Rate:", 30), [lpad(round(false_omission_rate(c,i), digits = 3),7) for i in 1: (label_size-1)]...)
    println(lpad("F1 Score:", 30), [lpad(round(f1_score(c,i), digits = 3),7) for i in 1: (label_size-1)]...)

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
    matrix = zeros(Number, size(labels)[2], size(labels)[2])
    for i in 1:size(expected)[2]
       matrix[dictionary[predicted[i]],dictionary[expected[i]]] += 1
    end
    tp, tn, fp, fn = init_confusion_params(matrix)
    return confusion_matrix(tp,tn,fp,fn,matrix,labels)
end
