export confusion_matrix, class_confusion, visualize, classification_report, condition_positive, condition_negative, predicted_positive, predicted_negative, correctly_classified, incorrectly_classified, sensitivity, specificity, precision, accuracy, balanced_accuracy, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, fowlkes_mallows_index, informedness, matthews_correlation_coeff
using Base: show, length
using LinearAlgebra: normalize
using Plots: heatmap

function init_confusion_params(matrix)
    tp = []; tn = []; fp = []; fn = []
     @inbounds for i in 1:size(matrix)[1]
        push!(tp, matrix[i,i])
        push!(fn, sum(matrix[i,:]) - tp[i] )
        push!(fp, sum(matrix[:,i]) -tp[i])
        push!(tn, (sum(matrix) - tp[i] - fn[i] - fp[i]))
    end
    return tp, tn, fp, fn
end

struct confusion_matrix #immutable struct
    true_positives::Array{Int}
    true_negatives::Array{Int}
    false_positives::Array{Int}
    false_negatives::Array{Int}
    matrix::Array{Number,2}
    Labels::Array{Union{Int,AbstractString}}
end

"""
Creates a confusion matrix using the expected and predicted values of shape either 1 x n or n x 1.

labels: If no labels are provided the function will infer the labels from the union of the unique elements in expected and predicted. Given labels must be of size 1 x n or n x 1.  Only string and integers labels are accepted.

normalize: If true, the final confusion matrix will be normalized

"""
function confusion_matrix(expected, predicted; labels = nothing, normalize = false)
    @assert size(expected) == size(predicted) "Sizes do not match"
    if labels == nothing
        labels = union(sort(unique(expected)), sort(unique(predicted)))
    end
    dictionary = Dict()
    j = 1
    for i in labels
        dictionary[i] = j
        j += 1
    end
    matrix = zeros(Number, length(labels), length(labels))
    @inbounds for i in 1:length(expected)
       matrix[dictionary[predicted[i]],dictionary[expected[i]]] += 1
    end
    tp, tn, fp, fn = init_confusion_params(matrix)
    if 0 in tp
        @warn "There are elements of value 0 in the true positives array. This may lead to false values for some functions"
    end
    if 0 in tn
        @warn "There are elements of value 0 in the true negatives array. This may lead to false values for some functions"
    end
    if 0 in fp
       @warn "There are elements of value 0 in the false positives array. This may lead to false values for some functions"
    end
    if 0 in fn
        @warn "There are elements of value 0 in the false negatives array. This may lead to false values for some functions"
    end
    if normalize
       matrix = [round(i, digits = 3) for i in LinearAlgebra.normalize(matrix)]
    end
    return confusion_matrix(tp,tn,fp,fn,matrix,labels)
end


function class_confusion(c::confusion_matrix, i::Union{Int,AbstractString})
    index = findfirst(x -> x == i, c.Labels)
    return [c.true_positives[index] c.false_positives[index]; c.false_negatives[index] c.true_negatives[index]]
end

function visualize(c::confusion_matrix)
    converted_labels = []
    for i in c.Labels
        push!(converted_labels, string(i))
    end
    heatmap(converted_labels, converted_labels, c.matrix, c = :dense)
end

function Base.show(io::IO, ::MIME"text/plain", c::confusion_matrix)
    printer = Int(round(size(c.matrix)[1] / 2)) +1
    len = maximum([length(string(i)) for i in c.Labels])[1]
    label_size = length(c.Labels)
    label_len = eltype(c.Labels) == String ? len +5 : len+7
    println(io, lpad("Expected\n", printer* label_len ))
    println(io, [lpad(i,label_len) for i in c.Labels]...)
    println(io, repeat("_", length(c.Labels) * label_len))
    for i in 1:size(c.matrix)[1]
        println(io,  [lpad(string(i),label_len) for i in c.matrix[i,:]]..., "   │", c.Labels[i], i == printer ? "\t\tPredicted" : " ")
    end
end

function classification_report(c::confusion_matrix; io::IO = Base.stdout, return_dict = false)
    len = maximum([length(string(i)) for i in c.Labels])
    label_size = length(c.Labels)
    label_len = eltype(c.Labels) == String ? len +5 : len +7
    println(io,"Summary:\n", summary(c))
    println(io,"True Positives: ", c.true_positives)
    println(io,"False Positives: ", c.false_positives)
    println(io,"True Negatives: ", c.true_negatives)
    println(io,"False Negatives: ", c.false_negatives)
    println(io,"\n",lpad("Overall Statistics", label_len * Int(round(size(c.matrix)[1] / 2)+1)), "\n")
    println(io,lpad(" ", 30), [lpad(i, label_len) for i in c.Labels]...)
    println(io,lpad("Condition Positive:", 30), [lpad(round(condition_positive(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Condition Negative:", 30), [lpad(round(condition_negative(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Predicted Positive:", 30), [lpad(round(predicted_positive(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Predicted Negative:", 30), [lpad(round(predicted_negative(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Correctly Classified:", 30), [lpad(round(correctly_classified(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Incorrectly Classified:", 30), [lpad(round(incorrectly_classified(c,i), digits = 3), label_len) for i in 1: (label_size)]...)
    println(io,lpad("Sensitivity:", 30), [lpad(round(sensitivity(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Specificity:", 30), [lpad(round(specificity(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Precision:", 30) ,  [lpad(round(precision(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Accuracy:", 30 ) ,  [lpad(round(accuracy(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Balanced Accuracy:", 30),    [lpad(round(balanced_accuracy(c,i),digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Positive Predictive Value:", 30),    [lpad(round(positive_predictive_value(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Negative Predictive Value:", 30),    [lpad(round(negative_predictive_value(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("False Negative Rate:", 30), [lpad(round(false_negative_rate(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("False Positive Rate:", 30), [lpad(round(false_positive_rate(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("False Discovery Rate:", 30), [lpad(round(false_discovery_rate(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("False Omission Rate:", 30), [lpad(round(false_omission_rate(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("F1 Score:", 30), [lpad(round(f1_score(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Prevalence Threshold:", 30), [lpad(round(prevalence_threshold(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Threat Score:", 30), [lpad(round(threat_score(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Matthews Correlation Coefficient", 30), [lpad(round(matthews_correlation_coeff(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Fowlkes Mallows Index:", 30), [lpad(round(fowlkes_mallows_index(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Informedness:", 30), [lpad(round(informedness(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    println(io,lpad("Markedness:", 30), [lpad(round(markedness(c,i), digits = 3),label_len) for i in 1: (label_size)]...)
    if return_dict
        result_dict = Dict()
        result_dict["true-positives"] = c.true_positives
        result_dict["false-positives"] = c.false_positives
        result_dict["true-negatives"] = c.true_negatives
        result_dict["false-negatives"] = c.false_negatives
        result_dict["condition-positive"] = [condition_positive(c,i)   for i in 1:size(c.matrix)[1]]
        result_dict["condition-negative"] = [condition_negative(c,i)   for i in 1:size(c.matrix)[1]]
        result_dict["predicted-positive"] = [predicted_positive(c,i)  for i in 1:size(c.matrix)[1]]
        result_dict["predicted-negative"] = [predicted_negative(c,i)  for i in 1:size(c.matrix)[1]]
        result_dict["correctly-classified"] = [ correctly_classified(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["incorrectly-classified"] = [incorrectly_classified(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["sensitivity"] = [sensitivity(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["specificity"] = [specificity(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["precision"] = [ precision(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["accuracy"] = [ accuracy(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["balanced Accuracy"] = [ balanced_accuracy(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["positive-predictive-value"] =  [ positive_predictive_value(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["negative-predictive-value"] = [ negative_predictive_value(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["false-negative-rate"]  = [ false_negative_rate(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["false-positive-rate"]  = [ false_positive_rate(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["false-discovery-rate"] = [ false_discovery_rate(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["false-omission-rate"]  = [ false_omission_rate(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["f1-score"] = [ f1_score(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["prevalence-threshold"] = [ prevalence_threshold(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["threat-score"] = [ threat_score(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["matthews-correlation-coefficient"] = [ matthews_correlation_coeff(c,i) for i in 1:size(c.matrix)[1]]
        result_dict["fowlkes-mallows-index"] = [ fowlkes_mallows_index(c,i)  for i in 1:size(c.matrix)[1]]
        result_dict["informedness"] = [ informedness(c,i)    for i in 1:size(c.matrix)[1]]
        result_dict["markedness"] = [ markedness(c,i)   for i in 1:size(c.matrix)[1]]
        return result_dict
    end
end

condition_positive(c::confusion_matrix,i = 1) = c.true_positives[i] + c.false_negatives[i]
condition_negative(c::confusion_matrix,i = 1) = c.true_negatives[i] + c.false_positives[i]
predicted_positive(c::confusion_matrix,i = 1) = c.true_positives[i] + c.false_positives[i]
predicted_negative(c::confusion_matrix,i = 1) = c.true_negatives[i] + c.false_negatives[i]
correctly_classified(c::confusion_matrix,i = 1) = c.true_positives[i] + c.true_negatives[i]
incorrectly_classified(c::confusion_matrix,i = 1) = c.false_positives[i] + c.false_negatives[i]

function sensitivity(c::confusion_matrix,i = 1)
    x = c.true_positives[i] / condition_positive(c,i)
    return isnan(x) ? 0 : x
end

function specificity(c::confusion_matrix,i = 1)
    x = c.true_negatives[i] / condition_negative(c,i)
    return isnan(x) ? 0 : x
end

function precision(c::confusion_matrix,i = 1)
   x = c.true_positives[i] / (c.true_positives[i] + c.false_positives[i]);
   return isnan(x) ? 0 : x
end

function accuracy(c::confusion_matrix,i = 1)
    x = (c.true_positives[i] + c.true_negatives[i] ) / (condition_positive(c,i) + condition_negative(c,i))
    return isnan(x) ? 0 : x
end

function balanced_accuracy(c::confusion_matrix,i = 1)
    x = (sensitivity(c,i) +  specificity(c,i)) / 2; return isnan(x) ? 0 : x
end

function positive_predictive_value(c::confusion_matrix, i = 1)
   x = c.true_positives[i] / condition_positive(c, i); return isnan(x) ? 0 : x
end

function negative_predictive_value(c::confusion_matrix,i = 1)
    x = c.true_negatives[i] / (c.true_negatives[i] + c.false_negatives[i]); return isnan(x) ? 0 : x
end

function false_negative_rate(c::confusion_matrix,i = 1)
    x = c.false_negatives[i] / condition_positive(c,i); return isnan(x) ? 0 : x
end

function false_positive_rate(c::confusion_matrix,i = 1)
    x = c.false_positives[i] / condition_negative(c,i); return isnan(x) ? 0 : x
end

function false_discovery_rate(c::confusion_matrix,i = 1)
    x = c.false_positives[i] / ( c.false_positives[i]  + c.true_negatives[i]);
    return isnan(x) ? 0 : x
end

function false_omission_rate(c::confusion_matrix,i = 1)
    x = 1 - negative_predictive_value(c,i); return isnan(x) ? 0 : x
end

function f1_score(c::confusion_matrix,i = 1)
    x = (2* c.true_positives[i] ) / (2* c.true_positives[i] + c.false_positives[i] + c.false_negatives[i]); return isnan(x) ? 0 : x
end

function prevalence_threshold(c::confusion_matrix, i = 1)
    x = (sqrt(sensitivity(c,i) * (-specificity(c,i) +1)) + specificity(c,i) -1) / (sensitivity(c,i) + specificity(c,i) -1)
    return isnan(x) ? 0 : x
end

function threat_score(c::confusion_matrix, i = 1)
    x = c.true_positives[i] / (c.true_positives[i] + c.false_negatives[i] + c.false_positives[i])
    return isnan(x) ? 0 : x
end

function matthews_correlation_coeff(c::confusion_matrix, i = 1)
    x = (c.true_positives[i] * c.true_negatives[i] - c.false_positives[i] * c.false_negatives[i]) / sqrt( abs((c.true_positives[i] + c.false_positives[i]) * (c.true_positives[i] + c.false_negatives[i]) *
        (c.true_negatives[i] + c.false_positives[i]) * (c.true_negatives[i] + c.false_negatives[i])))
    return isnan(x) ? 0 : x
end

function fowlkes_mallows_index(c::confusion_matrix, i = 1)
    x = sqrt(positive_predictive_value(c,i) * sensitivity(c,i))
    return isnan(x) ? 0 : x
end

function informedness(c::confusion_matrix, i = 1)
    x = specificity(c,i) + sensitivity(c,i) -1
end

function markedness(c::confusion_matrix, i = 1)
   x = precision(c,i) * negative_predictive_value(c,i) -1
end
