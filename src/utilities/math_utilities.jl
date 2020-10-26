using LinearAlgebra

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


condition_positive(c::confusion_matrix) = c.true_positives + c.false_negatives
condition_negative(c::confusion_matrix) = c.true_negatives + false_positives
predicted_positive(c::confusion_matrix) = c.true_positives + c.false_positives
predicted_negative(c::confusion_matrix) = c.true_negavites + c.false_negatives
correctly_classified(c::confusion_matrix) = c.true_positives + c.true_negatives
incorrectly_classified(c::confusion_matrix) = c.false_positives + c.false_negatives
sensitivity(c::confusion_matrix) = c.true_positives / condition_positive(c)
specificity(c::confusion_matrix) = c.true_negatives / condition_negative(c)
precision(c::confusion_matrix) = c.true_positives / (true_positives + false_positives)
accuracy(c::confusion_matrix) = (c.true_positives + c.true_negatives) / (condition_positives(c) + condition_negative(c))

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
    @assert size(expected) == size(predicted)
    @assert sort(unique(expected)) == sort(unique(predicted))
    if labels == nothing
        labels = sort(unique(expected))
    end
    @assert length(unique(typeof.(labels))) == 1
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
