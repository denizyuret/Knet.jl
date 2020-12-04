export confusion_matrix, class_confusion, visualize, classification_report, condition_positive, condition_negative, predicted_positive,predicted_negative, correctly_classified, incorrectly_classified, sensitivity, recall, precision, positive_predictive_value, accuracy_score, balanced_accuracy, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, matthews_correlation_coeff, fowlkes_mallows_index, informedness, markedness, cohen_kappa_score, hamming_loss, jaccard_score, confusion_params
using Plots: heatmap
using Statistics: mean

"""
    confusion_params(matrix::Array{Number,2})

Return the true positives, true negatives, false positives, false negatives arrays
from the given n x n matrix. If the provided matrix is not n x n, an assertion
exception: "Given matrix is not n x n" will be raised. As a visualization for the inner
calculation of the, the following [this source](https://devopedia.org/images/article/208/6541.1566280388.jpg) may be visited

"""
function confusion_params(matrix::Array{Number,2})
    @assert size(matrix)[1] == size(matrix)[2] "Given matrix is not n x n "
    tp = []; tn = []; fp = []; fn = []
    matrix_sum = sum(matrix)
     @inbounds for i in 1:size(matrix)[1]
        push!(tp, matrix[i,i])
        push!(fn, sum(matrix[i,:]) - tp[i] )
        push!(fp, sum(matrix[:,i]) -tp[i])
        push!(tn, (matrix_sum - tp[i] - fn[i] - fp[i]))
    end
    return tp, tn, fp, fn
end

function check_index(x, none_accepted; class_name = nothing, ith_class = nothing)
    if !none_accepted; @assert class_name != nothing || ith_class != nothing "No class name or class indexing value provided"; end
    if none_accepted && class_name == nothing == ith_class
        return -1
    elseif class_name != nothing
        @assert class_name in x "There is no such class in the labels of the given confusion matrix"
        index = findfirst(x -> x == class_name, x)
        return index
    else
        @assert ith_class >= 0 && ith_class <= length(x) "ith_class value is not in range"
        return ith_class
    end
end

function clear_output(x, zero_division)
    if true in [isnan(i) || isinf(i) for i in x]
        if zero_division == "warn" || zero_division == "0"
            if zero_division == "warn"; @warn "Zero division, replacing NaN or Inf with 0"; end;
            if length(x) > 1
                return replace(x, NaN => 0)
            else
                return 0
            end
        else
            if length(x) > 1
                return replace(x, NaN => 1)
            else
                return 1
            end
        end
    else return x
    end
end

"""
A struct for representing confusion matrix and related computations

## Fields
**`true_positives`** : An array that contains the true positive values of each label. For binary case,
true positives is a single value. For multiclass, ith element in the array corresponds to the
true positives of the ith class in the labels list.

**`true_negatives`** : An array that contains the true negative values of each label. For binary case,
true negatives is a single value. For multiclass, ith element in the array corresponds to the
true negatives of the ith class in the labels list.

**`false_positives`** : An array that contains the false positive values of each label. For binary case,
false positives is a single value. For multiclass, ith element in the array corresponds to the
false positives of the ith class in the labels list.

**false_negatives** : An array that contains the false negative values of each label. For binary case,
false negatives is a single value. For multiclass, ith element in the array corresponds to the
false negatives of the ith class in the labels list.

**`matrix`** : an n x n matrix where n is the length of labels. It represents the actual confusion matrix
from which true positives, true negatives, false positives and false negatives are calculated.

**`Labels`** : an array representing the labels which are used for printing and visualization purposes

**`zero division`** :
    \n\t"warn" => all NaN and Inf values are replaced with zeros and user is warned by @warn macro in the
    \tprocess

    \t"0" => all NaN and Inf values are replaced with zeros but the user is not warned by @warn macro in the
    \tprocess

    \t"1" => all NaN and Inf values are replaced with ones but the user is not warned by @warn macro in the
    \tprocess


"""

struct confusion_matrix
    true_positives::Array{Int}
    true_negatives::Array{Int}
    false_positives::Array{Int}
    false_negatives::Array{Int}
    matrix::Array{Number,2}
    Labels::Array{Union{Int,AbstractString}}
    zero_division::String
end

"""
## Constructors

```confusion_matrix(expected::Array{T,1}, predicted::Array{T,1}; <keyword arguments>) where T <: Union{Int, String}```

Return a confusion matrix object constructed by the expected and predicted arrays. Expected and predicted arrays
must be of size (n,1) or or vector type. Lengths of the expected and predicted arrays must be equal; thus,
there should be a prediction and a ground truth for each classification.

## Arguments

    \n**`labels`** : vector-like of shape (n,1), default = nothing
    \nList of labels to index the matrix. If nothing is given, those that appear at least once
        in expected or predicted are used in sorted order.

    \n**`normalize`** : boolean, default = nothing
    \nDetermines whether or not the confusion matrix (matrix field of the produced confusion matrix) will be normalized.

    \n**`sample_weight`** : Number, default = nothing
    \nSample weights which will be filled in the matrix before confusion params function is called

    \n**`zero_division`** : "warn", "0", "1", default = "warn"
    \n_See:_ confusion matrix fields

## Example
\n
```julia-repl
julia> y_true = [1,1,1,2,3,3,1,2,1,1,2,1];

julia> y_pred = [1,3,2,1,2,3,1,1,2,3,2,1];

julia> x = confusion_matrix(y_true, y_pred)
\n┌ Warning: No labels provided, constructing a label set by union of the unique elements in Expected
and Predicted arrays\n

                  1      2      3
            _____________________
                  3      2      2   │1
                  2      1      0   │2     Predicted
                  0      1      1   │3

```
## References
   [1] [Wikipedia entry for the Confusion matrix]
           (https://en.wikipedia.org/wiki/Confusion_matrix)
           (Note: Wikipedia and other references may use a different
           convention for axes)

_See: confusion params function_ \n
_Source:_ [script](https://github.com/emirhan422/KnetMetrics/blob/main/src/metrics/classification_metrics.jl)

"""
function confusion_matrix(expected::Array{T,1}, predicted::Array{T,1}; labels = nothing, normalize = false, sample_weight = 0, zero_division = "warn") where T <: Union{Int, String}
    @assert length(expected) == length(predicted) "Sizes of the expected and predicted values do not match"
    @assert eltype(expected) <: Union{Int, String} &&  eltype(predicted) <: Union{Int, String} "Expected and Predicted arrays must either be integers or strings"
    @assert eltype(expected) == eltype(predicted) "Element types of Expected and Predicted arrays do not match"
    if labels != nothing; @assert length(labels) != 0 "Labels array must contain at least one value"; end;
    @assert zero_division in ["warn", "0", "1"] "Unknown zero division behaviour specification"
    if labels == nothing
        @warn "No labels provided, constructing a label set by union of the unique elements in Expected and Predicted arrays"
        labels = union(unique(expected),unique(predicted))
        if eltype(labels) == Int
            sort!(labels)
        end
    end
    dictionary = Dict()
    for i in 1:length(labels)
        dictionary[labels[i]] = i
    end
    matrix = zeros(Number, length(labels), length(labels))
    if sample_weight != 0
        fill!(matrix, sample_weight)
    end
    @inbounds for i in 1:length(expected)
       matrix[dictionary[expected[i]],dictionary[predicted[i]]] += 1
    end
    tp, tn, fp, fn = confusion_params(matrix)
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
    return confusion_matrix(tp,tn,fp,fn,matrix,labels, zero_division)
end

"""
```class_confusion(c::confusion_matrix; class_name = nothing, ith_class = nothing)```

\nReturn a binary confusion matrix for the class denoted by `class_name` or `ith_class` arguments.

## Arguments

**`ith_class`** : Int, default = nothing
\n\tReturn the binary confusion matrix of the ith class in the Labels array. This will be ignored if class_name is not `nothing`
**`class_name`** : Int, String, default = nothing
\n\tReturn the binary confusion matrix of the class of given value if exists in the Labels array.

## Example
\n
```julia-repl
julia> y_true = [1,1,1,2,3,3,1,2,1,1,2,1];

julia> y_pred = [1,3,2,1,2,3,1,1,2,3,2,1];

julia> x = confusion_matrix(y_true, y_pred)
\n┌ Warning: No labels provided, constructing a label set by union of the unique elements in Expected
and Predicted arrays

julia> class_confusion(x, ith_class = 2)
2×2 Array{Int64,2}:
 1  3
 2  6

julia> class_confusion(x, class_name = 2)
2×2 Array{Int64,2}:
 1  3
 2  6
```
"""
function class_confusion(c::confusion_matrix; class_name = nothing, ith_class = nothing)
    index = check_index(c.Labels, false ,class_name = class_name, ith_class = ith_class)
    return [c.true_positives[index] c.false_positives[index]; c.false_negatives[index] c.true_negatives[index]]
end

"""
```visualize(c::confusion_matrix)```

Visualize the matrix of the given confusion matrix object by heatmap function of the Plots library.
"""
function visualize(c::confusion_matrix)
    converted_labels = []
    for i in c.Labels
        push!(converted_labels, string(i))
    end
    heatmap(converted_labels, converted_labels, c.matrix, c = :dense)
end

function Base.show(io::IO, ::MIME"text/plain", c::confusion_matrix)
    printer = Int(round(size(c.matrix)[1] / 2)) +1
    label_len = maximum([length(string(i)) for i in c.Labels])[1] + 6
    label_size = length(c.Labels)
    println(io, lpad("Expected\n", printer* label_len ))
    println(io, [lpad(i,label_len) for i in c.Labels]...)
    println(io, repeat("_", length(c.Labels) * label_len))
    for i in 1:size(c.matrix)[1]
        println(io,  [lpad(string(i),label_len) for i in c.matrix[i,:]]..., "   │", c.Labels[i], i == printer ? "\tPredicted" : " ")
    end
end

"""
```classification_report(c::confusion_matrix;<keyword arguments>)```

Return all the values listed below if `return_dict` is true. Else, write the values to the given IO element.

Returned dictionary:
```
    "true-positives" => c.true_positives
    "false-positives" => c.false_positives
    "true-negatives" => c.true_negatives
    "false-negatives" => c.false_negatives
    "condition-positive" => condition_positive(c)
    "condition-negative" => condition_negative(c)
    "predicted-positive" => predicted_positive(c)
    "predicted-negative" => predicted_negative(c)
    "correctly-classified" => correctly_classified(c)
    "incorrectly-classified" => incorrectly_classified(c)
    "sensitivity" => sensitivity(c)
    "specificity" => specificity(c)
    "precision" => precision(c)
    "accuracy-score" => accuracy_score(c)
    "balanced Accuracy" => balanced_accuracy(c)
    "positive-predictive-value" =>  positive_predictive_value(c)
    "negative-predictive-value" => negative_predictive_value(c)
    "false-negative-rate"  => false_negative_rate(c)
    "false-positive-rate"  => false_positive_rate(c)
    "false-discovery-rate" => false_discovery_rate(c)
    "false-omission-rate"  => false_omission_rate(c)
    "f1-score" => f1_score(c)
    "prevalence-threshold" => prevalence_threshold(c)
    "threat-score" => threat_score(c)
    "matthews-correlation-coefficient" => matthews_correlation_coeff(c)
    "fowlkes-mallows-index" => fowlkes_mallows_index(c)
    "informedness" => informedness(c)
    "markedness" => markedness(c)
    "jaccard-score-nonaverage" => jaccard_score(c, average = nothing)
    "jaccard-score-microaverage" => jaccard_score(c, average = "micro")
    "hamming-loss" => jaccard_score(c)
    "cohen-kappa-score" => cohen_kappa_score(c)
```

For a sample output to the given IO element, see Example section.

## Arguments

    \n**`io`** : ::IO, default = Base.stdout
    \n\tIO element to write to

    \n**`return_dict`** : default = false
    \n\tReturn a dictionary as specified below if true; print the values specified below if false

    \n**`target_names`** : vector-like, default = nothing
    \n\tIf not nothing, replace the labels of the given confusion matrix object whilst printing

    \n**`digits`** : Int, default = 2
    \n\tDetermines how the rounding procedure will be digitized. If `return_dict` is true, this will be ignored and the values
    will be placed into the dictionary with full precision

## Example

```julia-repl

julia> y_true = [1,1,1,2,3,3,1,2,1,1,2,1];

julia> y_pred = [1,3,2,1,2,3,1,1,2,3,2,1];

julia> x = confusion_matrix(y_true, y_pred)
┌ Warning: No labels provided, constructing a label set by union of the unique elements in Expected and Predicted arrays
└ @ Path
julia> classification_report(x)
Summary:
confusion_matrix
True Positives: [3, 1, 1]
False Positives: [2, 3, 2]
True Negatives: [3, 6, 8]
False Negatives: [4, 2, 1]

Labelwise Statistics

                                  1    2    3
           Condition Positive:  7.0  3.0  2.0
           Condition Negative:  5.0  9.0 10.0
           Predicted Positive:  5.0  4.0  3.0
           Predicted Negative:  7.0  8.0  9.0
         Correctly Classified:  6.0  7.0  9.0
       Incorrectly Classified:  6.0  5.0  3.0
                  Sensitivity: 0.43 0.33  0.5
                  Specificity:  0.6 0.67  0.8
                    Precision:  0.6 0.25 0.33
               Accuracy Score:  0.5 0.58 0.75
            Balanced Accuracy: 0.51  0.5 0.65
    Negative Predictive Value: 0.43 0.75 0.89
          False Negative Rate: 0.57 0.67  0.5
          False Positive Rate:  0.4 0.33  0.2
         False Discovery Rate:  0.4 0.33  0.2
          False Omission Rate: 0.57 0.25 0.11
                     F1 Score:  0.5 0.29  0.4
                Jaccard Score: 0.33 0.17 0.25
┌ Warning: Zero division, replacing NaN or Inf with 0
└ @ Main Path
         Prevalence Threshold:16.73  Inf 1.05
                 Threat Score: 0.33 0.17 0.25
Matthews Correlation Coefficient 0.03  0.0 0.26
        Fowlkes Mallows Index: 0.51 0.29 0.41
                 Informedness: 0.03  0.0  0.3
                   Markedness:-0.74-0.81 -0.7

General Statistics

              Accuracy Score:   0.1527777777777778
           Cohen Kappa Score:   0.07692307692307698
                Hamming Loss:   0.5833333333333334
               Jaccard Score:   0.2631578947368421
```
"""
function classification_report(c::confusion_matrix; io::IO = Base.stdout, return_dict = false, target_names = nothing, digits = 2)
    if return_dict
        result_dict = Dict()
        result_dict["true-positives"] = c.true_positives
        result_dict["false-positives"] = c.false_positives
        result_dict["true-negatives"] = c.true_negatives
        result_dict["false-negatives"] = c.false_negatives
        result_dict["condition-positive"] = condition_positive(c)
        result_dict["condition-negative"] = condition_negative(c)
        result_dict["predicted-positive"] = predicted_positive(c)
        result_dict["predicted-negative"] = predicted_negative(c)
        result_dict["correctly-classified"] = correctly_classified(c)
        result_dict["incorrectly-classified"] = incorrectly_classified(c)
        result_dict["sensitivity"] = sensitivity(c)
        result_dict["specificity"] = specificity(c)
        result_dict["precision"] = precision(c)
        result_dict["accuracy-score"] = accuracy_score(c)
        result_dict["balanced Accuracy"] = balanced_accuracy(c)
        result_dict["positive-predictive-value"] =  positive_predictive_value(c)
        result_dict["negative-predictive-value"] = negative_predictive_value(c)
        result_dict["false-negative-rate"]  = false_negative_rate(c)
        result_dict["false-positive-rate"]  = false_positive_rate(c)
        result_dict["false-discovery-rate"] = false_discovery_rate(c)
        result_dict["false-omission-rate"]  = false_omission_rate(c )
        result_dict["f1-score"] = f1_score(c)
        result_dict["prevalence-threshold"] = prevalence_threshold(c)
        result_dict["threat-score"] = threat_score(c)
        result_dict["matthews-correlation-coefficient"] = matthews_correlation_coeff(c)
        result_dict["fowlkes-mallows-index"] = fowlkes_mallows_index(c)
        result_dict["informedness"] = informedness(c)
        result_dict["markedness"] = markedness(c)
        result_dict["jaccard-score-nonaverage"] = jaccard_score(c, average = nothing)
        result_dict["jaccard-score-microaverage"] = jaccard_score(c, average = "micro")
        result_dict["hamming-loss"] = jaccard_score(c)
        result_dict["cohen-kappa-score"] = cohen_kappa_score(c)
        return result_dict
    else
        labels = target_names != nothing && length(target_names) == length(c.Labels) ? target_names : c.Labels
        len = maximum([length(string(i)) for i in labels])
        label_size = length(c.Labels)
        label_len = len + digits + 2
        println(io,"Summary:\n", summary(c))
        println(io,"True Positives: ", c.true_positives)
        println(io,"False Positives: ", c.false_positives)
        println(io,"True Negatives: ", c.true_negatives)
        println(io,"False Negatives: ", c.false_negatives)
        println(io,"\n",lpad("Labelwise Statistics", label_len * Int(round(size(c.matrix)[1] / 2)+1)), "\n")
        println(io,lpad(" ", 30), [lpad(i, label_len) for i in labels]...)
        println(io,lpad("Condition Positive:", 30), [lpad(round(i, digits = digits), label_len) for i in condition_positive(c)]...)
        println(io,lpad("Condition Negative:", 30), [lpad(round(i, digits = digits), label_len) for i in condition_negative(c)]...)
        println(io,lpad("Predicted Positive:", 30), [lpad(round(i, digits = digits), label_len) for i in predicted_positive(c)]...)
        println(io,lpad("Predicted Negative:", 30), [lpad(round(i, digits = digits), label_len) for i in predicted_negative(c)]...)
        println(io,lpad("Correctly Classified:", 30), [lpad(round(i, digits = digits), label_len) for i in correctly_classified(c)]...)
        println(io,lpad("Incorrectly Classified:", 30), [lpad(round(i, digits = digits), label_len) for i in incorrectly_classified(c)]...)
        println(io,lpad("Sensitivity:", 30), [lpad(round(i, digits = digits), label_len) for i in sensitivity(c)]...)
        println(io,lpad("Specificity:", 30), [lpad(round(i, digits = digits), label_len) for i in specificity(c)]...)
        println(io,lpad("Precision:", 30) , [lpad(round(i, digits = digits), label_len) for i in precision(c)]...)
        println(io,lpad("Accuracy Score:", 30 ) ,  [lpad(round(accuracy_score(c, ith_class = i), digits = digits), label_len) for i in 1:label_size]...)
        println(io,lpad("Balanced Accuracy:", 30), [lpad(round(i, digits = digits), label_len) for i in balanced_accuracy(c)]...)
        println(io,lpad("Negative Predictive Value:", 30), [lpad(round(i, digits = digits), label_len) for i in negative_predictive_value(c)]...)
        println(io,lpad("False Negative Rate:", 30), [lpad(round(i, digits = digits), label_len) for i in false_negative_rate(c)]...)
        println(io,lpad("False Positive Rate:", 30), [lpad(round(i, digits = digits), label_len) for i in false_positive_rate(c)]...)
        println(io,lpad("False Discovery Rate:", 30), [lpad(round(i, digits = digits), label_len) for i in false_discovery_rate(c)]...)
        println(io,lpad("False Omission Rate:", 30), [lpad(round(i, digits = digits), label_len) for i in false_omission_rate(c)]...)
        println(io,lpad("F1 Score:", 30), [lpad(round(i, digits = digits), label_len) for i in f1_score(c)]...)
        println(io,lpad("Jaccard Score:", 30), [lpad(round(i, digits = digits), label_len) for i in jaccard_score(c, average = nothing)]...)
        println(io,lpad("Prevalence Threshold:", 30), [lpad(round(i, digits = digits), label_len) for i in prevalence_threshold(c)]...)
        println(io,lpad("Threat Score:", 30), [lpad(round(i, digits = digits), label_len) for i in threat_score(c)]...)
        println(io,lpad("Matthews Correlation Coefficient", 30), [lpad(round(i, digits = digits), label_len) for i in matthews_correlation_coeff(c)]...)
        println(io,lpad("Fowlkes Mallows Index:", 30), [lpad(round(i, digits = digits), label_len) for i in fowlkes_mallows_index(c)]...)
        println(io,lpad("Informedness:", 30), [lpad(round(i, digits = digits), label_len) for i in informedness(c)]...)
        println(io,lpad("Markedness:", 30), [lpad(round(i, digits = digits), label_len) for i in markedness(c)]...)
        println(io,"\n",lpad("General Statistics", label_len * Int(round(size(c.matrix)[1] / 2)+1)), "\n")
        println(io, lpad("Accuracy Score:\t",30), accuracy_score(c))
        println(io, lpad("Cohen Kappa Score:\t", 30), cohen_kappa_score(c))
        println(io, lpad("Hamming Loss:\t", 30), hamming_loss(c))
        println(io, lpad("Jaccard Score:\t", 30), jaccard_score(c, average = "micro"))
    end
end

"""
```condition_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return condition positive values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Condition Positive: True Positives + False Negatives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition positive values for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_true = ["sample",  "knet", "metrics", "function",  "knet", "knet","function"];

julia> y_pred = ["sample",  "knet", "sample", "function",  "knet", "knet","knet"];

julia> x = confusion_matrix(y_true, y_pred);

julia> condition_positive(x)
4-element Array{Int64,1}:
 1
 3
 1
 2

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_true = ["sample",  "knet", "metrics", "function",  "knet", "knet","function"];

julia> y_pred = ["sample",  "knet", "sample", "function",  "knet", "knet","knet"];

julia> x = confusion_matrix(y_true, y_pred);

julia> condition_positive(x, class_name = "knet")
3

```

"""
function condition_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives + c.false_negatives
        return clear_output(x,c.zero_division)
    else
        x = c.true_positives[index] + c.false_negatives[index]
        return clear_output(x,c.zero_division)
    end
end

"""
```condition_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return condition positive values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Condition Negative: True Negatives + False Positives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return condition negative values for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [1,2,3,1,4,2,1,2,3,4,1,2,3,4,1,2,2,2,3,1];

julia> y_true = [1,2,1,1,4,2,4,2,2,4,1,2,3,2,1,2,2,2,1,1];

julia> x = confusion_matrix(y_pred, y_true, labels = [1,2,3,4]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> condition_negative(x)
4-element Array{Int64,1}:
 14
 13
 16
 17

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_pred = [1,2,3,1,4,2,1,2,3,4,1,2,3,4,1,2,2,2,3,1];

julia> y_true = [1,2,1,1,4,2,4,2,2,4,1,2,3,2,1,2,2,2,1,1];

julia> x = confusion_matrix(y_pred, y_true, labels = [1,2,3,4]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> condition_negative(x, class_name = 3)
16


```

"""
function condition_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_negatives + c.false_positives
        return clear_output(x,c.zero_division)
    else
        x = c.true_negatives[index] + c.false_positives[index]
        return clear_output(x,c.zero_division)
    end
end

"""
```predicted_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return predicted positive values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Predicted Positive: True Positives + False Positives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return predicted positive values for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia>  predicted_positive(x)
4-element Array{Int64,1}:
 7
 9
 1
 3

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> predicted_positive(x, class_name = 3)
9


```
"""
function predicted_positive(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives + c.false_positives
        return clear_output(x,c.zero_division)
    else
        x = c.true_positives[index] + c.false_positives[index]
        return clear_output(x,c.zero_division)
    end
end

"""
```predicted_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return predicted negative values of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Predicted Negative: Negatives + False Negatives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return predicted negative values for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> predicted_negative(x)
4-element Array{Int64,1}:
 13
 11
 19
 17

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> predicted_negative(x, class_name = 4)
19

```
"""
function predicted_negative(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_negatives + c.false_negatives
        return clear_output(x,c.zero_division)
    else
        x = c.true_negatives[index] + c.false_negatives[index]
        return clear_output(x,c.zero_division)
    end
end

"""
```correctly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return number of correctly classified instances of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Correctly Classified: True Positives + True Negatives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return number of correctly classified instances for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> correctly_classified(x)
4-element Array{Int64,1}:
 17
 18
 17
 18

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> correctly_classified(x, ith_class = 1)
17
```

"""
function correctly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives + c.true_negatives
        return clear_output(x,c.zero_division)
    else
        x = c.true_positives[index] + c.true_negatives[index]
        return clear_output(x,c.zero_division)
    end
end

"""
```incorrectly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing)```

Return number of incorrectly classified instances of either the whole confusion matrix or the classes specified by `class_name` or `ith_class`
arguments.

    Inorrectly Classified: False Negatives + False Positives

## Arguments

**`ith_class`** : Int, default = nothing
Return the results for the ith class in the ith label of the label list of the given confusion matrix object.

**`class_name`** : Int/String, default = nothing
Return the results for the class of the speicifed value in the ith label of the label list of the given confusion matrix object.

If both `class_name` and `ith_class` arguments are equal to `nothing`, return number of incorrectly classified instances for all the elements in the labels arrays

## Example

First example no indexing:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> incorrectly_classified(x)
4-element Array{Int64,1}:
 3
 2
 3
 2

```

Second example value of a specific class:\n\n

```julia-repl
julia> y_pred = [2, 3, 4, 2, 5, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 3, 3, 4, 2];

julia> y_true = [2, 3, 2, 2, 5, 3, 5, 3, 3, 5, 2, 3, 4, 3, 2, 3, 3, 3, 2, 2];

julia> x = confusion_matrix(y_pred, y_true, labels = [2,3,4,5]);
┌ Warning: There are elements of value 0 in the false positives array. This may lead to false values for some functions
└ @ Path
┌ Warning: There are elements of value 0 in the false negatives array. This may lead to false values for some functions
└ @ Path

julia> incorrectly_classified(x, ith_class = 2)
2

```

"""
function incorrectly_classified(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.false_positives + c.false_negatives
        return clear_output(x,c.zero_division)
    else
        x = c.false_positives[index] + c.false_negatives[index]
        return clear_output(x,c.zero_division)
    end
end

"""



"""
function sensitivity(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives ./ condition_positive(c)
        return clear_output(x, c.zero_division)
    else
        x = c.true_positives[index] / condition_positive(c, ith_class = index)
        return clear_output(x, c.zero_division)
    end
end


"""


"""
function recall(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    return sensitivity(c, ith_class = ith_class, class_name = class_name)
end


"""

"""
function specificity(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_negatives ./ condition_negative(c)
        return clear_output(x, c.zero_division)
    else
        x = c.true_negatives[index] / condition_negative(c, ith_class = index)
        return clear_output(x, c.zero_division)
    end
end


"""


"""
function precision(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = c.true_positives ./ (c.true_positives + c.false_positives)
        return clear_output(x, c.zero_division)
    else
        x = c.true_positives[index] / (c.true_positives[index] + c.false_positives[index])
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function positive_predictive_value(c::confusion_matrix; ith_class = nothing, class_name = nothing)
   return precision(c, class_name = class_name, ith_class = ith_class)
end

"""

"""
function accuracy_score(c::confusion_matrix; ith_class = nothing, class_name = nothing, normalize = true, sample_weight = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        accuracy_array = [accuracy_score(c, ith_class = i) for i in 1:length(c.true_positives)]
        if normalize
            x = sample_weight == nothing ? (sum(accuracy_array) / sum(c.matrix)) : (sum(accuracy_array .* sample_weight) / sum(c.matrix))
            return clear_output(x,c.zero_division)
        else
            x = sample_weight == nothing ? sum(accuracy_array) : dot(accuracy_array, sample_weight)
            return clear_output(x,c.zero_division)
        end
    else
        if normalize
            x = (c.true_positives[index] + c.true_negatives[index] ) / (condition_positive(c, ith_class = index) + condition_negative(c, ith_class = index))
            return clear_output(x, c.zero_division)
        else
            x = (c.true_positives[index])
            return clear_output(x, c.zero_division)
        end
    end
end

"""



"""
function balanced_accuracy(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [(sensitivity(c, ith_class = i) +  specificity(c, ith_class = i)) / 2 for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = (sensitivity(c,ith_class = index) +  specificity(c, ith_class = index)) / 2
        return clear_output(x,c.zero_division)
    end
end

"""



"""
function negative_predictive_value(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [c.true_negatives[i] / (c.true_negatives[i] + c.false_negatives[i]) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = c.true_negatives[index] / (c.true_negatives[index] + c.false_negatives[index])
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function false_negative_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [c.false_negatives[i] / condition_positive(c,ith_class = i) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = c.false_negatives[i] / condition_positive(c,ith_class = index)
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function false_positive_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [c.false_positives[i] / condition_negative(c,ith_class = i) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = c.false_negatives[i] / condition_positive(c,ith_class = index)
        return clear_output(x,c.zero_division)
    end
end

"""


"""
function false_discovery_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [c.false_positives[i] / ( c.false_positives[i]  + c.true_negatives[i]) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = c.false_positives[index] / ( c.false_positives[index]  + c.true_negatives[index])
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function false_omission_rate(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [1 - negative_predictive_value(c,ith_class = i) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = 1 - negative_predictive_value(c,ith_class = index)
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function f1_score(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [(2* c.true_positives[i] ) / (2* c.true_positives[i] + c.false_positives[i] + c.false_negatives[i]) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = (2* c.true_positives[index] ) / (2* c.true_positives[index] + c.false_positives[index] + c.false_negatives[index])
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function prevalence_threshold(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [(sqrt(abs(sensitivity(c,ith_class = i) * (-specificity(c,ith_class = i) +1) + specificity(c,ith_class = i) -1)) / (sensitivity(c,ith_class = i) + specificity(c,ith_class = i) -1)) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = (sqrt(abs(sensitivity(c,ith_class = index) * (-specificity(c,ith_class = index) +1) + specificity(c,ith_class = index) -1)) / (sensitivity(c,ith_class = index) + specificity(c,ith_class = index) -1))
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function threat_score(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [c.true_positives[i] / (c.true_positives[i] + c.false_negatives[i] + c.false_positives[i]) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = c.true_positives[index] / (c.true_positives[index] + c.false_negatives[index] + c.false_positives[index])
        return clear_output(x,c.zero_division)
    end
end

"""


"""
function matthews_correlation_coeff(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [(c.true_positives[i] * c.true_negatives[i] - c.false_positives[i] * c.false_negatives[i]) / sqrt( abs((c.true_positives[i] + c.false_positives[i]) * (c.true_positives[i] + c.false_negatives[i]) *
            (c.true_negatives[i] + c.false_positives[i]) * (c.true_negatives[i] + c.false_negatives[i])))
         for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = (c.true_positives[index] * c.true_negatives[index] - c.false_positives[index] * c.false_negatives[index]) / sqrt( abs((c.true_positives[index] + c.false_positives[index]) * (c.true_positives[index] + c.false_negatives[index]) *
            (c.true_negatives[index] + c.false_positives[index]) * (c.true_negatives[index] + c.false_negatives[index])))
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function fowlkes_mallows_index(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [sqrt(positive_predictive_value(c,ith_class = i) * sensitivity(c,ith_class = i)) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = sqrt(positive_predictive_value(c,ith_class = index) * sensitivity(c,ith_class = index))
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function informedness(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [ ( specificity(c,ith_class = i) + sensitivity(c,ith_class = i) -1) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = specificity(c,ith_class = index) + sensitivity(c,ith_class = index) -1
        return clear_output(x,c.zero_division)
    end
end

"""

"""
function markedness(c::confusion_matrix; ith_class = nothing, class_name = nothing)
    index = check_index(c.Labels, true, ith_class = ith_class, class_name = class_name)
    if index == -1
        x = [( precision(c,ith_class = i) * negative_predictive_value(c,ith_class = i) -1) for i in 1:length(c.true_positives)]
        return clear_output(x,c.zero_division)
    else
        x = specificity(c,ith_class = index) + sensitivity(c,ith_class = index) -1
        return clear_output(x,c.zero_division)
    end
end

"""
```cohen_kappa_score(c::confusion_matrix; weights = nothing) ```

Return Cohen's Kappa (a statistic that measures inter-annotator agreement)


"""
function cohen_kappa_score(c::confusion_matrix; weights = nothing)
#reference: scikitlearn.metrics.classification.cohen_kappa_score
    @assert weights in [nothing, "quadratic", "linear"] "Unknown kappa weighting type"
    w_mat = nothing
    sum0 = sum(c.matrix, dims = 1)
    sum1 = sum(c.matrix, dims = 2)
    expected = (sum1 * sum0) ./ sum(sum0)
    if weights == nothing
        w_mat = ones(length(c.Labels),length(c.Labels))
        for i in 1:length(c.Labels)
            w_mat[i,i] = 0
        end
    else
        w_mat = zeros(length(c.Labels),length(c.Labels))
        w_mat += [i for i in 1:length(c.Labels)]
        if weights == "linear"
            w_mat = abs(w_mat - transpose(w_mat))
        else
            w_mat = (w_mat - transpose(w_mat)) ^2
        end
    end
    x = sum(w_mat .* c.matrix) / sum(w_mat .* expected)
    return clear_output(1- x,c.zero_division)
end

"""


"""
function hamming_loss(c::confusion_matrix;)
    x = zeros(sum(c.matrix))
    x[1] = sum(c.false_negatives)
    return clear_output(mean(x), c.zero_division)
end

"""


"""
function jaccard_score(c::confusion_matrix; average = "binary", sample_weight = nothing)
    @assert average in [nothing, "binary", "weighted", "samples", "micro", "macro"] "Unknown averaging type"
    if sample_weight != nothing @assert length(sample_weight) == length(c.true_positives) "Dimensions of given sample weight does not match the confusion matrix"; end
    numerator = c.true_positives
    denominator =  c.true_positives + c.false_negatives + c.false_positives
    if average == nothing
        x = numerator ./ denominator
        return clear_output(x, c.zero_division)
    elseif average == "micro"
        numerator = sum(numerator)
        denominator = sum(denominator)
        x = numerator ./ denominator
        return clear_output(x, c.zero_division)
    elseif average == "macro"
        numerator = c.true_positives
        denominator =  c.true_positives + c.false_negatives + c.false_positives
        x = numerator ./ denominator
        return clear_output(mean(x), c.zero_division)
    elseif average == "weighted" || average == "samples"
        weights = nothing
        if average == "weighted"
            weights = c.false_negatives + c.true_positives
        elseif average == "samples"
            weights = sample_weight
        end
        score = numerator ./ denominator
        x = [weights[i] * score[i] for i in 1:length(c.Labels)]
        x / length(c.Labels)
        return clear_output(x, c.zero_division)
    else
        x = [(c.false_negatives[i] + c.true_positives[i])/length(c.Labels) for i in 1:length(c.Labels)]
        return clear_output(x, c.zero_division)
    end
end
