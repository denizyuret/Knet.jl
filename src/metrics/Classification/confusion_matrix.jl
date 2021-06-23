export confusion_params, confusion_matrix, class_confusion

#Confusion matrix related classes

using LinearAlgebra

"""
    confusion_params(matrix::Array{Number,2})

Return the true positives, true negatives, false positives, false negatives arrays
from the given n x n matrix. If the provided matrix is not n x n, an assertion
exception: "Given matrix is not n x n" will be raised. As a visualization for the inner
calculation of the function, [this page](https://devopedia.org/images/article/208/6541.1566280388.jpg) may be visited

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
    if true in [isnan(i) for i in x]
        if zero_division == "warn" || zero_division == "0"
            if zero_division == "warn"; @warn "Zero division, replacing NaN with 0"; end;
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
`true_positives` is a single value. For multiclass, ith element in the array corresponds to the
`true_positives` of the ith class in the labels list.

**`true_negatives`** : An array that contains the true negative values of each label. For binary case,
`true_negatives` is a single value. For multiclass, ith element in the array corresponds to the
`true_negatives` of the ith class in the labels list.

**`false_positives`** : An array that contains the false positive values of each label. For binary case,
`false_positives` is a single value. For multiclass, ith element in the array corresponds to the
`false_positives` of the ith class in the labels list.

**false_negatives** : An array that contains the false negative values of each label. For binary case,
`false_negatives` is a single value. For multiclass, ith element in the array corresponds to the
`false_negatives` of the ith class in the labels list.

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

## Keywords

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


julia> y_true = ["emirhan", "knet", "metrics", "confusion", "knet", "confusion", "emirhan", "metrics", "confusion"];

julia> y_pred = ["knet", "knet", "confusion", "confusion", "knet", "emirhan", "emirhan", "knet", "confusion"];

julia> x = confusion_matrix(y_true, y_pred, labels = ["emirhan", "knet", "confusion", "metrics"])

Expected

emirhan           knet      confusion        metrics
____________________________________________________________
1              1              0              0   │emirhan
0              2              0              0   │knet
1              0              2              0   │confusion       Predicted
0              1              1              0   │metrics

```
## References
   [1] [Wikipedia entry for the Confusion matrix]
           (https://en.wikipedia.org/wiki/Confusion_matrix)
           (Note: Wikipedia and other references may use a different
           convention for axes)

_See: confusion params function_ \n
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

## Keywords

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
