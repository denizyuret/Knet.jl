#Visualization Functions
export visualize

using Plots

gr()

function _plot(c::confusion_matrix; func, type, title, labels = nothing)
    x = nothing
    if type == "condition-positive"; x = func(labels, condition_positive(c) , title = title, labels = permutedims(labels))
    elseif type == "condition-negative"; x = func(labels, condition_negative(c) , title = title, labels = permutedims(labels))
    elseif type == "predicted-positive"; x = func(labels, predicted_positive(c) , title = title, labels = permutedims(labels))
    elseif type == "predicted-negative"; x = func(labels, predicted_positive(c) , title = title, labels = permutedims(labels))
    elseif type == "correctly-classified"; x = func(labels, correctly_classified(c) , title = title, labels = permutedims(labels))
    elseif type == "incorrectly-classified"; x = func(labels, incorrectly_classified(c) , title = title, labels = permutedims(labels))
    elseif type == "sensitivity-score"; x = func(labels, sensitivity_score(c) , title = title, labels = permutedims(labels))
    elseif type == "recall-score"; x = func(labels, recall_score(c), title = title, labels = permutedims(labels))
    elseif type == "specificity-score";x =  func(labels, specificity_score(c) ,title = title, labels = permutedims(labels))
    elseif type == "precision-score";x =  func(labels, precision_score(c), title = title, labels = permutedims(labels))
    elseif type == "positive-predictive-value"; x = func(labels, positive_predictive_value(c) , title = title, labels = permutedims(labels))
    elseif type == "accuracy-score"; x = func(labels, accuracy_score(c), title = title, labels = permutedims(labels))
    elseif type == "balanced-accuracy-score"; x = func(labels, balanced_accuracy_score(c), title = title, labels = permutedims(labels))
    elseif type == "negative-predictive-value"; x = func(labels, negative_predictive_value(c), title = title, labels = permutedims(labels))
    elseif type == "false-negative-rate"; x = func(labels, false_negative_rate(c), title = title, labels = permutedims(labels))
    elseif type == "false-positive-rate"; x = func(labels, false_positive_rate(c), title = title, labels = permutedims(labels))
    elseif type == "false-discovery-rate"; x = func(labels, false_discovery_rate(c), title = title, labels = permutedims(labels))
    elseif type == "false-omission-rate"; x = func(labels, false_omission_rate(c), title = title, labels = permutedims(labels))
    elseif type == "f1-score"; x = func(labels, f1_score(c), title = title, labels = permutedims(labels))
    elseif type == "prevalence-threshold"; x = func(labels, prevalence_threshold(c), title = title, labels = permutedims(labels))
    elseif type == "threat-score"; x = func(labels, threat_score(c), title = title, labels = permutedims(labels))
    elseif type == "matthews-correlation-coeff"; x = func(labels, matthews_correlation_coeff(c), title = title, labels = permutedims(labels))
    elseif type == "fowlkes-mallows-index"; x = func(labels,fowlkes-mallows-index(c), title = title, labels = permutedims(labels))
    elseif type == "informedness"; x = func(labels, informedness(c), title = title, labels = permutedims(labels))
    elseif type == "markedness"; x = func(labels, markedness(c), title = title, labels = permutedims(labels))
    elseif type == "cohen-kappa-score"; x = func(labels, cohen_kappa_score(c), title = title, labels = permutedims(labels))
    elseif type == "hamming-loss"; x = func(labels, hamming_loss(c), title = title, labels = permutedims(labels))
    elseif type == "jaccard-score"; x = func(labels, jaccard_score(c), title = title, labels = permutedims(labels))
    end
    return x
end


"""
```visualize(c::confusion_matrix, <keywords>)```

Visualize the given properties of the confusion matrix as specified

## Keywords
**`mode`** : String, Array of String; represents properties given below of the confusion matrix, can be either an array containing different properties
or a single string.\n
    Supported Modes: \n
    - `matrix`\n
    - `condition-positive`\n
    - `condition-negative`\n
    - `predicted-positive\n
    - `predicted-negative`\n
    - `correctly-classified`\n
    - `incorrectly-classified`\n
    - `sensitivity-score`\n
    - `recall-score`\n
    - `specificity-score`\n
    - `precision-score`\n
    - `positive-predictive-value`\n
    - `accuracy-score`\n
    - `balanced-accuracy-score`\n
    - `negative-predictive-value`\n
    - `false-negative-rate`\n
    - `false-positive-rate`\n
    - `false-discovery-rate`\n
    - `false-omission-rate`\n
    - `f1-score`\n
    - `prevalence-threshold`\n
    - `threat-score`\n
    - `matthews-correlation-coeff`\n
    - `fowlkes-mallows-index`\n
    - `informedness`\n
    - `markedness`\n
    - `cohen-kappa-score`\n
    - `hamming-loss`\n
    - `jaccard-score`\n

**`seriestype`** : String, denotes which visualization function will be used; default: heatmap
    Supported visualization functions:
        - `heatmap`
        - `bar`
        - `histogram`
        - `scatter`
        - `line`

**`title`** : String, denotes the title that will displayed above the drawn plot, default: nothing

**`labels`** : Vector, denotes the labels that will be used for the plot. If equals to nothing, labels of the given confusion matrix will be used.

"""
function visualize(c::confusion_matrix; mode = "matrix", seriestype::String = "heatmap", title= nothing, labels = nothing)
    @assert seriestype in ["scatter", "heatmap", "line", "histogram", "bar"] "Unknown visualization format"
    labels = labels != nothing ? labels : convert(Array{typeof(c.Labels[1])}, c.Labels)
    if title == nothing; title = mode isa Array ? mode : String(Base.copymutable(mode)); end
    title = title isa Array ? title : [title]
    mode = mode isa Array ? mode : [mode]
    plt = []
    for i in 1:length(mode)
        @assert mode[i] in ["matrix", "condition-positive", "condition-negative", "predicted-positive","predicted-negative", "correctly-classified", "incorrectly-classified", "sensitivity-score", "recall-score", "specificity-score", "precision-score", "positive-predictive-value", "accuracy-score", "balanced-accuracy-score", "negative-predictive-value", "false-negative-rate", "false-positive-rate", "false-discovery-rate",
         "false-omission-rate", "f1-score", "prevalence-threshold", "threat-score", "matthews-correlation-coeff", "fowlkes-mallows-index",
         "informedness", "markedness", "cohen-kappa-score", "hamming-loss", "jaccard-score"] "Unknown visualization mode"
        if mode[i] != "matrix"; @assert seriestype in ["scatter", "line", "histogram", "bar"] "The given mode does not support this visualization format"; end
        x = nothing
        if mode[i] == "matrix"
            if seriestype == "histogram"; x = histogram(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "scatter"; x = scatter(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "line"; x = plot(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "bar"; x = bar(labels, c.matrix, labels = permutedims(labels), title = title[i])
            elseif seriestype == "heatmap"; x = heatmap(labels, labels, c.matrix, labels = permutedims(labels), title = title[i])
            end
        else
            if seriestype == "histogram"; x = _plot(c; func = histogram, type = mode[i], title = title[i], labels = labels)
            elseif seriestype == "scatter"; x = _plot(c; func = scatter, type = mode[i], title = title[i], labels = labels)
            elseif seriestype == "line"; x =  _plot(c; func = plot, type = mode[i], title = title[i], labels = labels)
            elseif seriestype == "bar"; x =  _plot(c; func = bar, type = mode[i], title = title[i], labels = labels)
            elseif seriestype == "heatmap"; x =  _plot(c; func = heatmap, type = mode[i], title = title[i], labels = labels)
            end
        end
        push!(plt, x)
    end
    plot(plt..., layout = (length(plt), 1))
end
