using Knet
using Plots
using LinearAlgebra
using Base


include("model_utilities.jl")
include("math_utilities.jl")
include("visualization.utilities.jl")

export
        confusion_matrix,
        create_confusion_matrix,
        PCA,
        confusion_table,
        condition_positive,
        condition_negative,
        predicted_positive
        predicted_negative,
        correctly_classified,
        incorrectly_classified,
        sensitivity,
        specificity,
        precision,
        accuracy,
        balanced_accuracy,
        negative_predictive_value,
        false_negative_rate,
        false_positive_rate,
        false_discovery_rate,
        false_omission_rate,
        f1_score
