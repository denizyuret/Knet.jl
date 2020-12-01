module Metrics

import Base: show, length
import LinearAlgebra: normalize
import Plots: heatmap

include("classification_metrics.jl")

export confusion_matrix, class_confusion, visualize, classification_report, condition_positive, condition_negative, predicted_positive, predicted_negative, correctly_classified, incorrectly_classified, sensitivity, specificity, precision, accuracy, balanced_accuracy, negative_predictive_value, false_negative_rate, false_positive_rate, false_discovery_rate, false_omission_rate, f1_score, prevalence_threshold, threat_score, fowlkes_mallows_index, informedness, matthews_correlation_coeff, markedness, cohen_kappa_score, jaccard_score, zero_one_loss, fbeta_score, hamming_loss, hinge_loss

end
