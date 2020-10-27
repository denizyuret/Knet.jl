using Plots


"""function learning_curve(model; epochs = 10, suggestion = false)

end"""


function confusion_table(c::confusion_matrix, labels; update_labels = false)
    @assert size(c.Labels) == size(labels)
    if update_labels; c.Labels = labels; end
    converted_labels = []
    for i in 1:size(X.Labels,1)
        push!(converted_labels, string(c.Labels[i]))
    end
    heatmap(converted_labels, converted_labels, c.matrix, levels = length(converted_labels))
end

confusion_table(c::confusion_matrix) = confusion_table(c, c.Labels)
