using Plots


"""function learning_curve(model; epochs = 10, suggestion = false)

end"""


function confusion_table(c::confusion_matrix, labels, norm = false)
    @assert size(c.Labels) == size(labels) "Size of the given labels does not match the data"
    converted_labels = []
    for i in 1:size(X.Labels,1)
        push!(converted_labels, string(c.Labels[i]))
    end
    if norm == true
        heatmap(converted_labels, converted_labels, normalize(c.matrix), c = :dense)
    else
        heatmap(converted_labels, converted_labels, c.matrix, c = :dense)
    end
end

confusion_table(c::confusion_matrix, norm = false) = confusion_table(c, c.Labels, norm)
