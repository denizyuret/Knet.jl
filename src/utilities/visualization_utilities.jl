using Plots


"""function learning_curve(model; epochs = 10, suggestion = false)

end"""


function confusion_table(c::confusion_matrix, norm = false)
    converted_labels = []
    for i in c.Labels
        push!(converted_labels, string(i))
    end
    if norm == true
        heatmap(converted_labels, converted_labels, normalize(c.matrix), c = :dense)
    else
        heatmap(converted_labels, converted_labels, c.matrix, c = :dense)
    end
end
