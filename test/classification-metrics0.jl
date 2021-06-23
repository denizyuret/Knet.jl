using Test
using KnetMetrics

@testset "classification-metrics" begin
    random_true = rand(2:6,1000)
    random_pred = rand(2:6,1000)
    random_labels = [2,3,4,5,6]
    random_confusion_matrix = confusion_matrix(random_true, random_pred, labels = random_labels)

    @testset "binary-confusion-matrix" begin
        y_true = [1,1,2,2]
        y_pred = [1,1,2,2]
        x = confusion_matrix(y_true, y_pred)
        @test x.matrix == [2 0; 0 2]
    end

    y_true = [3, 2, 4, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 4, 2]
    y_pred = [3, 2, 5, 3, 2, 4, 4, 3, 2, 3, 4, 2, 2, 3, 4, 2, 3, 3, 4, 4]
    labels = [2, 3, 4, 5]

    x = confusion_matrix(y_true, y_pred, labels = labels)

    @testset "multiclass-confusion-matrix" begin
        @test x.matrix == [4 0 1 0; 0 5 1 0; 0 1 4 1; 2 1 0 0]
    end

    @testset "class-confusion" begin
        @test class_confusion(x,ith_class =2) == [5 2; 1 12]
        @test class_confusion(random_confusion_matrix,ith_class =2) == class_confusion(random_confusion_matrix, class_name = 3)
    end

    @testset "condition-positive" begin
        @test condition_positive(x) == [5, 6, 6, 3]
        @test condition_positive(x, ith_class = 2) == 6
        @test condition_positive(random_confusion_matrix, class_name = 3) == condition_positive(random_confusion_matrix, ith_class = 2)
    end

    @testset "condition-negative" begin
        @test condition_negative(x) == [15, 14, 14, 17]
        @test condition_negative(x,ith_class = 3) == 14
        @test condition_negative(random_confusion_matrix, class_name = 3) == condition_negative(random_confusion_matrix, ith_class = 2)
    end

    @testset "sensitivity-recall-score" begin
        @test isapprox(sensitivity_score(x), [0.8, 0.8333333333333334, 0.6666666666666666, 0.0])
        @test sensitivity_score(random_confusion_matrix) == recall_score(random_confusion_matrix)
    end

    @testset "specificity-score" begin
        @test isapprox(specificity_score(x),[0.8666666666666667, 0.8571428571428571, 0.8571428571428571, 0.9411764705882353])
    end

    @testset "precision-score" begin
        @test isapprox(precision_score(x), [0.6666666666666666, 0.7142857142857143, 0.6666666666666666, 0.0])
    end

    @testset "accuracy-score" begin
        @test isapprox(accuracy_score(x), 0.16499999999999998)
        @test accuracy_score(x, normalize = false) ==  3.3
    end

    @testset "balanced-accuracy-score" begin
        @test isapprox(balanced_accuracy_score(x),[0.8333333333333334, 0.8452380952380952, 0.7619047619047619, 0.47058823529411764])
    end

    @testset "negative-predictive-value" begin
        @test isapprox(negative_predictive_value(x), [0.9285714285714286, 0.9230769230769231, 0.8571428571428571, 0.8421052631578947])
    end

    @testset "false-positive-rate" begin
        @test isapprox(false_positive_rate(x), [0.13333333333333333, 0.14285714285714285, 0.14285714285714285, 0.058823529411764705])
    end

    @testset "false-negative-rate" begin
        @test isapprox(false_negative_rate(x),  [0.2, 0.16666666666666666, 0.3333333333333333, 1.0])
    end

    @testset "false-discovery-rate" begin
        @test isapprox(false_discovery_rate(x) , [0.13333333333333333, 0.14285714285714285, 0.14285714285714285, 0.058823529411764705])
    end

    @testset "false-omission-rate" begin
        @test isapprox(false_omission_rate(x),[0.0714285714285714, 0.07692307692307687, 0.1428571428571429, 0.1578947368421053])
    end

    @testset "f1-score" begin
        @test isapprox(f1_score(x),[0.7272727272727273, 0.7692307692307693, 0.6666666666666666, 0.0])
    end

    @testset "prevalence-threshold" begin
        @test isapprox(prevalence_threshold(x),[0.24494897427831755, 0.22347381718647807, 0.4165977904505312, -4.12310562561766])
    end

    @testset "threat-score" begin
        @test isapprox(threat_score(x),[0.5714285714285714, 0.625, 0.5, 0.0])
    end

    @testset "matthews-correlation-coefficient" begin
        @test isapprox(matthews_correlation_coeff(x),[0.629940788348712, 0.6633880657639324, 0.5238095238095238, -0.09637388493048533])
    end

    @testset "fowlkes-mallows-index" begin
        @test isapprox( fowlkes_mallows_index(x),[0.7302967433402214, 0.7715167498104596, 0.6666666666666666, 0.0])
    end

    @testset "informedness" begin
        @test isapprox(informedness(x) ,[0.6666666666666667, 0.6904761904761905, 0.5238095238095237, -0.05882352941176472])
    end

    @testset "markedness" begin
        @test isapprox(markedness(x), [-0.38095238095238093, -0.34065934065934056, -0.4285714285714286, -1.0])
    end

    @testset "hamming-loss" begin
        @test hamming_loss(x) == 0.35
    end

    @testset "jaccard-score" begin
        @test jaccard_score(x) == [1.25, 1.5, 1.5, 0.75]
        @test isapprox(0.48148148148148145, jaccard_score(x, average = "micro"))
        @test isapprox(0.42410714285714285,  jaccard_score(x, average = "macro"))
    end

    @testset "cohen-kappa-score" begin
        @test isapprox(0.5155709342560555, cohen_kappa_score(x))
    end

end
