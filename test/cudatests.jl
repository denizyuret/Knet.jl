for i in ("1", "01", "10", "11", "12", "20", "21")
    f = "cuda$(i)test.jl"
    info(f)
    include("../src/$f")
end

# Sun Aug 28 18:50:55 EEST 2016
#
# julia> Pkg.test("Knet")
# INFO: Testing Knet
# sqrt
#   1.198008 seconds (306.34 k allocations: 6.399 MB)
#   1.488229 seconds (305.40 k allocations: 6.363 MB)
# exp
#   0.765560 seconds (298.98 k allocations: 6.088 MB, 0.36% gc time)
#   1.539859 seconds (298.98 k allocations: 6.088 MB)
# log
#   0.740429 seconds (298.98 k allocations: 6.088 MB, 0.32% gc time)
#   1.654432 seconds (298.98 k allocations: 6.088 MB)
# sin
#   0.782896 seconds (298.98 k allocations: 6.088 MB)
#   1.974086 seconds (298.98 k allocations: 6.088 MB, 0.10% gc time)
# tanh
#   0.949269 seconds (298.98 k allocations: 6.088 MB)
#   1.781415 seconds (298.98 k allocations: 6.088 MB)
# neg
#   0.783159 seconds (298.98 k allocations: 6.088 MB)
#   1.411126 seconds (298.98 k allocations: 6.088 MB)
# inv
#   0.733850 seconds (298.98 k allocations: 6.088 MB, 0.23% gc time)
#   1.469844 seconds (298.98 k allocations: 6.088 MB)
# relu
#   0.734503 seconds (298.98 k allocations: 6.088 MB)
#   1.428948 seconds (298.98 k allocations: 6.088 MB)
# sigm
#   0.782921 seconds (298.98 k allocations: 6.088 MB)
#   1.572042 seconds (298.98 k allocations: 6.088 MB, 0.11% gc time)
# abs
#   0.754620 seconds (298.98 k allocations: 6.088 MB)
#   1.433652 seconds (298.98 k allocations: 6.088 MB, 0.11% gc time)
# abs2
#   0.734661 seconds (298.98 k allocations: 6.088 MB)
#   1.411047 seconds (298.98 k allocations: 6.088 MB)
# add
#   0.742177 seconds (305.10 k allocations: 6.339 MB)
#   1.407227 seconds (305.10 k allocations: 6.339 MB)
# sub
#   0.735742 seconds (298.98 k allocations: 6.088 MB, 0.25% gc time)
#   1.399725 seconds (298.98 k allocations: 6.088 MB)
# mul
#   0.735076 seconds (298.98 k allocations: 6.088 MB)
#   1.399998 seconds (298.98 k allocations: 6.088 MB)
# div
#   0.740939 seconds (298.98 k allocations: 6.088 MB)
#   1.534365 seconds (298.98 k allocations: 6.088 MB, 0.11% gc time)
# pow
#   0.745232 seconds (298.98 k allocations: 6.088 MB)
#   2.975677 seconds (298.98 k allocations: 6.088 MB)
# max
#   0.744763 seconds (298.98 k allocations: 6.088 MB)
#   1.408032 seconds (298.98 k allocations: 6.088 MB)
# min
#   0.743997 seconds (298.98 k allocations: 6.088 MB, 0.22% gc time)
#   1.406233 seconds (298.98 k allocations: 6.088 MB)
# pow
#   0.772437 seconds (305.10 k allocations: 6.339 MB, 0.23% gc time)
#   2.863349 seconds (305.11 k allocations: 6.339 MB)
# pow
#   0.765591 seconds (298.98 k allocations: 6.088 MB)
#   2.856764 seconds (298.98 k allocations: 6.088 MB, 0.06% gc time)
# pow
#   0.766089 seconds (298.98 k allocations: 6.088 MB)
#   2.856661 seconds (298.98 k allocations: 6.088 MB)
# add
#   0.840574 seconds (305.31 k allocations: 6.350 MB)
#   2.068473 seconds (305.32 k allocations: 6.350 MB)
# sub
#   0.833583 seconds (298.98 k allocations: 6.088 MB, 0.30% gc time)
#   2.061315 seconds (298.98 k allocations: 6.088 MB)
# mul
#   0.833596 seconds (298.98 k allocations: 6.088 MB, 0.27% gc time)
#   2.060891 seconds (298.98 k allocations: 6.088 MB)
# div
#   0.837831 seconds (298.98 k allocations: 6.088 MB)
#   2.096731 seconds (298.98 k allocations: 6.088 MB, 0.09% gc time)
# pow
#   0.937980 seconds (298.98 k allocations: 6.088 MB)
#   3.113095 seconds (298.98 k allocations: 6.088 MB, 0.06% gc time)
# max
#   0.836112 seconds (298.98 k allocations: 6.088 MB)
#   2.060026 seconds (298.98 k allocations: 6.088 MB)
# min
#   0.836224 seconds (298.98 k allocations: 6.088 MB)
#   2.059889 seconds (298.98 k allocations: 6.088 MB)
# add
#   0.918067 seconds (310.71 k allocations: 6.592 MB)
#   1.105991 seconds (309.96 k allocations: 6.552 MB)
#   1.283519 seconds (298.98 k allocations: 6.088 MB)
#   1.520820 seconds (298.98 k allocations: 6.088 MB)
#   2.091047 seconds (310.71 k allocations: 6.590 MB, 0.12% gc time)
#   1.444449 seconds (309.95 k allocations: 6.552 MB)
#   1.479492 seconds (298.98 k allocations: 6.088 MB)
#   1.599586 seconds (298.98 k allocations: 6.088 MB)
# sub
#   0.897188 seconds (298.98 k allocations: 6.088 MB, 0.25% gc time)
#   1.091974 seconds (298.98 k allocations: 6.088 MB)
#   1.310290 seconds (298.98 k allocations: 6.088 MB, 0.16% gc time)
#   1.576266 seconds (298.98 k allocations: 6.088 MB)
#   2.078716 seconds (298.98 k allocations: 6.088 MB)
#   1.434182 seconds (298.98 k allocations: 6.088 MB, 0.14% gc time)
#   1.482647 seconds (298.98 k allocations: 6.088 MB)
#   1.600311 seconds (298.98 k allocations: 6.088 MB, 0.11% gc time)
# mul
#   0.899609 seconds (298.98 k allocations: 6.088 MB)
#   1.091918 seconds (298.98 k allocations: 6.088 MB)
#   1.310454 seconds (298.98 k allocations: 6.088 MB, 0.14% gc time)
#   1.573036 seconds (298.98 k allocations: 6.088 MB)
#   2.078040 seconds (298.98 k allocations: 6.088 MB)
#   1.433069 seconds (298.98 k allocations: 6.088 MB)
#   1.478995 seconds (298.98 k allocations: 6.088 MB)
#   1.599840 seconds (298.98 k allocations: 6.088 MB, 0.15% gc time)
# div
#   0.898756 seconds (298.98 k allocations: 6.088 MB)
#   1.076243 seconds (298.98 k allocations: 6.088 MB)
#   1.273588 seconds (298.98 k allocations: 6.088 MB)
#   1.535823 seconds (298.98 k allocations: 6.088 MB)
#   2.122483 seconds (298.98 k allocations: 6.088 MB, 0.09% gc time)
#   1.561032 seconds (298.98 k allocations: 6.088 MB)
#   1.649356 seconds (298.98 k allocations: 6.088 MB)
#   1.834667 seconds (298.98 k allocations: 6.088 MB)
# pow
#   0.977069 seconds (298.98 k allocations: 6.088 MB)
#   1.097480 seconds (298.98 k allocations: 6.088 MB, 0.15% gc time)
#   1.308324 seconds (298.98 k allocations: 6.088 MB)
#   1.570295 seconds (298.98 k allocations: 6.088 MB)
#   3.174292 seconds (298.98 k allocations: 6.088 MB, 0.06% gc time)
#   3.131706 seconds (298.98 k allocations: 6.088 MB)
#   3.351452 seconds (298.98 k allocations: 6.088 MB, 0.05% gc time)
#   3.524023 seconds (298.98 k allocations: 6.088 MB)
# max
#   0.904631 seconds (298.98 k allocations: 6.088 MB)
#   1.078452 seconds (298.98 k allocations: 6.088 MB)
#   1.278753 seconds (298.98 k allocations: 6.088 MB)
#   1.542877 seconds (298.98 k allocations: 6.088 MB, 0.11% gc time)
#   2.083428 seconds (298.98 k allocations: 6.088 MB)
#   1.434031 seconds (298.98 k allocations: 6.088 MB)
#   1.477127 seconds (298.98 k allocations: 6.088 MB)
#   1.599171 seconds (298.98 k allocations: 6.088 MB)
# min
#   0.904480 seconds (298.98 k allocations: 6.088 MB, 0.18% gc time)
#   1.078259 seconds (298.98 k allocations: 6.088 MB)
#   1.278503 seconds (298.98 k allocations: 6.088 MB)
#   1.542717 seconds (298.98 k allocations: 6.088 MB, 0.12% gc time)
#   2.083427 seconds (298.98 k allocations: 6.088 MB)
#   1.434334 seconds (298.98 k allocations: 6.088 MB, 0.12% gc time)
#   1.478011 seconds (298.98 k allocations: 6.088 MB)
#   1.599201 seconds (298.98 k allocations: 6.088 MB)
# add
#   2.964730 seconds (404.17 k allocations: 7.825 MB)
#   3.059696 seconds (404.17 k allocations: 7.825 MB)
# add
#   2.958643 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
#   3.053202 seconds (398.98 k allocations: 7.614 MB)
# mul
#   2.956758 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
#   3.041163 seconds (398.98 k allocations: 7.614 MB)
# mul
#   2.954647 seconds (398.98 k allocations: 7.614 MB)
#   3.045259 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
# max
#   2.989635 seconds (398.98 k allocations: 7.614 MB)
#   3.085604 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
# max
#   2.981864 seconds (398.98 k allocations: 7.614 MB)
#   3.084174 seconds (398.98 k allocations: 7.614 MB)
# min
#   2.985036 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
#   3.081865 seconds (398.98 k allocations: 7.614 MB)
# min
#   2.989373 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
#   3.082634 seconds (398.98 k allocations: 7.614 MB)
# sum1
#   2.953099 seconds (398.98 k allocations: 7.614 MB)
#   3.045103 seconds (398.98 k allocations: 7.614 MB, 0.06% gc time)
# sum1
#   2.950990 seconds (398.98 k allocations: 7.614 MB)
#   3.051839 seconds (398.98 k allocations: 7.614 MB)
# sum2
#   2.952941 seconds (398.98 k allocations: 7.614 MB, 0.07% gc time)
#   3.045060 seconds (398.98 k allocations: 7.614 MB)
# sum2
#   2.960478 seconds (398.98 k allocations: 7.614 MB, 0.06% gc time)
#   3.046865 seconds (398.98 k allocations: 7.614 MB)
# nnz
#   2.977525 seconds (398.98 k allocations: 7.614 MB)
#   3.085898 seconds (398.98 k allocations: 7.614 MB, 0.06% gc time)
# nnz
#   2.984819 seconds (398.98 k allocations: 7.614 MB)
#   3.083199 seconds (398.98 k allocations: 7.614 MB)
# INFO: Knet tests passed
