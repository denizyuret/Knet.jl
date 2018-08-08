pushfirst!(LOAD_PATH, joinpath(dirname(@__FILE__), "../src"))
pushfirst!(LOAD_PATH, joinpath(dirname(@__FILE__), "../.."))
using Pkg
# Pkg.build("Knet") rene
using Knet, Test, Random, GC
