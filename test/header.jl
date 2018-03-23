using Compat
pushfirst!(LOAD_PATH, joinpath(dirname(@__FILE__), "../src"))
pushfirst!(LOAD_PATH, joinpath(dirname(@__FILE__), "../.."))
using Compat.Test, Knet, Compat.GC
