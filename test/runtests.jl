Pkg.add("ArgParse")
load_only=true
include(joinpath("..","examples","charlm.jl"))
CharLM.main("--gcheck 3")
