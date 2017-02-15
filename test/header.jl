if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    # Don't know how to make this conditional on Julia version in REQUIRE
    Pkg.installed("BaseTestNext") == nothing && Pkg.add("BaseTestNext")
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet

