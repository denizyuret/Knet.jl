using Knet, Base.Test

# TODO: figure out no KnetArray in cpu problem
# TODO: figure out and fix the warnings people get, check yurdakul's pull request

@testset "gpu" begin
    if gpu() >= 0

    end
end

nothing
