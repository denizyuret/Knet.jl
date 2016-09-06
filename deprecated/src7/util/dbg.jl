### Useful utilities

# @dbg: Uncomment this for debugging
# DBG=false; dbg()=DBG; dbg(b::Bool)=(global DBG=b); macro dbg(x) :(DBG && $(esc(x))) end
# @dbg: Uncomment this for production
macro dbg(x) nothing end        

# gpusync: Uncomment this for GPU profiling
#gpusync()=device_synchronize()
# gpusync: Uncomment this for production
gpusync()=nothing

# @date: Print date, expression; run and print elapsed time after execution
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
