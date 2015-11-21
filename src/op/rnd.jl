type Rnd <: Op; rgen; testrgen; Rnd(;rgen=Uniform(0,1),testrgen=nothing, o...)=new(rgen,testrgen); end
rnd(y; rgen=Uniform(0,1), testrgen=nothing)=(Rnd(rgen, testrgen), y)
ninputs(::Rnd)=0
overwrites(::Rnd)=false
back_reads_x(::Rnd)=false
back_reads_y(::Rnd)=false
infersize(::Rnd,y)=tuple(y)
forw(r::Rnd, y; trn=false, o...)=rgen!(trn || r.testrgen==nothing ? r.rgen : r.testrgen, y)
back(r::Rnd, dy; o...)=error("rnd has no backward pass")
