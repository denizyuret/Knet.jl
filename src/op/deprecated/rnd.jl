type Rnd <: Op; rgen; Rnd(;rgen=Uniform(0,1), o...)=new(rgen); end
Kenv.kdef(:rnd,Rnd)
ninputs(::Rnd)=0
canoverwrite(::Rnd)=false
back_reads_x(::Rnd)=false
back_reads_y(::Rnd)=false
infersize(::Rnd,y)=tuple(y)
forw(r::Rnd, y; o...)=rgen!(r.rgen, y)
back(r::Rnd, dy; o...)=error("rnd has no backward pass")
