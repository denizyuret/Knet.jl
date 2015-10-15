type Input <: Op; end
input(y)=(Input(),y)
ninputs(::Input)=0
overwrites(::Input)=false
back_reads_x(::Input)=false
back_reads_y(::Input)=false
infersize(::Input)=nothing
back(::Input,y...;o...)=nothing
