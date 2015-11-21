type Input <: Op; Input(;o...)=new(); end

"@knet function input() fetches the next network input."
input(y)=(Input(),y)

ninputs(::Input)=0
overwrites(::Input)=false
back_reads_x(::Input)=false
back_reads_y(::Input)=false
infersize(::Input,y)=tuple(y)
back(::Input,y...;o...)=nothing
