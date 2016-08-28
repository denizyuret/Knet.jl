type Input <: Op; Input(;o...)=new(); end
Kenv.kdef(:input,Input)
ninputs(::Input)=0
canoverwrite(::Input)=false
back_reads_x(::Input)=false
back_reads_y(::Input)=false
infersize(::Input,y)=tuple(y)
back(::Input,y...;o...)=nothing
