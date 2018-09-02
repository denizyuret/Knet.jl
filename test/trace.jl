using Cassette
macro trace(f,ex)
    Cx = gensym("C")
    Fx = gensym("F")
    quote
        Cassette.@context $Cx
        Cassette.prehook(::$Cx, ::typeof($f), args...) = println(($f,args...))
        Cassette.posthook(::$Cx, output, ::typeof($f), args...) = println(($f,args...,"=>",output))
        $Fx() = $ex
        Cassette.overdub($Cx(), $Fx)
    end
end
