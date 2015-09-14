function myequal(a,b)
    println(typeof(a)==typeof(b) ? (:type,true) : (:type,false))
    for n in union(fieldnames(a), fieldnames(b))
        if isdefined(a,n) && isdefined(b,n)
            println(isequal(a.(n), b.(n)) ? (n,true) : (n,false))
        elseif isdefined(a,n) || isdefined(b,n)
            println((n,false))
        else
            println((n,:undef))
        end
    end
end
