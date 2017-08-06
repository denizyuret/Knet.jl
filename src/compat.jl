# const A = B works in all julia versions as typealias replacement but
# const A{T} = B{T} only works after 0.6.
# This does the right thing without causing warnings
macro typealias6(t1,t2)
    if VERSION >= v"0.6-"
        esc(:(const $t1 = $t2))
    else
        Expr(:typealias, t1, t2)
    end
end
