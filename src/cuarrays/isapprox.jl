Base.isapprox(a::Array,b::CuArray;o...)=isapprox(a,Array(b);o...)
Base.isapprox(a::CuArray,b::Array;o...)=isapprox(Array(a),b;o...)
