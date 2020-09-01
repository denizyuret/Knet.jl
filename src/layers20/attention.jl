
abstract type AbstractAttention <: Layer end

struct MultiheadAttention <: AbstractAttention
    head::Int
    future::Bool
    iqproj::Dense
    ikproj::Dense
    ivproj::Dense
    oproj::Dense
    drop::Dropout
end

"""
    MultiheadAttention(head::Int, is::Int, hs::Int, os::Int; future::Bool=true, pdrop = 0.1)
Multihead dot product Attention Layer, `head` is the number of head, `is` is the input size, `hs` is the hidden size of input projection layer of each head,
`os` is the output size. When `future` is `false`, the k-th token can't see tokens at > k. `pdrop` is the dropout rate.
"""
MultiheadAttention(head::Int,
                   is::Int,
                   hs::Int,
                   os::Int;
                   future::Bool=true, pdrop = 0.1) = MultiheadAttention(head,
                                                                        future,
                                                                        Dense(input=is,output=hs*head),
                                                                        Dense(input=is,output=hs*head),
                                                                        Dense(input=is,output=hs*head),
                                                                        Dense(input=hs*head, output=os),
                                                                        Dropout(pdrop),
                                                                        )


function Base.show(io::IO, mh::MultiheadAttention)
    hs = div(size(mh.iqproj)[1], mh.head)
    is = size(mh.iqproj)[end]
    os = size(mh.oproj)[1]

    print(io, "MultiheadAttention(")
    print(io, "head=$(mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(is)=>$(os)")

    print(io, ", dropout=$(mh.drop.p))")
end

function (mh::MultiheadAttention)(query,
                                  key,
                                  value;
                                  mask=nothing)
    qs = size(query)
    ks = size(key)
    vs = size(value)
    if length(qs) == 3
        #size(ipq) == (h, q_seq_len, batch)
        ipq = mh.iqproj(query)
        ipk = mh.ikproj(key)
        ipv = mh.ivproj(value)

        h = size(ipq, 1)
        hs = div(h, mh.head)

        #size(ipq) == (hs, q_seq_len, head, batch)
        ipq = permutedims(reshape(ipq, hs, mh.head, qs[2], qs[3]), [1, 3, 2, 4])
        ipk = permutedims(reshape(ipk, hs, mh.head, ks[2], ks[3]), [1, 3, 2, 4])
        ipv = permutedims(reshape(ipv, hs, mh.head, vs[2], vs[3]), [1, 3, 2, 4])

        #size(ipq) == (hs, q_seq_len, head * batch)
        ipq = reshape(ipq, hs, qs[2], :)
        ipk = reshape(ipk, hs, ks[2], :)
        ipv = reshape(ipv, hs, vs[2], :)

        atten = attention(ipq,ipk,ipv;
                          mask=mask,
                          future=mh.future,
                          dropout=mh.drop)

        atten = permutedims(reshape(atten, hs, qs[2], mh.head, qs[3]), [1, 3, 2, 4]) #size(atten) == (hs, head, ql, b)
        atten = reshape(atten, h, qs[2], qs[3]) #size(atten) == (h, ql, b)

        return  mh.oproj(atten)
    else
      ipq = mh.iqproj(query)
      ipk = mh.ikproj(key)
      ipv = mh.ivproj(value)

      h = size(ipq)[1] #h == hs * head
      hs = div(h, mh.head)

      #size(hq) == (hs, seq_len, head)
      hq = permutedims(reshape(ipq, hs, mh.head, :), [1, 3, 2])
      hk = permutedims(reshape(ipk, hs, mh.head, :), [1, 3, 2])
      hv = permutedims(reshape(ipv, hs, mh.head, :), [1, 3, 2])

      atten = attention(hq, hk, hv;
                        mask=mask,
                        future=mh.future,
                        dropout=mh.drop)

      # size(atten) == (head*hs, seq_len)
      atten = reshape(permutedims(atten, [1, 3, 2]), h, :)

      return mh.oproj(atten)
    end
end

function attention(query,
                   key,
                   value;
                   mask=nothing,
                   future::Bool = false,
                   dropout=nothing)
    T = eltype(query)
    dk = size(key, 1)
    score = bmm(key, query; transA=true)
    score = score ./ convert(T , sqrt(dk))

    s = size(score)

    if mask !== nothing
        #weird issue on @. mask = (1 - mask) * -1e9 which casue mask to be -Inf
        mask = (T(1.0) .- mask) .* T(-1e9)
        ms = size(mask)
        #score = score .+ mask; use broadcast instead of repeat mask for head
        score = reshape(reshape(score, s[1:end-1]..., :, ms[end]) .+ reshape(mask, ms[1:end-1]..., 1, ms[end]), s)
    end

    if !future
        #without ... will cause data move back to cpu
        fmask = convert(arrtype,tril!(fill!(Matrix{T}(undef,s[1:end-1]...),T(-1e9)),-1))
        #fmask = tril!(fill!(similar(score, s[1:end-1]...), convert(T, -1e9)), -1)
        score = score .+ fmask
    end

    score = softmax(score;dims=1)
    dropout !== nothing && (score = dropout(score))
    bmm(value, score) #size(return) == (dims, q_seq_len, batch)
end
