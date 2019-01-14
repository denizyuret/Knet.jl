# Based on https://github.com/cloud-oak/Tqdm.jl by @cloud-oak under Mozilla Public License 2.0
# Modified for Knet by Deniz Yuret

using Printf
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds

mutable struct Progress{I}
    iter::I
    current::Int
    nprint::Int
    start_time::UInt
    print_time::UInt
    print_interval::UInt
    width::Int
    alpha::Float64
    avg::Float64
end

progress(iter::I; width=max(64,displaysize()[2]), alpha=0.001, interval=0.1) where {I} =
    Progress{I}(iter,0,0,time_ns(),0,Int(1e9*interval),width,alpha,Inf)

progress(i::Int; o...)=progress(1:n; o...)
progress!(x...; o...)=(for x in progress(x...; o...) end)

length(p::Progress) = length(p.iter)
size(p::Progress) = size(p.iter)
eltype(p::Progress) = eltype(p.iter)
IteratorSize(::Type{Progress{I}}) where {I} = IteratorSize(I)
IteratorEltype(::Type{Progress{I}}) where {I} = IteratorEltype(I)

@propagate_inbounds function iterate(p::Progress, s...)
    next = iterate(p.iter, s...)
    if next !== nothing
        p.current += 1
        (x, s) = next
        if p.alpha > 0 && x isa Number
            p.avg = (p.avg === Inf ? value(x) : p.alpha * value(x) + (1-p.alpha) * p.avg)
        end
    end
    display_progress(p, next === nothing)
    return next
end

function display_progress(p::Progress, last=false)
    curr_time = time_ns()
    if !last && (curr_time < p.print_time + p.print_interval)
        return
    end
    p.print_time = curr_time
    p.nprint += 1
    seconds   = (curr_time - p.start_time) * 1e-9
    speed     = p.current / seconds
    

    fval_string = isfinite(p.avg) ? @sprintf("%.2e  ", p.avg) : ""

    if haslength(p)
        ETA = (length(p) - p.current) / speed
        percentage_string = string(@sprintf("%.2f%%",p.current/length(p)*100))
        status_string = string(p.current, "/", length(p),
                               " [", format_time(seconds), "-", format_time(ETA),
                               ", ", @sprintf("%.2f/s", speed),"]")

    else
        ETA = Inf
        percentage_string = ""
        status_string = string(p.current, # "/", length(p),
                               " [", format_time(seconds), # "-", format_time(ETA),
                               ", ", @sprintf("%.2f/s", speed),"]")
    end

    print("\r")
    print(fval_string)
    print(percentage_string)
    print("┣")

    width = p.width - length(fval_string) - length(percentage_string) - length(status_string) - 2

    if (haslength(p))
        cellvalue = length(p) / width
        full_cells, remain = divrem(p.current, cellvalue)
        full_cells = round(Int, full_cells)
        print(repeat("█", full_cells))
        if (full_cells < width)
	    part = floor(Int, 8 * remain / cellvalue)
	    print(EIGHTS[part])
            print(repeat(" ", width - full_cells - 1))
        end
    else
        offset = p.nprint % 8
        print(repeat(" ", offset))
        print("/")
        segments, remain = divrem(width - offset - 1, 8)
        print(repeat("       /", Int(segments)))
        print(repeat(" ", Int(remain)))
    end
    print("┫ ")

    print(status_string)
    last && println()
end

function format_time(seconds)
    if seconds != Inf
        mins,s  = divrem(round(Int,seconds), 60)
        h, m    = divrem(mins, 60)
    else
        h=0;m=Inf;s=Inf
    end
    if h!=0
         return @sprintf("%02d:%02d:%02d",h,m,s)
    else
         return @sprintf("%02d:%02d",m,s)
    end
end

EIGHTS = Dict(0 => ' ',
	      1 => '▏',
	      2 => '▎',
	      3 => '▍',
	      4 => '▌',
	      5 => '▋',
	      6 => '▊',
	      7 => '▉',
	      8 => '█')
