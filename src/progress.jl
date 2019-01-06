# Based on https://github.com/cloud-oak/Tqdm.jl by @cloud-oak under Mozilla Public License 2.0
# Modified for Knet by Deniz Yuret

using Printf

mutable struct Progress
    current::Int
    total::Int
    width::Int
    start_time::UInt
    print_time::UInt
    Progress(total=0, width=displaysize()[2])=new(0,total,width,time_ns(),0)
end

function progress!(t::Progress, fval=nothing; display_interval = 0.1)
    t.current += 1
    curr_time = time_ns()
    if t.current != t.total && curr_time < t.print_time + display_interval * 1e9; return; end
    t.print_time = curr_time
    seconds   = (curr_time - t.start_time) * 1e-9
    speed     = t.current / seconds
    ETA       = t.total > 0 ? (t.total-t.current) / speed : Inf
    # print(repeat("\r", t.width))
    print("\r")
    fval_string = fval != nothing ? @sprintf("%.2e  ", value(fval)) : ""
    percentage_string = t.total > 0 ? string(@sprintf("%.2f%%",t.current/t.total*100)) : ""
    status_string = string(t.current, (t.total > 0 ? "/$(t.total)" : ""),
                            " [", format_time(seconds), (ETA==Inf ? "" : "-" * format_time(ETA)),
                            ", ", @sprintf("%.2f/sec", speed),"]")

    width = t.width - length(fval_string) - length(percentage_string)-length(status_string) - 2

    print(fval_string)
    print(percentage_string)
    print("┣")

    if (t.total <= 0)
        offset = t.current % 10
        print(repeat(" ", offset))
        segments, remain = divrem(width - offset, 10)
        print(repeat("/         ", Int(segments)))
        print(repeat(" ", Int(remain)))
    else
        cellvalue = t.total / width
        full_cells, remain = divrem(t.current, cellvalue)
        full_cells = round(Int, full_cells)
        print(repeat("█", full_cells))
        if (full_cells < width)
	    part = floor(Int, 8 * remain / cellvalue)
	    print(EIGHTS[part])
            print(repeat(" ", width - full_cells - 1))
        end
    end
    print("┫ ")

    print(status_string)
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

