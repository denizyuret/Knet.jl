function gaussian(a...; mean=0.0, std=0.01)
	return randn(a...) * std + mean;
end

function xavier(a...)
    w = rand(a...)
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w))
        fanin = div(length(w), fanout)
    end
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end