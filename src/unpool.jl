"""
TODO: Implement the following optional arguments
padding=0
stride=window ???
"""
#Commented out section -> general case
function updims{T,N}(x::KnetArray{T,N}; window=2)
    if !isa(window,Number) error("Window size must be a number!") end
    #if !isa(padding,Number) error("Padding size must be a number!") end
    #if stride != window error("Stride must be equal to window!") end

    ntuple(N) do i
        if i < N-1
            (size(x,i)-1)*window + window
            #(size(x,i)-1)*stride + window - 2*padding
        else
            size(x,i)
        end
    end
end

"""
Simple unpooling (=upsampling) as in:
https://swarbrickjones.wordpress.com/2015/04/29/convolutional-autoencoders-in-pythontheanolasagne/
Example input-output
x (2x2)
6   14
8   16
unpool (2x2 window)
6   6   14  14
6   6   14  14
8   8   16  16
8   8   16  16
"""
function unpool(x; window=2)
    y = similar(x,updims(x; window=window))
    Knet.poolx(y,x,x*window^2; window=window,mode=1)#where the did window^2 come from ?
end