using DelimitedFiles, Statistics, Knet

"""

    housing([test]; [url, file])

Return (xtrn,ytrn,xtst,ytst) from the [UCI Boston
Housing](https://archive.ics.uci.edu/ml/machine-learning-databases/housing)
dataset The dataset has housing related information for 506
neighborhoods in Boston from 1978. Each neighborhood has 14
attributes, the goal is to use the first 13, such as average number of
rooms per house, or distance to employment centers, to predict the
14’th attribute: median dollar value of the houses.

`test=0` by default and `xtrn` (13,506) and `ytrn` (1,506) contain the
whole dataset. Otherwise data is shuffled and split into train and
test portions using the ratio given in `test`. xtrn and xtst are
always normalized by subtracting the mean and dividing into standard
deviation.

"""
function housing(test=0.0;
                 file=Knet.dir("data", "housing", "housing.data"),
                 url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
    if !isfile(file)
        isdir(dirname(file)) || mkpath(dirname(file))
        @info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    # @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14:14,:]
    x = (x .- mean(x,dims=2)) ./ std(x,dims=2) # Data normalization
    if test == 0
        xtrn = xtst = x
        ytrn = ytst = y
    else
        r = randperm(size(x,2))          # trn/tst split
        n = round(Int, (1-test) * size(x,2))
        xtrn=x[:,r[1:n]]
        ytrn=y[:,r[1:n]]
        xtst=x[:,r[n+1:end]]
        ytst=y[:,r[n+1:end]]
    end
    return (xtrn, ytrn, xtst, ytst)
end

nothing
