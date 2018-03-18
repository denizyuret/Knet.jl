using KUnet
KUnet.atype(Array)
KUnet.ftype(Float64)
nc = 3
nd = 5
x = nothing
z = nothing
net = [Mmul(zeros(nc,nd)),PercLoss()]
for i=1:2
    x = floor(10*rand(nd,1))
    z = zeros(nc,1); z[rand(1:nc)]=1;
    @show x'
    @show z'
    @show forw(net, x)'
    @show back(net, z)
    update!(net)
    @show net[1].w.data
    @show net[1].w.diff
end
