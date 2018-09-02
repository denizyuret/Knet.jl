using Knet
@show hidden = 100
@show winit = Gaussian(0,.01)
@show lr = 1.0
@show gclip = 5.0
@show batchsize = 5
maxnorm = zeros(2)
losscnt = zeros(2)

sgen1=SketchEngine(`xzcat sdfkl32KCsd_enTenTen12.vert.xz`; dict="tenten10k.dict", column=1)
sgen2=SketchEngine(`xzcat sdfkl32KCsd_enTenTen12.vert.xz`; dict="tentenpos58.dict", column=2)
data=TagData(sgen1,sgen2; batchsize=batchsize)
@show maxtoken(data,1)
@show vocab = maxtoken(data,2)

fnet = Net(lstm; out=hidden)
bnet = Net(lstm; out=hidden)
pnet = Net(add2; out=vocab, f=soft, ninputs=2)

model = Tagger(fnet, bnet, pnet)
setopt!(model; lr=lr)
train(model, data, softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt, lossreport=1000)

:ok

