using Knet,AutoGrad
a = map(t->map(a->isa(a,Array)?KnetArray(a):a,t), AutoGrad.alltests());
for ai in a; println(ai); AutoGrad.runtests([ai]); end
