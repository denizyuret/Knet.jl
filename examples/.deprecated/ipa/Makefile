DICT=cmudict.0.6_lg.w-p-gp
DATA=lhs.trn rhs.trn lhs.tst rhs.tst lhs.dict rhs.dict

pred.out: predict.jl lhs.tst lhs.dict rhs.dict
	cat lhs.tst | julia predict.jl --model model.jld --dict lhs.dict rhs.dict > $@

model.jld: train.jl ${DATA}
	julia train.jl --data lhs.trn rhs.trn lhs.tst rhs.tst --dict lhs.dict rhs.dict --best $@

lhs.trn: ${DICT}
	cat $< | awk 'NR%10>0' | cut -f1 | perl -lne 'print join(" ", split(""))' > $@

rhs.trn: ${DICT}
	cat $< | awk 'NR%10>0' | cut -f2 > $@

lhs.tst: ${DICT}
	cat $< | awk 'NR%10==0' | cut -f1 | perl -lne 'print join(" ", split(""))' > $@

rhs.tst: ${DICT}
	cat $< | awk 'NR%10==0' | cut -f2 > $@

lhs.dict: ${DICT}
	cat $< | cut -f1 | perl -lne 'print join("\n", split(""))' | sort -u > $@

rhs.dict: ${DICT}
	cat $< | cut -f2 | perl -lne 'print join("\n", split)' | sort -u > $@

