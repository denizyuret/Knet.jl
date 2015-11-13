DICT=cmudict.0.6_lg.w-p-gp
DATA=lhs.trn rhs.trn lhs.tst rhs.tst lhs.dict rhs.dict

model.jld: train.jl $DATA
	julia train.jl --data lhs.trn rhs.trn lhs.tst rhs.tst --dict lhs.dict rhs.dict --hidden 512 --gclip 5 --lr 2 --fbias 1 --seed 23 --drop1 0.8 --drop2 0.4 --epochs 30 --best $@

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
