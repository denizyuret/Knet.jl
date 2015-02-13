function m = maxdiff(f1, f2)
a1 = h5read(f1, '/data');
a2 = h5read(f2, '/data');
d = a1 - a2;
m = max(abs(d(:)));
fprintf(2, 'maxdiff=%g\n', m);
end
