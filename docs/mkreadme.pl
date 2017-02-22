#!/usr/bin/perl -w
use strict;
while(<>) {
    s/\]\(tutorial.md/](/g;
    s!\]\((\S+?)\.md#!](http://denizyuret.github.io/Knet.jl/latest/$1.html#!g;
    next if /Introduction-to-Knet-1/;
    print;
}
