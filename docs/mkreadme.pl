#!/usr/bin/perl -w
use strict;
while(<>) {
    next if /Introduction-to-Knet-1/;
    s/^    (\s*- \[.+?\]\(tutorial.md#)/$1/;
    s/\]\(tutorial.md/](/g;
    s!\]\((\S+?)\.md#!](http://denizyuret.github.io/Knet.jl/latest/$1.html#!g;
    print;
}
