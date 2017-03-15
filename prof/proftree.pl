#!/usr/bin/perl -w
use strict;
while(<>) {
    chomp;
    my $line = '';
    if (m!/([^./]+\.jl):(\d+); !) {
	my ($file, $lnum) = ($1,$2);
	if (-e $file) {
	    $line = `head -$lnum $file | tail -1`;
	    $line =~ s/^\s+//; $line =~ s/\s+$//;
	    s/;.*/; $line/;
	}
    }
    print("$_\n");
}
