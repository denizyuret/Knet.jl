#!/usr/bin/perl -w
use strict;
while(<>) {
    chomp;
    my $line = '';
    if (m!/([^./]+\.jl);.* line: (\d+)!) {
	my ($file, $lnum) = ("../src/$1",$2);
	if (-e $file) {
	    $line = `head -$lnum $file | tail -1`;
	    $line =~ s/^\s+//; $line =~ s/\s+$//;
	}
    }
    print("$_  $line\n");
}
