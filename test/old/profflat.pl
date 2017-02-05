#!/usr/bin/perl -w
# For flat format
use strict;
while(<>) {
    chomp;
    my ($time,$file,$func,$lnum) = split;
    my $line = '';
    if (-e $file) {
	$line = `head -$lnum $file | tail -1`;
	$line =~ s/^\s+//; $line =~ s/\s+$//;
    }
    print("$_  $line\n");
}
