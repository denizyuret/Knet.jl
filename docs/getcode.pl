#!/usr/bin/perl -w
use strict;

# Extract Julia code from rst files

my $code = 0;

while(<>) {
    if (/^\.\. doctest::/) {
	$code = 1;
    } elsif (/^\.\. testcode::/) {
	$code = 2;
    } elsif ($code == 1) {
	if (/^\s+julia>\s*(.+)/) {
	    print "$1\n";
	} elsif (/^\S/) {
	    $code = 0;
	}
    } elsif ($code == 2) {
	if (/^\S/) {
	    $code = 0;
	} else {
	    print;
	}
    }
}
