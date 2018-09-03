#!/usr/bin/perl -w
use strict;
my $state = 0;
while(<>) {
    if ($state == 0) {
	if (/"""/) {
	    print; $state = 1;
	}
    } elsif ($state == 1) {
	print;
	if (/"""/) {
	    $state = 2;
	}
    } elsif ($state == 2) {
	print;
	print "end\n";
	last;
    }
}
