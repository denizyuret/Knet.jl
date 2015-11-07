#!/usr/bin/perl -w
use strict;
use Getopt::Std;
use Data::Dumper;
our $opt_l = 100;
getopt('l');
my $sent=[];
my @data=();
my $nsent = 0;
open(FP, "xzcat sdfkl32KCsd_enTenTen12.vert.xz|") or die $!;
while(<FP>) {
    if (/^<\/s>/) {
	if (scalar(@$sent) <= $opt_l) {
	    push @data, $sent;
	    $nsent++;
	    # die Dumper(\@data) if scalar(@data)==5;
	    if (scalar(@data) >= 1024) {
		@data = sort { scalar(@$a) <=> scalar(@$b) } @data;
		for my $s (@data) {
		    print join(' ', @$s)."\n";
		}
		@data = ();
		# last if $nsent >= 4096;
	    }
	}
	$sent = [];
    } elsif (/^\<[^\t]/) {
	next;
    } elsif (/^\S+/) {
	push @$sent, $&;
    }
}
close(FP)
