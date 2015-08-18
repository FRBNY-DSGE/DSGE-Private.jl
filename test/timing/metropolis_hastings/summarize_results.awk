#!/bin/awk -f

# Taken from:
# http://www.grymoire.com/Unix/Scripts/average.awk


BEGIN {
# How many lines
	lines=0;
	total=0;
}
{
# this code is executed once for each line
# increase the number of files
	lines++;
# increase the total size, which is field #1
	total+=$4;
}
END {
# end, now output the total
	print lines " trials performed";
	print "total time is ", total;
	if (lines > 0 ) {
		print "average time per trial is ", total/lines;
	} else {
		print "average time per trial is 0";
	}
}