#!/usr/bin/env python2
#
#  average-me.py S Kondrat aka Valiska <valiska@gmail.com> 2013
#
#  Read the data file and coarse-grain the given columns within a given 'window'
#
#  Usage: average-me.py -f FILE{S} [ -k LINES{s} ] -d dX
#

from __future__ import print_function
import numpy as np
import pandas as pd

import os
import sys
import fileinput
import operator
import math
import string
import time
import datetime

import warnings

warnings.simplefilter("error")

def printf(str, *args):
    print(str % args, end='')

class MyData(object):
	def __init__(self, array = None, t0 = None, dt = None):

	#	if array is not None and not isinstance(array, MyData):
	#		self.append (array)

		if array is not None and isinstance(array, MyData) and t0 is not None and dt is not None:
	        	self = coarseGrain (array, t0, dt)

		else:
			self.length = 0;
			self.size = 0;
			self.data = None
	
	def append (self, array):

		if debug:
			printf ('Appending an array of length %i as %ith element\n', array.shape[1], self.size)
			print (array)

		if self.length == 0:
			self.length = array.shape[1]
			self.data=array.copy()

		elif self.length != array.shape[1]:
		# let's be strict
			printf ('self.length %i and the added array length %i are different\n', self.length, array.shape[1])
		    	sys.exit()
		else:
			self.data=np.concatenate((self.data,  array),  axis=0) # TODO: avoid ugly copying here
		self.size = self.size + array.shape[0];
		if debug:
			printf ('Appended an array of size %i summing to %i elements\n', array.shape[0], self.size)

	def sort (self):
		self.data.view(",".join([self.data.dtype.str] * self.data.shape[1])).sort(order='f0',axis=0) #nd inplace sort
		



# Command line parser
from optparse import OptionParser
#set up command-line options
parser = OptionParser()

parser.add_option("-f", "--files", help="comma-separated files with data.", metavar="FILE", type="string", dest="files")
parser.add_option("-F", "--files-file", help="file with the list of data files.", metavar="FILE", type="string", dest="File")
parser.add_option("-k", "--lines", help="comma-separated list of lines to process.", metavar="VAL", type="string", dest="lines")
parser.add_option("-w", "--window", help="coarse-grain window (applies to the first line in -k option).", metavar="VAL", type="float", dest="dt")
parser.add_option("-s", "--start", help="start avereging from this value of the first line from -k option.", metavar="VAL", type="float", dest="t0")

#
# Grab options
#
(options, args) = parser.parse_args()

# Comma separated files 
if options.files and options.File:
	printf ('The file names are missing (-f/--file or -F/--files-file)\n')
    	sys.exit()

files=[]
if options.files:
	files = options.files.split (',')

elif options.File:
	for line in fileinput.input(options.File):
		fs = line.split (' ')
		for f in fs:
			f1 = f.strip('\r\n')
			files.append(f1)


else:
	printf ('The file names are missing (-f/--file or -F/--files-file)\n')
    	sys.exit()

# Lines to process
lines = []
if options.lines:
	kstr = options.lines.split (',')
	for j in range(0, len(kstr), 1):
		lines.append (int (kstr[j]))

# Coarse graining window, dt
if options.dt:
	dt = options.dt
else:	
	printf ("The 'window' for coarse-graining not specified (-w/--window)\n")
    	sys.exit()

# Starting t0
if options.t0:
	t0 = options.t0
else:
	t0 = 0.0;

debug = False
debugRead = False

Data = MyData()
tstart=datetime.datetime.now()
for f in files:
	if debugRead:
		printf ('***************************\n')
		printf('file %s\n' , f)
	if os.path.isfile(f) and os.path.getsize(f) > 0:
		X=pd.io.parsers.read_csv(f, sep=' ',usecols=tuple(lines),  header=None,  index_col=False, dtype=np.float64,  skipinitialspace=True, comment='#')
	else:
		print("Warning: Skipping non-existing or empty file: %s" % (f),  file=sys.stderr)
	Data.append (X.values);
	X=None

# Save info lines to the file
t = datetime.datetime.now()
tdiff_load=t-tstart
print('# Reading files took %s\n' % str(tdiff_load),file=sys.stderr)

printf ('# Created by average-me.py on %s at %s\n', t.strftime("%d-%m-%Y"), t.strftime("%H:%M:%S %Z"))
printf ('# Files used:\n')

for f in files:
	printf ('#	%s\n', f)

printf ('# Starting from %f, using coarse-graining widnow %f\n', t0, dt)
printf ('# Columns for coarse-graining: %s\n', lines)

# Sort data in case we read unsorted data or different data files
Data.sort();

t2=datetime.datetime.now()
tdiff_sort=t2-t
print('#ozi: Sorting took %s\n' %str(tdiff_sort),file=sys.stderr)

### ozi: now calculating 
intervals=np.arange(t0, Data.data[-1, 0],  dt)
np.append(intervals, intervals[-1]+dt) # adding the last interval stop position 
interval_index=np.searchsorted(Data.data[:, 0], intervals)
format_string="  ".join(["%f"] * Data.length)
for ii  in range(1, len(interval_index)):
	print(format_string %  tuple(Data.data[interval_index[ii-1]:interval_index[ii], :].mean(axis=0)),  end='')
	printf(" %d\n",  interval_index[ii]-interval_index[ii-1] )
print(format_string %  tuple(Data.data[interval_index[-1]:, :].mean(axis=0)), end='')
printf(" %d\n",  Data.size-interval_index[-1] )
### end_ozi

t3=datetime.datetime.now()
tdiff_calc=t3-t2
print('#ozi: Calculating took %s\n' % str(tdiff_calc),file=sys.stderr)

