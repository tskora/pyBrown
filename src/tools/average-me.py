#!/usr/bin/env python2
#
#  average-me.py S Kondrat aka Valiska <valiska@gmail.com> 2013
#
#  Read the data file and coarse-grain the given columns within a given 'window'
#
#  Usage: average-me.py -f FILE{S} [ -k LINES{s} ] -d dX
#

from __future__ import print_function

import os
import sys
import fileinput
import operator
import math
import string
import time
import datetime

def printf(str, *args):
    print(str % args, end='')

def MyFun (array):
	return array[0];

class MyData(object):
	def __init__(self, array = None, t0 = None, dt = None):

	#	if array is not None and not isinstance(array, MyData):
	#		self.append (array)

		if array is not None and isinstance(array, MyData) and t0 is not None and dt is not None:
	        	self = coarseGrain (array, t0, dt)

		else:
			self.length = 0;
			self.size = 0;
			self.data = []
	
	def append (self, array):

		if debug:
			printf ('Appending an array of length %i as %ith element\n', len (array), self.size)
			print (array)

		if self.length == 0:
		       	self.length = len(array)

		elif self.length != len (array):
		# let's be strick
			printf ('self.length %i and the added array length %i are different\n', self.length, len (array))
		    	sys.exit()
	
		self.data.append (array);
		self.size = self.size + 1;

	def sort (self):
		self.data.sort(key=MyFun);

############################################################
###  BUG: For unknown reasons this does not work 
###  as a function/copy constructor
###
###  the elements of the CoarseGrained.data array are all
###  replaced by the last X array (see the code below)
############################################################
def coarseGrain (orig, t0, dt):
	orig.sort();
	CoarseGrained = MyData ();
	N = 0; 
	X = [0.0] * orig.length

	for j in range (0, orig.size, 1):
#		printf ('t = %g, interval (%f, %f)\n', orig.data[j][0], t0, t0 + dt)

		if orig.data[j][0] >= t0 and orig.data[j][0] <= t0 + dt:
			N = N + 1
			for i in range (0, orig.length, 1):
				X[i] = X[i] + orig.data[j][i]
			#printf ('N=%i:\n', N)

		else:
			if N != 0:
				for i in range (0, orig.length, 1):
					X[i] = X[i] / N
					#printf ('X%i=%f\n', i, X[i])
				#print (X);
				CoarseGrained.append (X);

			# reset now
			for i in range (0, orig.length, 1):
				X[i] = orig.data[j][i]
			t0 = t0 + dt; N = 1

		if N != 0:
			for i in range (0, orig.length, 1):
				X[i] = X[i] / N
				#printf ('X%i=%f\n', i, X[i])
			#print (X);
			self.append (X);

	return CoarseGrained;

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
#	for f in files:
#		printf ("f='%s'\n", f)

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
#debug = True
#debugRead = True
debugRead = False

Data = MyData()

for f in files:
	if debugRead:
		printf ('***************************\n')
		printf ('file %s\n', f)

	for line in fileinput.input(f):
		if debugRead:
			printf ('line: %s\n', line)

		tokens = line.split()
		if len (tokens) == 0 or tokens[0] == "#":
			continue

		if debugRead:
			printf ('length of lines: %i, tokens: %i\n', len (lines), len (tokens))

		if len(lines) == 0:
			for i in range (0, len(tokens), 1):
				lines.append (i);
		if debugRead:
			printf ('line: %s\n', line)

		X = [];
		for i in range (0, len(lines), 1):
			if len(tokens) < lines[i]: 
				printf ('Error reading file %s, number of collumnds %i, expected larger than %i.\n', f, len(tokens), lines[i])
			    	sys.exit()
			else:
				val = float(tokens[lines[i]]);
				X.append (val)
		Data.append (X);

# Save info lines to the file
t = datetime.datetime.now()
printf ('# Created by average-me.py on %s at %s\n', t.strftime("%d-%m-%Y"), t.strftime("%H:%M:%S %Z"))
printf ('# Files used:\n')

for f in files:
	printf ('#	%s\n', f)

printf ('# Starting from %f, using coarse-graining widnow %f\n', t0, dt)
printf ('# Columns for coarse-graining: %s\n', lines)

# Coarse grain the read data
# FIXME: the function below, and its copy-constructor version
# do not work, see BUG note above in the function
#
# So we simply copy->paste the code below and print to stdout
#
# CoarseGrained = coarseGrain (Data, t0, dt)
# Data.coarseGrain (t0, dt)
#
# The copy pasted code follows:
#
# Sort data in case we read unsorted data or different data files
Data.sort();

# N is number of point within the interval, X data to average
N = 0; 
X = [0.0] * Data.length
inInterval = False
#debug = True

for j in range (0, Data.size, 1):

	if debug:
		printf ('\nt = %g, interval (%f, %f)\n', Data.data[j][0], t0, t0 + dt)
	inInterval = False

	while not inInterval:

		if debug:
			printf ('inInterval loop: t = %g, interval (%f, %f)\n', Data.data[j][0], t0, t0 + dt)

		# if within an interval, sum up
		if Data.data[j][0] >= t0 and Data.data[j][0] < t0 + dt:
			N = N + 1
			inInterval = True
			for i in range (0, Data.length, 1):
				if debug:
					printf ('Adding %g to X[%i]\n', Data.data[j][i], i)
				X[i] = X[i] + Data.data[j][i]

		else:
			# if we have some points within the interval, average and print out
			if N != 0:
				if debug:
					printf ('N = %i\n', N)
				for i in range (0, Data.length, 1):
					X[i] = X[i] / N
					printf (' %f ', X[i])

				printf ('%i \n', N)

			# reset/clear
			if debug:
				printf ('\n************ Clearing/resetting:\n')

			while Data.data[j][0] >= t0 + dt:
				t0 = t0 + dt
			if debug:
				printf ('t = %g, interval (%f, %f)\n', Data.data[j][0], t0, t0 + dt)

			N = 1
			inInterval = True
			for i in range (0, Data.length, 1):
				X[i] = Data.data[j][i]
				if debug:
					printf ('Adding %g to X[%i]\n', Data.data[j][i], i)
# Look maybe we have something left
if N != 0:
	for i in range (0, Data.length, 1):
		X[i] = X[i] / N
		printf (' %f ', X[i])
	printf ('%i \n', N)

# This would be the code if no BUG
#for j in range (0, len(CoarseGrained.data), 1):
#	for i in range (0, CoarseGrained.length, 1):
#		printf (' %f ', CoarseGrained.data[j][i])
#	printf ('\n')

