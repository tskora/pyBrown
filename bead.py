# pyBD is a Brownian and Stokesian dynamics simulation tool
# Copyright (C) 2021  Tomasz Skora (tskora@ichf.edu.pl)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses.

import math
import numpy as np

from output import timing

#-------------------------------------------------------------------------------

class Bead():

	def __init__(self, coords, hydrodynamic_radius, label = "XXX", mobile = True):

		self.r = np.array(coords)
		self.a = hydrodynamic_radius
		self.label = label
		self.mobile = mobile

	#-------------------------------------------------------------------------------

	def translate(self, vector):

		if self.mobile: self.r += vector

	#-------------------------------------------------------------------------------

	def keep_in_box(self, box_length):

		for i in range(3):
			while self.r[i] < -box_length / 2:
				self.r[i] += box_length
			while self.r[i] >= box_length / 2:
				self.r[i] -= box_length

	#-------------------------------------------------------------------------------

	def __str__(self):

		return "{}: {}, radius = {}".format(self.label, self.r, self.a)

	#-------------------------------------------------------------------------------

	def __repr__(self):

		return self.__str__()

	#-------------------------------------------------------------------------------

	def __eq__(self, p):

		if isinstance( p, Bead ):
			return ( np.all( self.r == p.r ) ) and ( self.a == p.a )
		return False

#-------------------------------------------------------------------------------

def pointer_pbc(bead1, bead2, box_size):

	pointer = bead2.r - bead1.r

	for i in range(3):
		while pointer[i] >= box_size/2:
			pointer[i] -= box_size
		while pointer[i] <= -box_size/2:
			pointer[i] += box_size

	return pointer

#-------------------------------------------------------------------------------

def distance_pbc(bead1, bead2, box_size):

	pointer = bead1.r - bead2.r

	for i in range(3):
		while pointer[i] >= box_size/2:
			pointer[i] -= box_size
		while pointer[i] <= -box_size/2:
			pointer[i] += box_size

	return math.sqrt( np.sum( pointer**2 ) )

#-------------------------------------------------------------------------------

def overlap_pbc(bead1, bead2, box_size):

	dist = distance_pbc(bead1, bead2, box_size)

	return dist <= bead1.a + bead2.a

#-------------------------------------------------------------------------------