# pyBrown is a Brownian and Stokesian dynamics simulation tool
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

#-------------------------------------------------------------------------------

class Bead():
	"""This is a class representing spherical beads used in Brownian and Stokesian dynamics simulations.

	:param r: cartesian coordinates of the bead center
	:type r: class: `numpy.ndarray(3)`
	:param a: hydrodynamic radius
	:type a: `float`
	:param label: bead label
	:type label: `string`
	:param mobile: is bead mobile
	:type mobile: `bool`
	"""

	def __init__(self, coords, hydrodynamic_radius, label = "XXX", mobile = True):
		"""Constructor method

		:param coords: cartesian coordinates of the bead center
		:type coords: `[float, float, float]`
		:param hydrodynamic_radius: hydrodynamic radius
		:type hydrodynamic_radius: `float`
		:param label: bead label, defaults to `"XXX"`
		:type label: `string`
		:param mobile: is bead mobile, defaults to `True`
		:type mobile: `bool`
		"""

		self.r = np.array(coords)
		self.a = hydrodynamic_radius
		self.label = label
		self.mobile = mobile

	#-------------------------------------------------------------------------------

	def translate(self, vector):
		"""Moves a bead by the provided vector.
		
		:param vector: translation vector
		:type vector: class: `numpy.ndarray(3)`
		"""

		if self.mobile: self.r += vector

	#-------------------------------------------------------------------------------

	def translate_and_return_flux(self, vector, normal, plane_point):
		"""Moves a bead by the provided vector and returns flux through the provided plane
		
		:param vector: translation vector
		:type vector: class: `numpy.ndarray(3)`
		:param normal: plane normal vector
		:type noraal: class: `numpy.ndarray(3)`
		:param plane_point: any point in the plane
		:type plane_point: class: `numpy.ndarray(3)`
		
		:return: flux through the provided plane
		:rtype: `int`
		"""

		r0 = np.array( [self.r[0], self.r[1], self.r[2]] )

		f0 = np.dot( normal, (r0 - plane_point) ) > 0.0

		self.translate(vector)

		f1 = np.dot( normal, (self.r - plane_point) ) > 0.0

		if f0 == f1: return 0

		if f0: return -1

		else: return 1

	#-------------------------------------------------------------------------------

	def keep_in_box(self, box_length):
		"""Moves the bead into the box if it is outside of it by using a combination of
		translations parallel to the box sides and with length of a box length
		
		:param box_length: simulation box length
		:type box_length: `float`
		"""

		for i in range(3):
			while self.r[i] < -box_length / 2:
				self.r[i] += box_length
			while self.r[i] >= box_length / 2:
				self.r[i] -= box_length

	#-------------------------------------------------------------------------------

	def __str__(self):

		if self.mobile: mobile_string = "mobile"
		else: mobile_string = "immobile"

		return "{}: {}, radius = {}, {}".format(self.label, self.r, self.a, mobile_string)

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
	"""Computes the voctor pointing from `bead1` to `bead2` (its closest translational replica)
	
	:param bead1: bead from which the vector points
	:type bead1: class: `bead.Bead`
	:param bead2: bead to which the vector points
	:type bead2: class: `bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`
	
	:return: pointing vector
	:rtype: class: `numpy.ndarray(3)`
	"""

	pointer = bead2.r - bead1.r

	for i in range(3):
		while pointer[i] >= box_size/2:
			pointer[i] -= box_size
		while pointer[i] <= -box_size/2:
			pointer[i] += box_size

	return pointer

#-------------------------------------------------------------------------------

def distance_pbc(bead1, bead2, box_size):
	"""Computes the distance between `bead1` and `bead2` (its closest translational replica)

	:param bead1: bead
	:type bead1: class: `bead.Bead`
	:param bead2: bead
	:type bead2: class: `bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`

	:return: distance
	:rtype: `float`
	"""

	pointer = bead1.r - bead2.r

	for i in range(3):
		while pointer[i] >= box_size/2:
			pointer[i] -= box_size
		while pointer[i] <= -box_size/2:
			pointer[i] += box_size

	return math.sqrt( np.sum( pointer**2 ) )

#-------------------------------------------------------------------------------

def overlap_pbc(bead1, bead2, box_size):
	"""Checks if there is an overlap between `bead1` and `bead2` (its closest translational replica)
	
	:param bead1: bead
	:type bead1: class: `bead.Bead`
	:param bead2: bead
	:type bead2: class: `bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`

	:return: do beads overlap?
	:rtype: `bool`
	"""

	dist = distance_pbc(bead1, bead2, box_size)

	return dist <= bead1.a + bead2.a

#-------------------------------------------------------------------------------