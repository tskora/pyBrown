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
	"""This is a class representing spherical beads used in Brownian and Stokesian
	dynamics simulations. First and foremost, it contains geometric parameters of the
	bead -- the position of its center and its hydrodynamic radius. Apart from that,
	bead's label (name) is present there as well. Finally, bead parametrization with
	respect to Lennard-Jones interaction potential is contained -- Lennard-Jones energy
	and radius. Energy units are declared independently in input ``.json`` file.

	:param r: cartesian coordinates of the bead center
	:type r: class: `numpy.ndarray(3)`
	:param a: hydrodynamic radius
	:type a: `float`
	:param label: bead label
	:type label: `string`
	:param mobile: switching on/off mobility of the bead
	:type mobile: `bool`
	:param hard_core_radius: Lennard-Jones radius: for two interacting particles
							 the potential crosses :math:`0` for separation equal to
							 sum of their Lennard-Jones (hard core) radii.
	:type hard_core_radius: `float`
	:param epsilon_LJ: Lennard-Jones energy: for two interacting particles the
					   depth of the potential (in case of using both LJ6 and LJ12 terms)
					   is equal to geometric mean of their Lennard-Jones energies.
	:type epsilon_LJ: `float`

	Constructor method

	:param coords: cartesian coordinates of the bead center
	:type coords: `[float, float, float]`
	:param hydrodynamic_radius: hydrodynamic radius
	:type hydrodynamic_radius: `float`
	:param label: bead label, defaults to `"XXX"`
	:type label: `string`
	:param hard_core_radius: radius used in computing Lennard-Jones
							 potential, defaults to `hydrodynamic_radius`
	:type hard_core_radius: `float`
	:param epsilon_LJ: energy of Lennard-Jones interaction (for two beads
					   interaction energy is a geometric mean of their Lennard-Jones
					   energies), defaults to `0.0`
	:type epsilon_LJ: `float`
	:param mobile: switching on/off mobility of the bead, defaults to `True`
	:type mobile: `bool`
	"""

	def __init__(self, coords, hydrodynamic_radius, label = "XXX", hard_core_radius = None, epsilon_LJ = 0.0, mobile = True):

		self.r = np.array(coords)
		self.a = hydrodynamic_radius
		self.label = label
		self.mobile = mobile
		self.hard_core_radius = hard_core_radius if hard_core_radius is not None else self.a
		self.epsilon_LJ = epsilon_LJ

	#-------------------------------------------------------------------------------

	def translate(self, vector):
		"""Moves a bead by the provided vector.
		
		:param vector: translation vector
		:type vector: class: `numpy.ndarray(3)`
		"""

		if self.mobile: self.r += vector

	#-------------------------------------------------------------------------------

	def translate_and_return_flux(self, vector, normal, plane_point):
		"""Moves a bead by the provided vector and returns flux through the provided plane.
		Plane is defined by its normal vector and any point lying on the plane. `1` is returned
		if particle goes through the plana in the direction of the normal vector, `-1` if in the
		opposite direction and `0` if the particle does not go through the plane whatsoever.
		
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
		translations parallel to the box sides and with length of a box length. Box's interior
		lies between `-box_length/2` and `box_length/2`.
		
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

def pointer(bead1, bead2):

	return bead2.r - bead1.r

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

	r = pointer(bead1, bead2)

	for i in range(3):
		while r[i] >= box_size/2:
			r[i] -= box_size
		while r[i] <= -box_size/2:
			r[i] += box_size

	return r

#-------------------------------------------------------------------------------

def distance(bead1, bead2):

	r = pointer(bead1, bead2)

	return math.sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] )

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

	r = pointer(bead1, bead2)

	for i in range(3):
		while r[i] >= box_size/2:
			r[i] -= box_size
		while r[i] <= -box_size/2:
			r[i] += box_size

	return math.sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] )

#-------------------------------------------------------------------------------

def overlap(bead1, bead2):

	return distance(bead1, bead2) <= bead1.a + bead2.a

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