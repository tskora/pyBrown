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

# import ctypes
# import math
import numpy as np

# from array import array
# from ctypes.util import find_library

# lib_name = "pyBrown"
# lib_path = find_library(lib_name)
# lib = ctypes.cdll.LoadLibrary( lib_path )

#-------------------------------------------------------------------------------

class Plane():
	"""This is a class representing planes/walls appearing in Brownian and Stokesian
	dynamics simulations. First and foremost, it contains geometric parameters of the
	plane -- the position of any point on it and its normal vector. Apart from that,
	plane's label (name) is present there as well.

	:param plane_point: cartesian coordinates of the point on the plane (any point)
	:type plane_point: class: `numpy.ndarray(3)`
	:param normal vector: vector normal to the plane
	:type normal_vector: class: `numpy.ndarray(3)`
	:param label: plane label
	:type label: `string`

	Constructor method

	:param plane_point: cartesian coordinates of the point on the plane (any point)
	:type plane_point: `[float, float, float]`
	:param normal_vector: vector normal to the plane
	:type normal_vector: `[float, float, float]`
	:param label: bead label, defaults to `"XXX"`
	:type label: `string`
	"""

	def __init__(self, plane_point, normal_vector, label = "XXX"):

		self.plane_point = np.array(plane_point)
		self.normal_vector = np.array(normal_vector)
		self.label = label

	#-------------------------------------------------------------------------------

	def keep_in_box(self, box_length):
		# """Moves the bead into the box if it is outside of it by using a combination of
		# translations parallel to the box sides and with length of a box length. Box's interior
		# lies between `-box_length/2` and `box_length/2`.
		
		# :param box_length: simulation box length
		# :type box_length: `float`
		# """

		for i in range(3):
			while self.plane_point[i] < -box_length / 2:
				self.plane_point[i] += box_length
			while self.plane_point[i] >= box_length / 2:
				self.plane_point[i] -= box_length

	#-------------------------------------------------------------------------------

	def __str__(self):

		return "{}: plane_point = {}, normal_vector = {}".format(self.label, self.plane_point, self.normal_vector)

	#-------------------------------------------------------------------------------

	def __repr__(self):

		return self.__str__()

	#-------------------------------------------------------------------------------

	def __eq__(self, p):

		if isinstance( p, Plane ):
			return ( np.all( self.normal_vector == p.normal_vector ) ) and ( np.dot(p.plane_point - self.plane_point, self.normal_vector) == 0.0 )
		return False

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------