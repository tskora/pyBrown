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

import ctypes
import math
import numpy as np

from array import array

from distutils import sysconfig
suffix = sysconfig.get_config_var('EXT_SUFFIX')
lib = ctypes.cdll.LoadLibrary( 'libpybrown'+suffix )

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

	def __init__(self, coords, hydrodynamic_radius, label = "XXX", hard_core_radius = None, epsilon_LJ = 0.0, mobile = True, bead_id = None):

		self.r = np.array(coords)
		self.dims = len(self.r)
		self.a = hydrodynamic_radius
		self.label = label
		self.mobile = mobile
		self.hard_core_radius = hard_core_radius if hard_core_radius is not None else self.a
		self.epsilon_LJ = epsilon_LJ
		self.bead_id = bead_id

		self.bonded_with = []
		self.bonded_how = {}

		self.angled_with = []
		self.angled_how = {}

	#-------------------------------------------------------------------------------

	def translate(self, vector):
		"""Moves a bead by the provided vector.
		
		:param vector: translation vector
		:type vector: class: `numpy.ndarray(3)`
		"""

		if self.mobile: self.r += vector

	#-------------------------------------------------------------------------------

	def translate_and_return_flux(self, vector, plane):
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

		f0 = np.dot( plane.normal_vector, (r0 - plane.plane_point) ) > 0.0

		self.translate(vector)

		f1 = np.dot( plane.normal_vector, (self.r - plane.plane_point) ) > 0.0

		if f0 == f1: return 0

		if f0: return -1

		else: return 1

	#-------------------------------------------------------------------------------

	def translate_and_check_for_plane_crossing(self, vector, planes):

		r0 = np.array( [self.r[0], self.r[1], self.r[2]] )

		self.translate(vector)

		r1 = np.array( [self.r[0], self.r[1], self.r[2]] )

		for plane in planes:

			r0f = r0 + self.a*plane.normal_vector

			r0b = r0 - self.a*plane.normal_vector

			r1f = r1 + self.a*plane.normal_vector

			r1b = r1 - self.a*plane.normal_vector

			f0f = np.dot( plane.normal_vector, (r0f - plane.plane_point) ) > 0.0

			f0b = np.dot( plane.normal_vector, (r0b - plane.plane_point) ) > 0.0

			f1f = np.dot( plane.normal_vector, (r1f - plane.plane_point) ) > 0.0

			f1b = np.dot( plane.normal_vector, (r1b - plane.plane_point) ) > 0.0

			if f0f == f1f and f0b == f1b: continue

			else: return True

		return False

	#-------------------------------------------------------------------------------

	def keep_in_box(self, box_length):
		"""Moves the bead into the box if it is outside of it by using a combination of
		translations parallel to the box sides and with length of a box length. Box's interior
		lies between `-box_length/2` and `box_length/2`.
		
		:param box_length: simulation box length
		:type box_length: `float`
		"""

		for i in range(self.dims):
			while self.r[i] < -box_length / 2:
				self.r[i] += box_length
			while self.r[i] >= box_length / 2:
				self.r[i] -= box_length

	#-------------------------------------------------------------------------------

	def is_there_bond_between(self, bead):

		return ( bead.bead_id in self.bonded_with or self.bead_id in bead.bonded_with  )

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

def pole_pointer(bead1, bead2, bead3):

	shift = bead2.r - bead1.r

	shift *= bead2.a / np.linalg.norm(shift)

	return bead3.r - bead2.r - shift

#-------------------------------------------------------------------------------

def pointer_pbc(bead1, bead2, box_size):
	"""Computes the voctor pointing from `bead1` to `bead2` (its closest translational replica)
	
	:param bead1: bead from which the vector points
	:type bead1: class: `pyBrown.bead.Bead`
	:param bead2: bead to which the vector points
	:type bead2: class: `pyBrown.bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`
	
	:return: pointing vector
	:rtype: class: `numpy.ndarray(3)`
	"""

	assert bead1.dims == bead2.dims
	r = pointer(bead1, bead2)

	for i in range(bead1.dims):
		while r[i] >= box_size/2:
			r[i] -= box_size
		while r[i] <= -box_size/2:
			r[i] += box_size

	return r

#-------------------------------------------------------------------------------

def pole_pointer_pbc(bead1, bead2, bead3, box_size):

	assert bead1.dims == bead2.dims and bead1.dims == bead3.dims
	r = pole_pointer(bead1, bead2, bead3)

	for i in range(bead1.dims):
		while r[i] >= box_size/2:
			r[i] -= box_size
		while r[i] <= -box_size/2:
			r[i] += box_size

	return r

#-------------------------------------------------------------------------------

def angle_pbc(r12, r23):

	assert len(r12) == len(r23)

	if len(r12) == 3:
		dist12 = math.sqrt( r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2] )
		dist23 = math.sqrt( r23[0]*r23[0] + r23[1]*r23[1] + r23[2]*r23[2] )
		return np.rad2deg( np.arccos( -( r12[0]*r23[0] + r12[1]*r23[1] + r12[2]*r23[2] ) / dist12 / dist23 ) )
	elif len(r12) == 2:
		dist12 = math.sqrt( r12[0]*r12[0] + r12[1]*r12[1] )
		dist23 = math.sqrt( r23[0]*r23[0] + r23[1]*r23[1] )	
		return np.rad2deg( np.arccos( -( r12[0]*r23[0] + r12[1]*r23[1] ) / dist12 / dist23 ) )
	else:
		1/0

#-------------------------------------------------------------------------------

def compute_pointer_pbc_matrix(beads, box_length, dims = 3):

	c_double = ctypes.c_double

	N = len(beads)
	N_c = ctypes.c_int(N)
	dims_c = ctypes.c_int(dims)

	r_list = [ ri  for b in beads for ri in b.r]
	v0 = array('d', r_list)
	r = (c_double * len(v0)).from_buffer(v0)

	p_list = [0.0]*N*N*dims
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	box_length_c = ctypes.c_double(box_length)

	lib.pointer_pbc_matrix(r, N_c, box_length_c, p, dims_c)

	rij = np.reshape(p, (N, N, dims))

	return rij

#-------------------------------------------------------------------------------

def compute_pointer_immobile_pbc_matrix(mobile_beads, immobile_beads, box_length, dims = 3):

	c_double = ctypes.c_double

	N_mob = len(mobile_beads)
	N_mob_c = ctypes.c_int(N_mob)

	N_immob = len(immobile_beads)
	N_immob_c = ctypes.c_int(N_immob)

	dims_c = ctypes.c_int(dims)

	r_mob_list = [ ri  for b in mobile_beads for ri in b.r]
	v0 = array('d', r_mob_list)
	r_mob = (c_double * len(v0)).from_buffer(v0)

	r_immob_list = [ ri  for b in immobile_beads for ri in b.r]
	v1 = array('d', r_immob_list)
	r_immob = (c_double * len(v1)).from_buffer(v1)

	p_list = [0.0]*N_mob*N_immob*dims
	v2 = array('d', p_list)
	p = (c_double * len(v2)).from_buffer(v2)

	box_length_c = ctypes.c_double(box_length)

	lib.pointer_immobile_pbc_matrix(r_mob, r_immob, N_mob_c, N_immob_c, box_length_c, p, dims_c)

	rik = np.reshape(p, (N_mob, N_immob, dims))

	return rik

#-------------------------------------------------------------------------------

def distance(bead1, bead2):

	assert bead1.dims == bead2.dims
	r = pointer(bead1, bead2)

	if len(r) == 3: return math.sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] )
	elif len(r) == 2: return math.sqrt( r[0]*r[0] + r[1]*r[1] )
	else: 1/0

#-------------------------------------------------------------------------------

def distance_pbc(bead1, bead2, box_size):
	"""Computes the distance between `bead1` and `bead2` (its closest translational replica)

	:param bead1: bead
	:type bead1: class: `pyBrown.bead.Bead`
	:param bead2: bead
	:type bead2: class: `pyBrown.bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`

	:return: distance
	:rtype: `float`
	"""

	r = pointer_pbc(bead1, bead2, box_size)

	if len(r) == 3:	return math.sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] )
	elif len(r) == 2:	return math.sqrt( r[0]*r[0] + r[1]*r[1] )
	else: 1/0

#-------------------------------------------------------------------------------

def pole_distance_pbc(bead1, bead2, bead3, box_size):

	r = pole_pointer_pbc(bead1, bead2, bead3, box_size)

	if len(r) == 3: return math.sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] )
	elif len(r) == 2: return math.sqrt( r[0]*r[0] + r[1]*r[1])
	else: 1/0

#-------------------------------------------------------------------------------

def overlap(bead1, bead2, epsilon = 0.0):

	return distance(bead1, bead2) <= bead1.a + bead2.a + epsilon

#-------------------------------------------------------------------------------

def overlap_pbc(bead1, bead2, box_size, epsilon = 0.0):
	"""Checks if there is an overlap between `bead1` and `bead2` (its closest translational replica)
	
	:param bead1: bead
	:type bead1: class: `pyBrown.bead.Bead`
	:param bead2: bead
	:type bead2: class: `pyBrown.bead.Bead`
	:param box_size: simulation box length
	:type box_size: `float`

	:return: do beads overlap?
	:rtype: `bool`
	"""

	dist = distance_pbc(bead1, bead2, box_size)

	return dist <= bead1.a + bead2.a + epsilon

#-------------------------------------------------------------------------------

def build_connection_matrix(beads):

	connection_matrix = np.zeros((len(beads), len(beads)), dtype = int)

	for i in range(len(beads)):

		for j in range(len(beads)):

			if beads[i].is_there_bond_between(beads[j]): connection_matrix[i][j] = connection_matrix[j][i] = 1

			if not (beads[i].mobile or beads[j].mobile): connection_matrix[i][j] = connection_matrix[j][i] = 1

	return connection_matrix

#-------------------------------------------------------------------------------

def check_overlaps(beads, box_length, overlap_treshold, connection_matrix, dims = 3):

	c_double = ctypes.c_double
	c_int = ctypes.c_int

	N = len(beads)
	N_c = ctypes.c_int(N)

	dims_c = ctypes.c_int(dims)

	r_list = [ ri  for b in beads for ri in b.r]
	v0 = array('d', r_list)
	r = (c_double * len(v0)).from_buffer(v0)

	a_list = [ b.a  for b in beads]
	v1 = array('d', a_list)
	a = (c_double * len(v1)).from_buffer(v1)

	box_length_c = ctypes.c_double(box_length)

	overlap_treshold_c = ctypes.c_double(overlap_treshold)

	connection_matrix_list = [ connection_matrix[i][j] for i in range(N) for j in range(N) ]
	v2 = array('i', connection_matrix_list)
	connection_matrix_c = (c_int * len(v2)).from_buffer(v2)
	
	overlaps = lib.check_overlaps(r, a, N_c, box_length_c, overlap_treshold_c, connection_matrix_c, dims_c)

	return overlaps == 1

#-------------------------------------------------------------------------------

def get_bead_with_id(beads, bead_id):

	for i in range(len(beads)):

		if beads[i].bead_id == bead_id: return i

	return None

#-------------------------------------------------------------------------------