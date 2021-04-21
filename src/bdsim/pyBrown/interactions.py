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

#-------------------------------------------------------------------------------

class Interactions():

	def __init__(self, force_function, energy_function, auxiliary_force_parameters, energy_unit = "joule"):

		self.force = force_function

		self.energy = energy_function

		self.auxiliary_force_parameters = auxiliary_force_parameters

	#-------------------------------------------------------------------------------

	def compute_forces_and_energy(self, beads, pointers, F):

		E = 0.0

		for i in range(1, len(beads)):

			beadi = beads[i]

			for j in range(i):

				beadj = beads[j]

				pointerij = pointers[i][j]

				f = self._compute_force(beadi, beadj, pointerij, self.auxiliary_force_parameters)

				F[3*i:3*(i+1)] += f

				F[3*j:3*(j+1)] -= f

				E += self._compute_pair_energy(beadi, beadj, pointerij, self.auxiliary_force_parameters)

		return E

	#-------------------------------------------------------------------------------

	def _compute_force(self, bead1, bead2, pointer, auxiliary_force_parameters):

		return self.force(bead1, bead2, pointer, **auxiliary_force_parameters)

	#-------------------------------------------------------------------------------

	def _compute_pair_energy(self, bead1, bead2, pointer, auxiliary_force_parameters):

		return self.energy(bead1, bead2, pointer, **auxiliary_force_parameters)

	#-------------------------------------------------------------------------------

	def __str__(self):

		string_template = 'Force: {}, energy: {}, auxiliary parameters: {}'

		return string_template.format(self.force.__name__, self.energy.__name__, self.auxiliary_force_parameters)

	#-------------------------------------------------------------------------------

	def __repr__(self):

		return self.__str__()

#-------------------------------------------------------------------------------

def set_interactions(input_data):

	interactions_for_simulation = []

	_set_lennard_jones_interactions(input_data, interactions_for_simulation)

	return interactions_for_simulation

#-------------------------------------------------------------------------------

def _set_lennard_jones_interactions(input_data, interactions_for_simulation):

	if not input_data["lennard_jones_6"] and not input_data["lennard_jones_12"]:

		return

	aux_keywords = [ "lennard_jones_alpha" ]

	aux = { keyword: input_data[keyword] for keyword in aux_keywords }

	if input_data["lennard_jones_6"] and input_data["lennard_jones_12"]:

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, aux)

	elif input_data["lennard_jones_6"]:

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, aux)

	elif input_data["lennard_jones_12"]:

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, aux)

	interactions_for_simulation.append(i)

#-------------------------------------------------------------------------------

def LJ_6_attractive_force(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	dist = math.sqrt(dist2)

	s2 = sigma * sigma / dist2

	s6 = s2 * s2 * s2

	versor = pointer / dist

	return 6.0*lennard_jones_alpha*epsilon*s6/dist*versor

#-------------------------------------------------------------------------------

def LJ_6_attractive_energy(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	s2 = sigma * sigma / dist2

	s6 = s2 * s2 * s2

	return -lennard_jones_alpha*epsilon*s6

#-------------------------------------------------------------------------------

def LJ_12_repulsive_force(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	dist = math.sqrt(dist2)

	s2 = sigma * sigma / dist2

	s12 = s2 * s2 * s2 * s2 * s2 * s2

	versor = pointer / dist

	return -12.0*lennard_jones_alpha*epsilon*s12/dist*versor

#-------------------------------------------------------------------------------

def LJ_12_repulsive_energy(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	s2 = sigma * sigma / dist2

	s12 = s2 * s2 * s2 * s2 * s2 * s2

	return lennard_jones_alpha*epsilon*s12

#-------------------------------------------------------------------------------

def LJ_6_12_energy(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	s2 = sigma * sigma / dist2

	s6 = s2 * s2 * s2

	s12 = s6 * s6

	return lennard_jones_alpha*epsilon*(s12 - s6)

#-------------------------------------------------------------------------------

def LJ_6_12_force(bead1, bead2, pointer, lennard_jones_alpha):

	sigma = bead1.hard_core_radius + bead2.hard_core_radius

	epsilon = math.sqrt( bead1.epsilon_LJ * bead2.epsilon_LJ )

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	dist = math.sqrt(dist2)

	s2 = sigma * sigma / dist2

	s6 = s2 * s2 * s2

	s12 = s6 * s6

	versor = pointer / dist

	return lennard_jones_alpha*epsilon/dist*(6*s6-12*s12)*versor

#-------------------------------------------------------------------------------

def kcal_per_mole_to_joule(value):

	return 10**(-21) * 6.9477 * value

#-------------------------------------------------------------------------------
