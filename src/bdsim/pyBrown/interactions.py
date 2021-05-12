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

import importlib
import math
import numpy as np

from scipy.constants import elementary_charge

#-------------------------------------------------------------------------------

class Interactions():

	def __init__(self, force_function, energy_function, auxiliary_force_parameters):

		self.force = force_function

		self.energy = energy_function

		self.auxiliary_force_parameters = auxiliary_force_parameters

	#-------------------------------------------------------------------------------

	def compute_forces_and_energy(self, beads, pointers, F):

		E = 0.0

		temporary_force = np.zeros(3*len(beads))

		for i in range(1, len(beads)):

			beadi = beads[i]

			for j in range(i):

				beadj = beads[j]

				pointerij = pointers[i][j]

				f = self._compute_force(beadi, beadj, pointerij, self.auxiliary_force_parameters)

				temporary_force[3*i:3*(i+1)] += f

				temporary_force[3*j:3*(j+1)] -= f

				E += self._compute_pair_energy(beadi, beadj, pointerij, self.auxiliary_force_parameters)

		self._rearrange_force_to_ommit_immobile_beads(beads, temporary_force, F)

		return E

	#-------------------------------------------------------------------------------

	def _compute_force(self, bead1, bead2, pointer, auxiliary_force_parameters):

		return self.force(bead1, bead2, pointer, **auxiliary_force_parameters)

	#-------------------------------------------------------------------------------

	def _compute_pair_energy(self, bead1, bead2, pointer, auxiliary_force_parameters):

		return self.energy(bead1, bead2, pointer, **auxiliary_force_parameters)

	#-------------------------------------------------------------------------------

	def _rearrange_force_to_ommit_immobile_beads(self, beads, temporary_force, F):

		i = 0

		j = 0

		while i < len(beads):

			if beads[i].mobile:

				F[3*j:3*(j+1)] += temporary_force[3*i:3*(i+1)]

				j += 1

			i += 1

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

	_set_coulomb_interactions(input_data, interactions_for_simulation)

	_set_custom_interactions(input_data, interactions_for_simulation)

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

def _set_coulomb_interactions(input_data, interactions_for_simulation):

	if not input_data["coulomb"]:

		return

	aux_keywords = [ "dielectric_constant" ]

	aux = { keyword: input_data[keyword] for keyword in aux_keywords }

	i = Interactions(coulomb_force, coulomb_energy, aux)

	interactions_for_simulation.append(i)

#-------------------------------------------------------------------------------

def _set_custom_interactions(input_data, interactions_for_simulation):

	if not input_data["custom_interactions"]:

		return

	filename = input_data["custom_interactions_filename"]

	aux = input_data["auxiliary_custom_interactions_keywords"]

	custom_module = importlib.import_module( filename.split('.')[0] )

	energy_functions = [ function_name for function_name in dir(custom_module) if function_name[0] != "_" if function_name[-6:] == "energy" ]

	force_functions = [ function_name for function_name in dir(custom_module) if function_name[0] != "_" if function_name[-5:] == "force" ]

	energy_dictionary = { e[:-6]:e for e in energy_functions }

	force_dictionary = { f[:-5]:f for f in force_functions }

	assert len(energy_functions) == len(force_functions), """unequal number of energy and force 
															 functions in custom interactions file"""

	assert energy_dictionary.keys() == force_dictionary.keys(), """different names of force and
																  energy functions in custom 
																  interactions file"""

	for key in energy_dictionary.keys():

		ff = eval( 'custom_module.' + force_dictionary[key] )

		ef = eval( 'custom_module.' + energy_dictionary[key] )

		interactions_for_simulation.append( Interactions(ff, ef, aux) )

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

def coulomb_energy(bead1, bead2, pointer, dielectric_constant):

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	dist = math.sqrt(dist2)

	k = elementary_charge * elementary_charge / ( 4.0 * np.pi * dielectric_constant )

	return k * bead1.charge * bead2.charge / dist

#-------------------------------------------------------------------------------

def coulomb_force(bead1, bead2, pointer, dielectric_constant):

	dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

	dist = math.sqrt(dist2)

	k = elementary_charge * elementary_charge / ( 4.0 * np.pi * dielectric_constant )

	versor = pointer / dist

	return -k * bead1.charge * bead2.charge / dist2 * versor

#-------------------------------------------------------------------------------

def kcal_per_mole_to_joule(value):

	return 10**(-21) * 6.9477 * value

#-------------------------------------------------------------------------------

def eV_to_joule(value):

	return 10**(-19) * 1.602176634 * value

#-------------------------------------------------------------------------------