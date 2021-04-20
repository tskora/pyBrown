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

import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from pyBrown.bead import Bead, pointer
from pyBrown.interactions import Interactions, LJ_6_attractive_energy, LJ_6_attractive_force,\
								 LJ_12_repulsive_energy, LJ_12_repulsive_force,\
								 LJ_6_12_energy, LJ_6_12_force

#-------------------------------------------------------------------------------

class TestInteractions(unittest.TestCase):

	def setUp(self):

		pass

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma(self):

		alpha = 4.0

		epsilon = 1.0

		d = 2.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, epsilon_LJ = epsilon)

		b2 = Bead([0.0, 0.0, d], 1.0, epsilon_LJ = epsilon)

		beads = [ b1, b2 ]

		rij = np.zeros((len(beads), len(beads), 3))

		for i in range(1, len(beads)):
			for j in range(0, i):
				rij[i][j] = pointer(beads[i], beads[j])
				rij[j][i] = -rij[i][j]

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, {"alpha": alpha})

		F = np.zeros(6)

		E = 0.0

		E = i.compute_forces_and_energy([b1, b2], rij, F, E)

		self.assertEqual(E, -alpha*epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 6.0*alpha*epsilon/d, 0.0, 0.0, -6.0*alpha*epsilon/d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_12_particles_in_sigma(self):

		alpha = 4.0

		epsilon = 1.0

		d = 2.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, epsilon_LJ = epsilon)

		b2 = Bead([0.0, 0.0, d], 1.0, epsilon_LJ = epsilon)

		beads = [ b1, b2 ]

		rij = np.zeros((len(beads), len(beads), 3))

		for i in range(1, len(beads)):
			for j in range(0, i):
				rij[i][j] = pointer(beads[i], beads[j])
				rij[j][i] = -rij[i][j]

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, {"alpha": alpha})

		F = np.zeros(6)

		E = 0.0

		E = i.compute_forces_and_energy([b1, b2], rij, F, E)

		self.assertEqual(E, alpha*epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -12.0*alpha*epsilon/d, 0.0, 0.0, 12.0*alpha*epsilon/d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_in_sigma(self):

		alpha = 4.0

		epsilon = 1.0

		d = 2.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, epsilon_LJ = epsilon)

		b2 = Bead([0.0, 0.0, d], 1.0, epsilon_LJ = epsilon)

		beads = [ b1, b2 ]

		rij = np.zeros((len(beads), len(beads), 3))

		for i in range(1, len(beads)):
			for j in range(0, i):
				rij[i][j] = pointer(beads[i], beads[j])
				rij[j][i] = -rij[i][j]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, {"alpha": alpha})

		ia = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, {"alpha": alpha})

		ir = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, {"alpha": alpha})

		F = np.zeros(6)

		Fa = np.zeros(6)

		Fr = np.zeros(6)

		E = Ea = Er = 0.0

		E = i.compute_forces_and_energy([b1, b2], rij, F, E)

		Ea = ia.compute_forces_and_energy([b1, b2], rij, Fa, Ea)

		Er = ir.compute_forces_and_energy([b1, b2], rij, Fr, Er)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -6.0*alpha*epsilon/d, 0.0, 0.0, 6.0*alpha*epsilon/d])

		self.assertSequenceEqual(list(Fa + Fr), [0.0, 0.0, -6.0*alpha*epsilon/d, 0.0, 0.0, 6.0*alpha*epsilon/d])

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------