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

from pyBrown.bead import Bead, pointer, compute_pointer_pbc_matrix, compute_pointer_immobile_pbc_matrix
from pyBrown.interactions import Interactions, LJ_6_attractive_energy, LJ_6_attractive_force,\
								 LJ_12_repulsive_energy, LJ_12_repulsive_force,\
								 LJ_6_12_energy, LJ_6_12_force, harmonic_bond_force, \
								 harmonic_bond_energy, harmonic_angle_force, harmonic_angle_energy, \
								 _set_custom_interactions

#-------------------------------------------------------------------------------

class TestInteractions(unittest.TestCase):

	def setUp(self):

		self.alpha = 4.0

		self.epsilon = 1.0

		self.d = 2.0

		self.a = 1.0

		self.b1 = Bead([0.0, 0.0, 0.0], self.a, epsilon_LJ = self.epsilon)

		self.b2 = Bead([0.0, 0.0, self.d], self.a, epsilon_LJ = self.epsilon)

		self.beads = [ self.b1, self.b2 ]

		self.rij = compute_pointer_pbc_matrix(self.beads, box_length = 1000000.0)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq(self):

		F = np.zeros(6)

		E = 0.0

		dist_eq = self.d

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_1_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False

		beads_mobile = [ self.b2 ]
		beads_immobile = [ self.b1 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_2_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b2.mobile = False

		beads_mobile = [ self.b1 ]
		beads_immobile = [ self.b2 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_12_immobile(self):

		F = np.zeros(0)

		E = 0.0

		dist_eq = self.d

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False
		self.b2.mobile = False

		beads_mobile = [  ]
		beads_immobile = [ self.b1, self.b2 ]

		rij = [ ]
		rik = [ ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_plus_1(self):

		F = np.zeros(6)

		E = 0.0

		dist_eq = self.d - 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant*(self.d - dist_eq), 0.0, 0.0, -force_constant*(self.d - dist_eq)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_plus_1_1_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d - 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False

		beads_mobile = [ self.b2 ]
		beads_immobile = [ self.b1 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -force_constant*(self.d - dist_eq)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_plus_1_2_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d - 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b2.mobile = False

		beads_mobile = [ self.b1 ]
		beads_immobile = [ self.b2 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant*(self.d - dist_eq)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_plus_1_12_immobile(self):

		F = np.zeros(0)

		E = 0.0

		dist_eq = self.d - 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False
		self.b2.mobile = False

		beads_mobile = [  ]
		beads_immobile = [ self.b1, self.b2 ]

		rij = [ ]
		rik = [ ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_minus_1(self):

		F = np.zeros(6)

		E = 0.0

		dist_eq = self.d + 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -force_constant*(dist_eq - self.d), 0.0, 0.0, force_constant*(dist_eq - self.d)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_minus_1_1_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d + 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False

		beads_mobile = [ self.b2 ]
		beads_immobile = [ self.b1 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant*(dist_eq - self.d)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_minus_1_2_immobile(self):

		F = np.zeros(3)

		E = 0.0

		dist_eq = self.d + 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b2.mobile = False

		beads_mobile = [ self.b1 ]
		beads_immobile = [ self.b2 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -force_constant*(dist_eq - self.d)])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbell_eq_minus_1_12_immobile(self):

		F = np.zeros(0)

		E = 0.0

		dist_eq = self.d + 1.0

		force_constant = 1.0

		self.b1.bead_id = 1
		self.b2.bead_id = 2

		self.b1.bonded_with.append(self.b2.bead_id)
		self.b1.bonded_how[self.b2.bead_id] = [ dist_eq, force_constant ]

		self.b1.mobile = False
		self.b2.mobile = False

		beads_mobile = [ ]
		beads_immobile = [ self.b1, self.b2 ]

		rij = [ ]
		rik = [ ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbells(self):

		F = np.zeros(3*5)

		E = 0.0

		dist_eq_1 = self.d + 1.0
		dist_eq_2 = self.d - 1.0

		force_constant_1 = 1.0
		force_constant_2 = 2.0

		b1 = Bead([0.0, 0.0, 0.0], self.a)
		b2 = Bead([0.0, 0.0, self.d], self.a)
		b3 = Bead([0.0, 2*self.d, self.d/2], self.a)
		b4 = Bead([0.0, 4*self.d, self.d/2], self.a)
		b5 = Bead([0.0, 5*self.d, self.d/2], self.a)

		b1.bead_id = 1
		b2.bead_id = 2
		b3.bead_id = 3
		b4.bead_id = 4
		b5.bead_id = 5

		b1.bonded_with.append(b2.bead_id)
		b1.bonded_how[b2.bead_id] = [ dist_eq_1, force_constant_1 ]

		b5.bonded_with.append(b4.bead_id)
		b5.bonded_how[b4.bead_id] = [ dist_eq_2, force_constant_2 ]

		beads = [b1, b2, b3, b4, b5]
		rij = compute_pointer_pbc_matrix(beads, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads, immobile_beads = [], pointers_mobile = rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, 0.5*force_constant_1*(dist_eq_1 - self.d)**2 + 0.5*force_constant_2*(dist_eq_2 - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -force_constant_1*(dist_eq_1 - self.d), 0.0, 0.0, force_constant_1*(dist_eq_1 - self.d), 0.0, 0.0, 0.0, 0.0, force_constant_2*(self.d - dist_eq_2), 0.0, 0.0, -force_constant_2*(self.d - dist_eq_2), 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbells_14_immobile(self):

		F = np.zeros(3*3)

		E = 0.0

		dist_eq_1 = self.d + 1.0
		dist_eq_2 = self.d - 1.0

		force_constant_1 = 1.0
		force_constant_2 = 2.0

		b1 = Bead([0.0, 0.0, 0.0], self.a)
		b2 = Bead([0.0, 0.0, self.d], self.a)
		b3 = Bead([0.0, 2*self.d, self.d/2], self.a)
		b4 = Bead([0.0, 4*self.d, self.d/2], self.a)
		b5 = Bead([0.0, 5*self.d, self.d/2], self.a)

		b1.bead_id = 1
		b2.bead_id = 2
		b3.bead_id = 3
		b4.bead_id = 4
		b5.bead_id = 5

		b1.mobile = False
		b4.mobile = False

		b1.bonded_with.append(b2.bead_id)
		b1.bonded_how[b2.bead_id] = [ dist_eq_1, force_constant_1 ]

		b5.bonded_with.append(b4.bead_id)
		b5.bonded_how[b4.bead_id] = [ dist_eq_2, force_constant_2 ]

		beads_mobile = [b2, b3, b5]
		beads_immobile = [b1, b4]
		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant_1*(dist_eq_1 - self.d)**2 + 0.5*force_constant_2*(dist_eq_2 - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant_1*(dist_eq_1 - self.d), 0.0, 0.0, 0.0, 0.0, -force_constant_2*(self.d - dist_eq_2), 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_harmonic_dumbbells_135_immobile(self):

		F = np.zeros(2*3)

		E = 0.0

		dist_eq_1 = self.d + 1.0
		dist_eq_2 = self.d - 1.0

		force_constant_1 = 1.0
		force_constant_2 = 2.0

		b1 = Bead([0.0, 0.0, 0.0], self.a)
		b2 = Bead([0.0, 0.0, self.d], self.a)
		b3 = Bead([0.0, 2*self.d, self.d/2], self.a)
		b4 = Bead([0.0, 4*self.d, self.d/2], self.a)
		b5 = Bead([0.0, 5*self.d, self.d/2], self.a)

		b1.bead_id = 1
		b2.bead_id = 2
		b3.bead_id = 3
		b4.bead_id = 4
		b5.bead_id = 5

		b1.mobile = False
		b3.mobile = False
		b5.mobile = False

		b1.bonded_with.append(b2.bead_id)
		b1.bonded_how[b2.bead_id] = [ dist_eq_1, force_constant_1 ]

		b5.bonded_with.append(b4.bead_id)
		b5.bonded_how[b4.bead_id] = [ dist_eq_2, force_constant_2 ]

		beads_mobile = [b2, b4]
		beads_immobile = [b1, b3, b5]
		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.5*force_constant_1*(dist_eq_1 - self.d)**2 + 0.5*force_constant_2*(dist_eq_2 - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant_1*(dist_eq_1 - self.d), 0.0, force_constant_2*(self.d - dist_eq_2), 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq(self):

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		beads = [b1, b2, b3]

		rij = compute_pointer_pbc_matrix(beads, box_length = 1000000.0)

		b1.angled_with.append([b2.bead_id, b3.bead_id])
		b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

		F = harmonic_angle_force(b1, b2, b3, rij[0][1], rij[1][2], box_length = 1000000.0)

		E = harmonic_angle_energy(b1, b2, b3, rij[0][1], rij[1][2], box_length = 1000000.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2(self):

		F = np.zeros(9)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		beads = [b1, b2, b3]

		rij = compute_pointer_pbc_matrix(beads, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads, immobile_beads = [], pointers_mobile = rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_1_immobile(self):

		F = np.zeros(6)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b1.mobile = False

		beads_mobile = [b2, b3]
		beads_immobile = [b1]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_2_immobile(self):

		F = np.zeros(6)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b2.mobile = False

		beads_mobile = [b1, b3]
		beads_immobile = [b2]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_3_immobile(self):

		F = np.zeros(6)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b3.mobile = False

		beads_mobile = [b1, b2]
		beads_immobile = [b3]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_12_immobile(self):

		F = np.zeros(3)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b1.mobile = False
		b2.mobile = False

		beads_mobile = [b3]
		beads_immobile = [b1, b2]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_13_immobile(self):

		F = np.zeros(3)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b1.mobile = False
		b3.mobile = False

		beads_mobile = [b2]
		beads_immobile = [b1, b3]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_23_immobile(self):

		F = np.zeros(3)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b2.mobile = False
		b3.mobile = False

		beads_mobile = [b1]
		beads_immobile = [b2, b3]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = 1000000.0)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = 1000000.0)

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_eq_2_123_immobile(self):

		F = np.zeros(0)

		E = 0.0

		angle_eq = 90.0

		force_constant = 1.0

		b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
		b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
		b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

		b1.mobile = False
		b2.mobile = False
		b3.mobile = False

		beads_mobile = []
		beads_immobile = [b1, b2, b3]

		rij = []
		rik = []

		i = Interactions(harmonic_angle_force, harmonic_angle_energy, bonded = True, how_many_body = 3)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_noneq_fundamentals(self):

		for angle in np.linspace(0.0, 360.0, 100):

			angle_eq = angle

			force_constant = 1.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			beads = [b1, b2, b3]

			rij = compute_pointer_pbc_matrix(beads, box_length = 1000000.0)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			F = harmonic_angle_force(b1, b2, b3, rij[0][1], rij[1][2], box_length = 1000000.0)

			E = harmonic_angle_energy(b1, b2, b3, rij[0][1], rij[1][2], box_length = 1000000.0)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			for j in range(3):
				self.assertAlmostEqual(list(F[:3] + F[3:6] + F[6:])[j], 0.0, places = 10)

			self.assertAlmostEqual(np.dot(F[:3],  rij[0][1]), 0.0, places = 10)
			self.assertAlmostEqual(np.dot(F[6:],  rij[1][2]), 0.0, places = 10)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_noneq_fundamentals_2(self):

		ref_forces = []

		for k, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(9)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			beads_mobile = [b1, b2, b3]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = [], pointers_mobile = rij, pointers_mobile_immobile = [], F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			for j in range(3):
				self.assertAlmostEqual(list(F[:3] + F[3:6] + F[6:])[j], 0.0, places = 10)

			self.assertAlmostEqual(np.dot(F[:3],  rij[0][1]), 0.0, places = 10)
			self.assertAlmostEqual(np.dot(F[6:],  rij[1][2]), 0.0, places = 10)

			ref_forces.append(F)

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(6)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b1.mobile = False

			beads_mobile = [b2, b3]
			beads_immobile = [b1]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertAlmostEqual(np.dot(F[3:],  -rij[1][0]), 0.0, places = 10)

			self.assertSequenceEqual( list( ref_forces[j][3:6] ), list( F[:3] ) )
			self.assertSequenceEqual( list( ref_forces[j][6:] ), list( F[3:] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(6)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b2.mobile = False

			beads_mobile = [b1, b3]
			beads_immobile = [b2]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertAlmostEqual(np.dot(F[:3],  rik[0][0]), 0.0, places = 10)
			self.assertAlmostEqual(np.dot(F[3:],  -rik[1][0]), 0.0, places = 10)

			self.assertSequenceEqual( list( ref_forces[j][:3] ), list( F[:3] ) )
			self.assertSequenceEqual( list( ref_forces[j][6:] ), list( F[3:] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(6)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b3.mobile = False

			beads_mobile = [b1, b2]
			beads_immobile = [b3]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertAlmostEqual(np.dot(F[:3],  rij[0][1]), 0.0, places = 10)

			self.assertSequenceEqual( list( ref_forces[j][:3] ), list( F[:3] ) )
			self.assertSequenceEqual( list( ref_forces[j][3:6] ), list( F[3:] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(3)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b1.mobile = False
			b2.mobile = False

			beads_mobile = [b3]
			beads_immobile = [b1, b2]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertAlmostEqual(np.dot(F[:3],  -rik[0][1]), 0.0, places = 10)

			self.assertSequenceEqual( list( ref_forces[j][6:] ), list( F[:3] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(3)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b1.mobile = False
			b3.mobile = False

			beads_mobile = [b2]
			beads_immobile = [b1, b3]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertSequenceEqual( list( ref_forces[j][3:6] ), list( F[:3] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(3)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b2.mobile = False
			b3.mobile = False

			beads_mobile = [b1]
			beads_immobile = [b2, b3]

			rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
			rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2 * (np.pi / 180.0)**2, places = 10 )

			self.assertAlmostEqual(np.dot(F[:3],  rik[0][0]), 0.0, places = 10)

			self.assertSequenceEqual( list( ref_forces[j][:3] ), list( F[:3] ) )

		for j, angle in enumerate( np.linspace(0.0, 360.0, 100) ):

			F = np.zeros(0)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			b1.mobile = False
			b2.mobile = False
			b3.mobile = False

			beads_mobile = [ ]
			beads_immobile = [b1, b2, b3]

			rij = [ ]
			rik = [ ]

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

			self.assertAlmostEqual(E, 0.0, places = 10 )

			self.assertSequenceEqual( [], list( F ) )

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_against_derivative(self):

		from pyBrown.bead import angle_pbc

		box_length = 1000000.0

		dx = 0.0000001

		grid_points = [ np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]),
						np.array([2.0, 4.0, 7.0]), np.array([-np.pi, np.pi, np.pi**2]) ]

		angles = [ 30.0, 60.0, 90.0, 120.0 ]

		ks = [0.01, 1.0]

		for r1 in grid_points:

			for r2 in grid_points:

				for r3 in grid_points:

					if (r1 == r2).all() or (r1 == r3).all() or (r2 == r3).all(): continue

					for angle_eq in angles:

						for force_constant in ks:

							F = np.zeros(9)

							b1 = Bead(r1, 1.0, bead_id = 1)
							b2 = Bead(r2, 1.0, bead_id = 2)
							b3 = Bead(r3, 1.0, bead_id = 3)

							beads = [b1, b2, b3]

							rij = compute_pointer_pbc_matrix(beads, box_length = box_length)

							b1.angled_with.append([b2.bead_id, b3.bead_id])
							b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

							E = harmonic_angle_energy(b1, b2, b3, rij[0][1], rij[1][2], 1000000.0)
							F = harmonic_angle_force(b1, b2, b3, rij[0][1], rij[1][2], 1000000.0)

							Fder = np.zeros(9)

							trans = np.array([dx, 0.0, 0.0])
							Fder[0] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r-trans, b3.r-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r+trans, b3.r-b2.r, 1000000.0) ) / 2 / dx
							Fder[3] = -( harmonic_angle_energy(b1, b2, b3, b2.r+trans-b1.r, b3.r-b2.r-trans, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-trans-b1.r, b3.r-b2.r+trans, 1000000.0) ) / 2 / dx
							Fder[6] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r+trans-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r-trans-b2.r, 1000000.0) ) / 2 / dx
							trans = np.array([0.0, dx, 0.0])
							Fder[1] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r-trans, b3.r-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r+trans, b3.r-b2.r, 1000000.0) ) / 2 / dx
							Fder[4] = -( harmonic_angle_energy(b1, b2, b3, b2.r+trans-b1.r, b3.r-b2.r-trans, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-trans-b1.r, b3.r-b2.r+trans, 1000000.0) ) / 2 / dx
							Fder[7] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r+trans-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r-trans-b2.r, 1000000.0) ) / 2 / dx
							trans = np.array([0.0, 0.0, dx])
							Fder[2] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r-trans, b3.r-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r+trans, b3.r-b2.r, 1000000.0) ) / 2 / dx
							Fder[5] = -( harmonic_angle_energy(b1, b2, b3, b2.r+trans-b1.r, b3.r-b2.r-trans, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-trans-b1.r, b3.r-b2.r+trans, 1000000.0) ) / 2 / dx
							Fder[8] = -( harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r+trans-b2.r, 1000000.0) - harmonic_angle_energy(b1, b2, b3, b2.r-b1.r, b3.r-trans-b2.r, 1000000.0) ) / 2 / dx
							
							for j in range(9):

								self.assertAlmostEqual(F[j], Fder[j], places = 6)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, -self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma_1_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False

		beads_mobile = [b2]
		beads_immobile = [b1]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma_2_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b2.mobile = False

		beads_mobile = [b1]
		beads_immobile = [b2]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma_12_immobile(self):

		box_length = 1000000.0

		F = np.zeros(0)

		E = 0.0

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False
		b2.mobile = False

		beads_mobile = []
		beads_immobile = [b1, b2]

		rij = [ ]
		rik = [ ]

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_12_particles_in_sigma(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = F)
 
		self.assertEqual(E, self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -12.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 12.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_12_particles_in_sigma_1_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False

		beads_mobile = [b2]
		beads_immobile = [b1]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)
 
		self.assertEqual(E, self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 12.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_12_particles_in_sigma_2_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b2.mobile = False

		beads_mobile = [b1]
		beads_immobile = [b2]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)
 
		self.assertEqual(E, self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -12.0*self.alpha*self.epsilon/self.d])

	def test_force_and_energy_of_LJ_12_particles_in_sigma_12_immobile(self):

		box_length = 1000000.0

		F = np.zeros(0)

		E = 0.0

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False
		b2.mobile = False

		beads_mobile = []
		beads_immobile = [b1, b2]

		rij = [ ]
		rik = [ ]

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)
 
		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_in_sigma(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ia = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ir = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		Fa = np.zeros(6)

		Fr = np.zeros(6)

		Ea = Er = 0.0

		E += i.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = F)

		Ea += ia.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = Fa)

		Er += ir.compute_forces_and_energy(mobile_beads = self.beads, immobile_beads = [], pointers_mobile = self.rij, pointers_mobile_immobile = [], F = Fr)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

		self.assertSequenceEqual(list(Fa + Fr), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

	def test_force_and_energy_of_LJ_6_12_particles_in_sigma_1_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ia = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ir = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		Fa = np.zeros(3)

		Fr = np.zeros(3)

		Ea = Er = 0.0

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False

		beads_mobile = [b2]
		beads_immobile = [b1]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		Ea += ia.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fa)

		Er += ir.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fr)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

		self.assertSequenceEqual(list(Fa + Fr), [0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

	def test_force_and_energy_of_LJ_6_12_particles_in_sigma_2_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ia = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ir = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		Fa = np.zeros(3)

		Fr = np.zeros(3)

		Ea = Er = 0.0

		b1 = self.b1
		b2 = self.b2

		b2.mobile = False

		beads_mobile = [b1]
		beads_immobile = [b2]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		Ea += ia.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fa)

		Er += ir.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fr)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d])

		self.assertSequenceEqual(list(Fa + Fr), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d])

	def test_force_and_energy_of_LJ_6_12_particles_in_sigma_12_immobile(self):

		box_length = 1000000.0

		F = np.zeros(0)

		E = 0.0

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ia = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		ir = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		Fa = np.zeros(0)

		Fr = np.zeros(0)

		Ea = Er = 0.0

		b1 = self.b1
		b2 = self.b2

		b1.mobile = False
		b2.mobile = False

		beads_mobile = [ ]
		beads_immobile = [b1, b2]

		rij = [ ]
		rik = [ ]

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		Ea += ia.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fa)

		Er += ir.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = Fr)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

		self.assertSequenceEqual(list(Fa + Fr), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_at_minimum(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		b3 = Bead([2*2.0**(1/6), 0.0, 0.0], 1.0, epsilon_LJ = self.epsilon)

		beads = [self.b1, b3]

		rij = compute_pointer_pbc_matrix(beads, box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads, immobile_beads = [], pointers_mobile = rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, -self.epsilon)

		for j in range(6):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_at_minimum_1_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		b1 = self.b1
		b3 = Bead([2*2.0**(1/6), 0.0, 0.0], 1.0, epsilon_LJ = self.epsilon)

		b1.mobile = False

		beads_mobile = [b3]
		beads_immobile = [b1]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -self.epsilon)

		for j in range(3):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_at_minimum_2_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		b1 = self.b1
		b3 = Bead([2*2.0**(1/6), 0.0, 0.0], 1.0, epsilon_LJ = self.epsilon)

		b3.mobile = False

		beads_mobile = [b1]
		beads_immobile = [b3]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -self.epsilon)

		for j in range(3):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_at_minimum_12_immobile(self):

		box_length = 1000000.0

		F = np.zeros(0)

		E = 0.0

		b1 = self.b1
		b3 = Bead([2*2.0**(1/6), 0.0, 0.0], 1.0, epsilon_LJ = self.epsilon)

		b1.mobile = False
		b3.mobile = False

		beads_mobile = [ ]
		beads_immobile = [b1, b3]

		rij = [ ]
		rik = [ ]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle(self):

		box_length = 1000000.0

		F = np.zeros(9)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		beads = [ bead4, bead5, bead6 ]

		rij = compute_pointer_pbc_matrix(beads, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads, immobile_beads = [], pointers_mobile = rij, pointers_mobile_immobile = [], F = F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(9):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_1_immobile(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False

		beads_mobile = [ bead5, bead6 ]
		beads_immobile = [ bead4 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(6):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_2_immobile(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5.mobile = False

		beads_mobile = [ bead4, bead6 ]
		beads_immobile = [ bead5 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(6):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_3_immobile(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False

		beads_mobile = [ bead4, bead5 ]
		beads_immobile = [ bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(6):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_12_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead5.mobile = False

		beads_mobile = [ bead6 ]
		beads_immobile = [ bead4, bead5 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -2*self.epsilon)

		for j in range(3):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_13_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead6.mobile = False

		beads_mobile = [ bead5 ]
		beads_immobile = [ bead4, bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -2*self.epsilon)

		for j in range(3):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_23_immobile(self):

		box_length = 1000000.0

		F = np.zeros(3)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5.mobile = False
		bead6.mobile = False

		beads_mobile = [ bead4 ]
		beads_immobile = [ bead5, bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -2*self.epsilon)

		for j in range(3):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_on_stable_equilateral_triangle_123_immobile(self):

		box_length = 1000000.0

		F = np.zeros(0)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead5.mobile = False
		bead6.mobile = False

		beads_mobile = [ ]
		beads_immobile = [ bead4, bead5, bead6 ]

		rij = [ ]
		rik = [ ]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [])

	#---------------------------------------------------------------------------

	def test_energy_of_LJ_6_12_particles_on_unstable_equilateral_triangle(self):

		box_length = 1000000.0

		A = 2 / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		F_ref = np.zeros(9)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		beads_mobile = [ bead4, bead5, bead6 ]
		beads_immobile = []

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = [ ]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F_ref)

		self.assertAlmostEqual(E, 0.0, places = 7)

		F = np.zeros(6)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False

		beads_mobile = [ bead5, bead6 ]
		beads_immobile = [ bead4 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[3:]))

		F = np.zeros(6)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5.mobile = False

		beads_mobile = [ bead4, bead6 ]
		beads_immobile = [ bead5 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref)[:3]+list(F_ref)[6:])

		F = np.zeros(6)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead6.mobile = False

		beads_mobile = [ bead4, bead5 ]
		beads_immobile = [ bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[:6]))

		F = np.zeros(3)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead5.mobile = False

		beads_mobile = [ bead6 ]
		beads_immobile = [ bead4, bead5 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[6:]))

		F = np.zeros(3)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead6.mobile = False

		beads_mobile = [ bead5 ]
		beads_immobile = [ bead4, bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[3:6]))

		F = np.zeros(3)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5.mobile = False
		bead6.mobile = False

		beads_mobile = [ bead4 ]
		beads_immobile = [ bead5, bead6 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[:3]))

		F = np.zeros(0)

		E = 0.0

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)
		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		bead4.mobile = False
		bead5.mobile = False
		bead6.mobile = False

		beads_mobile = [ ]
		beads_immobile = [ bead4, bead5, bead6 ]

		rij = [ ]
		rik = [ ]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertSequenceEqual(list(F), list(F_ref[0:0]))

	#---------------------------------------------------------------------------

	def test_energy_of_LJ_particles_with_dummy_immobile_particles(self):

		box_length = 1000000.0

		F = np.zeros(9)

		E = 0.0

		A = 2*2.0**(1/6) / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)

		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		dummy1 = Bead([5.0, 5.0, 0.0], 1.0, epsilon_LJ = 0.0, mobile = False)

		dummy2 = Bead([3.0, 3.0, 0.0], 1.0, epsilon_LJ = 0.0, mobile = False)

		dummy3 = Bead([0.0, 0.0, 0.0], 1.0, epsilon_LJ = 0.0, mobile = False)

		beads_mobile = [ bead4, bead5, bead6 ]
		beads_immobile = [ dummy1, dummy2, dummy3 ]

		rij = compute_pointer_pbc_matrix(beads_mobile, box_length = box_length)
		rik = compute_pointer_immobile_pbc_matrix(beads_mobile, beads_immobile, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(mobile_beads = beads_mobile, immobile_beads = beads_immobile, pointers_mobile = rij, pointers_mobile_immobile = rik, F = F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(9):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_setting_custom_interactions(self):

		input_data = {"custom_interactions": True,
					  "custom_interactions_filename": 'foo.py',
					  "auxiliary_custom_interactions_keywords": {"test_parameter": "test_value"}}

		interactions = []

		_set_custom_interactions(input_data, interactions)

		print(interactions)

		self.assertEqual(interactions[0].auxiliary_force_parameters["test_parameter"], "test_value")
		self.assertEqual(interactions[1].auxiliary_force_parameters["test_parameter"], "test_value")

		F = np.zeros(6)
		
		E = 0

		E += interactions[0].compute_forces_and_energy(self.beads, self.rij, F)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------