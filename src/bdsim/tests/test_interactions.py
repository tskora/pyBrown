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

from pyBrown.bead import Bead, pointer, compute_pointer_pbc_matrix
from pyBrown.interactions import Interactions, LJ_6_attractive_energy, LJ_6_attractive_force,\
								 LJ_12_repulsive_energy, LJ_12_repulsive_force,\
								 LJ_6_12_energy, LJ_6_12_force, harmonic_bond_force, \
								 harmonic_bond_energy, harmonic_angle_force, harmonic_angle_energy

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

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

		# self.b2.bonded_with.append(self.b1.bead_id)
		# self.b2.bonded_how[self.b1.bead_id] = [ dist_eq, force_constant ]

		i = Interactions(harmonic_bond_force, harmonic_bond_energy, bonded = True, how_many_body = 2)

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, force_constant*(self.d - dist_eq), 0.0, 0.0, -force_constant*(self.d - dist_eq)])

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

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, 0.5*force_constant*(dist_eq - self.d)**2)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -force_constant*(dist_eq - self.d), 0.0, 0.0, force_constant*(dist_eq - self.d)])

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

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

			self.assertEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2)

			self.assertSequenceEqual(list(F[:3] + F[3:6] + F[6:]), [0.0, 0.0, 0.0])

			self.assertAlmostEqual(np.dot(F[:3],  rij[0][1]), 0.0, places = 10)
			self.assertAlmostEqual(np.dot(F[6:],  rij[1][2]), 0.0, places = 10)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_angle_noneq_fundamentals_2(self):

		for angle in np.linspace(0.0, 360.0, 100):

			F = np.zeros(9)

			angle_eq = angle

			force_constant = 1.0

			box_length = 1000000.0

			b1 = Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1)
			b2 = Bead([0.5, 0.0, 0.5], 1.0, bead_id = 2)
			b3 = Bead([1.0, 0.0, 0.0], 1.0, bead_id = 3)

			beads = [b1, b2, b3]

			rij = compute_pointer_pbc_matrix(beads, box_length = box_length)

			b1.angled_with.append([b2.bead_id, b3.bead_id])
			b1.angled_how[(b2.bead_id, b3.bead_id)] = [ angle_eq, force_constant ]

			i = Interactions(harmonic_angle_force, harmonic_angle_energy, auxiliary_force_parameters = {"box_length": box_length}, bonded = True, how_many_body = 3)

			E = i.compute_forces_and_energy(beads, rij, F)

			self.assertEqual(E, 0.5 * force_constant * (angle_eq - 90.0)**2)

			self.assertSequenceEqual(list(F[:3] + F[3:6] + F[6:]), [0.0, 0.0, 0.0])

			self.assertAlmostEqual(np.dot(F[:3],  rij[0][1]), 0.0, places = 10)
			self.assertAlmostEqual(np.dot(F[6:],  rij[1][2]), 0.0, places = 10)

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_particles_in_sigma(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		i = Interactions(LJ_6_attractive_force, LJ_6_attractive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, -self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_12_particles_in_sigma(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		i = Interactions(LJ_12_repulsive_force, LJ_12_repulsive_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		self.assertEqual(E, self.alpha*self.epsilon)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -12.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 12.0*self.alpha*self.epsilon/self.d])

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

		E += i.compute_forces_and_energy(self.beads, self.rij, F)

		Ea += ia.compute_forces_and_energy(self.beads, self.rij, Fa)

		Er += ir.compute_forces_and_energy(self.beads, self.rij, Fr)

		self.assertEqual(Ea + Er, 0.0)

		self.assertEqual(E, 0.0)

		self.assertSequenceEqual(list(F), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

		self.assertSequenceEqual(list(Fa + Fr), [0.0, 0.0, -6.0*self.alpha*self.epsilon/self.d, 0.0, 0.0, 6.0*self.alpha*self.epsilon/self.d])

	#---------------------------------------------------------------------------

	def test_force_and_energy_of_LJ_6_12_particles_at_minimum(self):

		box_length = 1000000.0

		F = np.zeros(6)

		E = 0.0

		b3 = Bead([2*2.0**(1/6), 0.0, 0.0], 1.0, epsilon_LJ = self.epsilon)

		beads = [self.b1, b3]

		rij = np.zeros((len(beads), len(beads), 3))

		for i in range(1, len(beads)):
			for j in range(0, i):
				rij[i][j] = pointer(beads[i], beads[j])
				rij[j][i] = -rij[i][j]

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(beads, rij, F)

		self.assertEqual(E, -self.epsilon)

		for j in range(6):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

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

		E += i.compute_forces_and_energy(beads, rij, F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(9):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

	#---------------------------------------------------------------------------

	def test_energy_of_LJ_6_12_particles_on_unstable_equilateral_triangle(self):

		box_length = 1000000.0

		F = np.zeros(9)

		E = 0.0

		A = 2 / ( np.cos(5*np.pi/6) - np.cos(np.pi/6) )

		bead4 = Bead([A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)

		bead5 = Bead([A*np.cos(5*np.pi/6), A*np.sin(5*np.pi/6), 0.0], 1.0, epsilon_LJ = 1.0)

		bead6 = Bead([A*np.cos(3*np.pi/2), A*np.sin(3*np.pi/2), 0.0], 1.0, epsilon_LJ = 1.0)

		beads = [ bead4, bead5, bead6 ]

		rij = compute_pointer_pbc_matrix(beads, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(beads, rij, F)

		self.assertAlmostEqual(E, 0.0, places = 7)

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

		beads = [ dummy1, bead4, dummy2, bead5, dummy3, bead6 ]

		rij = compute_pointer_pbc_matrix(beads, box_length = box_length)

		i = Interactions(LJ_6_12_force, LJ_6_12_energy, auxiliary_force_parameters = {"lennard_jones_alpha": self.alpha}, bonded = False, how_many_body = 2)

		E += i.compute_forces_and_energy(beads, rij, F)

		self.assertEqual(E, -3*self.epsilon)

		for j in range(9):

			self.assertAlmostEqual(F[j], 0.0, places = 7)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------