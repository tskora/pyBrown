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
import os.path
import sys

import unittest

from pybrown.bead import Bead, compute_pointer_pbc_matrix
from pybrown.diffusion import RPY_Smith_M_matrix

#-------------------------------------------------------------------------------

class TestDiffusionVsHydrolib(unittest.TestCase):

	def setUp(self):

		pass

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_equal_size_dist100_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 100.0], 51.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_e = 4.209804e-03
		D_bdbox_diag_r = 4.209804e-03
		D_bdbox_coupling_ee = 1.498395e-03
		D_bdbox_coupling_rr = 2.286797e-03

		# print(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0])
		# print(D_bdbox_coupling_ee/D_bdbox_diag_e)

		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_e, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_e, delta = 0.0001)

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_equal_size_dist150_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 150.0], 51.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_e = 4.209804e-03
		D_bdbox_diag_r = 4.209804e-03
		D_bdbox_coupling_ee = 7.775705e-04
		D_bdbox_coupling_rr = 1.639596e-03

		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_e, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_e, delta = 0.0001)

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_equal_size_dist200_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 200.0], 51.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_e = 4.209804e-03
		D_bdbox_diag_r = 4.209804e-03
		D_bdbox_coupling_ee = 4.751021e-04
		D_bdbox_coupling_rr = 1.242301e-03

		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_e, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_e, delta = 0.0001)

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_diff_size_dist100_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 100.0], 10.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_1 = 4.209804e-03
		D_bdbox_diag_2 = 2.138692e-02
		D_bdbox_coupling_ee = 1.367394e-03
		D_bdbox_coupling_rr = 2.555235e-03 

		self.assertAlmostEqual(M_pybrown_rpy[0][0]/M_pybrown_rpy[3][3], D_bdbox_diag_1/D_bdbox_diag_2, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_1, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_1, delta = 0.0001)

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_diff_size_dist60_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 60.0], 10.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_1 = 4.209804e-03
		D_bdbox_diag_2 = 2.138692e-02
		D_bdbox_coupling_ee = 2.952219e-03
		D_bdbox_coupling_rr = 3.628254e-03

		self.assertAlmostEqual(M_pybrown_rpy[0][0]/M_pybrown_rpy[3][3], D_bdbox_diag_1/D_bdbox_diag_2, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_1, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_1, delta = 0.0001)

	#-------------------------------------------------------------------------------

	def test_RPY_Smith_diff_size_dist40_bdbox(self):

		box_size = 750.0

		beads = [ Bead([0.0, 0.0, 0.0], 51.0), Bead([0.0, 0.0, 40.0], 10.0) ]
		pointers = compute_pointer_pbc_matrix(beads, box_size)

		M_pybrown_rpy = RPY_Smith_M_matrix(beads = beads, pointers = pointers, box_length = box_size, alpha = np.sqrt(np.pi), m = 2, n = 2)

		D_bdbox_diag_1 = 4.209804e-03
		D_bdbox_diag_2 = 2.138692e-02
		D_bdbox_coupling_ee = 3.803894e-03
		D_bdbox_coupling_rr = 3.807062e-03

		self.assertAlmostEqual(M_pybrown_rpy[0][0]/M_pybrown_rpy[3][3], D_bdbox_diag_1/D_bdbox_diag_2, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[3][0]/M_pybrown_rpy[0][0], D_bdbox_coupling_ee/D_bdbox_diag_1, delta = 0.0001)
		self.assertAlmostEqual(M_pybrown_rpy[5][2]/M_pybrown_rpy[0][0], D_bdbox_coupling_rr/D_bdbox_diag_1, delta = 0.0001)

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------