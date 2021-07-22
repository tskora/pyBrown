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
import os.path
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from pyBrown.bead import Bead, compute_pointer_pbc_matrix
from pyBrown.diffusion import RPY_Smith_M_matrix, JO_R_lubrication_correction_F_matrix

#-------------------------------------------------------------------------------

class TestDiffusion(unittest.TestCase):

	def test_R_lub_corr_symmetry_cichocki(self):

		box_length = 30.0

		lubrication_cutoff = 10.0
		cichocki_correction = True

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([4.0, 0.0, 0.0], 2.0), Bead([0.0, 0.0, 0.0], 1.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		R_1 = JO_R_lubrication_correction_F_matrix(beads_1, pointers_1, lubrication_cutoff, cichocki_correction)
		R_2 = JO_R_lubrication_correction_F_matrix(beads_2, pointers_2, lubrication_cutoff, cichocki_correction)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0), Bead([10.0, 0.0, 0.0], 3.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([10.0, 0.0, 0.0], 3.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		R_1 = JO_R_lubrication_correction_F_matrix(beads_1, pointers_1, lubrication_cutoff, cichocki_correction)
		R_2 = JO_R_lubrication_correction_F_matrix(beads_2, pointers_2, lubrication_cutoff, cichocki_correction)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

	#-------------------------------------------------------------------------------

	def test_R_lub_corr_symmetry_no_cichocki(self):

		box_length = 30.0

		lubrication_cutoff = 10.0
		cichocki_correction = False

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([4.0, 0.0, 0.0], 2.0), Bead([0.0, 0.0, 0.0], 1.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		R_1 = JO_R_lubrication_correction_F_matrix(beads_1, pointers_1, lubrication_cutoff, cichocki_correction)
		R_2 = JO_R_lubrication_correction_F_matrix(beads_2, pointers_2, lubrication_cutoff, cichocki_correction)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0), Bead([10.0, 0.0, 0.0], 3.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([10.0, 0.0, 0.0], 3.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		R_1 = JO_R_lubrication_correction_F_matrix(beads_1, pointers_1, lubrication_cutoff, cichocki_correction)
		R_2 = JO_R_lubrication_correction_F_matrix(beads_2, pointers_2, lubrication_cutoff, cichocki_correction)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

	#-------------------------------------------------------------------------------

	def test_M_RPY_Smith_symmetry(self):

		box_length = 30.0
		alpha = np.sqrt(np.pi)
		m = 2
		n = 2

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([4.0, 0.0, 0.0], 2.0), Bead([0.0, 0.0, 0.0], 1.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		M_1 = RPY_Smith_M_matrix(beads_1, pointers_1, box_length, alpha, m, n)
		M_2 = RPY_Smith_M_matrix(beads_2, pointers_2, box_length, alpha, m, n)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(M_1[i][j], M_2[(i+3)%N][(j+3)%N], places = 7)

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0), Bead([10.0, 0.0, 0.0], 3.0) ]
		pointers_1 = compute_pointer_pbc_matrix(beads_1, box_length)

		beads_2 = [ Bead([10.0, 0.0, 0.0], 3.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_2 = compute_pointer_pbc_matrix(beads_2, box_length)

		M_1 = RPY_Smith_M_matrix(beads_1, pointers_1, box_length, alpha, m, n)
		M_2 = RPY_Smith_M_matrix(beads_2, pointers_2, box_length, alpha, m, n)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(M_1[i][j], M_2[(i+3)%N][(j+3)%N], places = 7)		

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------