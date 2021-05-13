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

from pyBrown.bead import Bead, pointer_pbc
from pyBrown.diffusion import JO_R_lubrication_correction_matrix

#-------------------------------------------------------------------------------

class TestDiffusion(unittest.TestCase):

	def test_R_lub_corr_symmetry(self):

		box_length = 30.0

		lubrication_cutoff = 10.0

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_1 = [ [ pointer_pbc(bi, bj, box_length) for bj in beads_1 ] for bi in beads_1 ]

		beads_2 = [ Bead([4.0, 0.0, 0.0], 2.0), Bead([0.0, 0.0, 0.0], 1.0) ]
		pointers_2 = [ [ pointer_pbc(bi, bj, box_length) for bj in beads_2 ] for bi in beads_2 ]

		R_1 = JO_R_lubrication_correction_matrix(beads_1, pointers_1, lubrication_cutoff)
		R_2 = JO_R_lubrication_correction_matrix(beads_2, pointers_2, lubrication_cutoff)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

		beads_1 = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0), Bead([10.0, 0.0, 0.0], 3.0) ]
		pointers_1 = [ [ pointer_pbc(bi, bj, box_length) for bj in beads_1 ] for bi in beads_1 ]

		beads_2 = [ Bead([10.0, 0.0, 0.0], 3.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([4.0, 0.0, 0.0], 2.0) ]
		pointers_2 = [ [ pointer_pbc(bi, bj, box_length) for bj in beads_2 ] for bi in beads_2 ]

		R_1 = JO_R_lubrication_correction_matrix(beads_1, pointers_1, lubrication_cutoff)
		R_2 = JO_R_lubrication_correction_matrix(beads_2, pointers_2, lubrication_cutoff)

		N = 3*len(beads_1)

		for i in range(N):
			for j in range(N):
				self.assertAlmostEqual(R_1[i][j], R_2[(i+3)%N][(j+3)%N], places = 7)

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------