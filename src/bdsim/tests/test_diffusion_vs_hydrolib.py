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
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from pyBrown.bead import Bead, compute_pointer_pbc_matrix
from pyBrown.diffusion import JO_2B_R_matrix, RPY_M_matrix

#-------------------------------------------------------------------------------

class TestDiffusionVsHydrolib(unittest.TestCase):

	def setUp(self):

		self.R_hydrolib_infty = 2.7279079034593843

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([3000.0, 0.0, 0.0], 1.0)
		beads = [ beadi, beadj ]
		self.R_pybrown_infty = np.linalg.inv( RPY_M_matrix(beads, compute_pointer_pbc_matrix(beads, 1000000.0)) )[1][1]

	#-------------------------------------------------------------------------------

	def test_JO_R2B_3(self):

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([3.0, 0.0, 0.0], 1.0)

		R_pybrown = JO_2B_R_matrix(beadi, beadj) / self.R_pybrown_infty

		R_hydrolib_3 = np.array( [ [ 3.7312230406281288, 0.0, 0.0, -1.8263361553120150, 0.0, 0.0 ],
			   			   		   [ 0.0, 2.9791344182689072, 0.0, 0.0, -0.81066685570688679, 0.0 ],
			   			   		   [ 0.0, 0.0, 2.9791344182348460, 0.0, 0.0, -0.81066685563897056 ],
			   			   		   [ -1.8263361553120159, 0.0, 0.0, 3.7312230406281275, 0.0, 0.0 ],
			   			   		   [ 0.0, -0.81066685570688635, 0.0, 0.0, 2.9791344182689063, 0.0 ],
			   			   		   [ 0.0, 0.0, -0.81066685563897056, 0.0, 0.0, 2.9791344182348469 ] ] )

		R_hydrolib_3 /= self.R_hydrolib_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_hydrolib_3[i][j], delta = 0.001)

	#-------------------------------------------------------------------------------

	def test_JO_R2B_2_5(self):

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([2.5, 0.0, 0.0], 1.0)

		R_pybrown = JO_2B_R_matrix(beadi, beadj) / self.R_pybrown_infty

		R_hydrolib_2_5 = np.array( [ [ 4.6992346984090050, 0.0, 0.0, -2.8635820567271963, 0.0, 0.0 ],
  								     [ 0.0, 3.1776367039374520, 0.0, 0.0, -1.0940991803244664, 0.0 ],
								     [ 0.0, 0.0, 3.1776367038873845, 0.0, 0.0, -1.0940991802409445 ],
									 [ -2.8635820567271959, 0.0, 0.0, 4.6992346984090059, 0.0, 0.0 ],
									 [ 0.0, -1.0940991803244662, 0.0, 0.0, 3.1776367039374511, 0.0 ],
									 [ 0.0, 0.0, -1.0940991802409441, 0.0, 0.0, 3.1776367038873850 ] ] )

		R_hydrolib_2_5 /= self.R_hydrolib_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_hydrolib_2_5[i][j], delta = 0.001)

	#-------------------------------------------------------------------------------

	def test_JO_R2B_2_1(self):

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([2.1, 0.0, 0.0], 1.0)

		R_pybrown = JO_2B_R_matrix(beadi, beadj) / self.R_pybrown_infty

		R_hydrolib_2_1 = np.array( [ [ 10.998747953251131, 0.0, 0.0, -9.2231504639498034, 0.0, 0.0 ],
									 [ 0.0, 3.7998794346119538, 0.0, 0.0, -1.7999650596806469, 0.0 ],
									 [ 0.0, 0.0, 3.7998794345686240, 0.0, 0.0, -1.7999650596075831 ],
									 [ -9.2231504639498034, 0.0, 0.0, 10.998747953251129, 0.0, 0.0 ],
									 [ 0.0, -1.7999650596806469, 0.0, 0.0, 3.7998794346119538, 0.0 ],
									 [ 0.0, 0.0, -1.7999650596075829, 0.0, 0.0, 3.7998794345686235 ] ] )

		R_hydrolib_2_1 /= self.R_hydrolib_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_hydrolib_2_1[i][j], delta = 0.002)

	#-------------------------------------------------------------------------------

	def test_JO_R2B_2_01(self):

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([2.01, 0.0, 0.0], 1.0)

		R_pybrown = JO_2B_R_matrix(beadi, beadj) / self.R_pybrown_infty

		R_hydrolib_2_01 = np.array( [ [ 73.745617583400403, 0.0, 0.0, -71.984145115914345, 0.0, 0.0 ],
									  [ 0.0, 4.8195602732507634, 0.0, 0.0, -2.8404032182622507, 0.0 ],
									  [ 0.0, 0.0, 4.8195602733199330, 0.0, 0.0, -2.8404032183035017 ],
									  [ -71.984145115914345, 0.0, 0.0, 73.745617583400403, 0.0, 0.0 ],
									  [ 0.0, -2.8404032182622507, 0.0, 0.0, 4.8195602732507634, 0.0 ],
									  [ 0.0, 0.0, -2.8404032183035013, 0.0, 0.0, 4.8195602733199330 ] ] )

		R_hydrolib_2_01 /= self.R_hydrolib_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_hydrolib_2_01[i][j], delta = 0.005)

	#-------------------------------------------------------------------------------

	def test_JO_R2B_2_001(self):

		beadi = Bead([0.0, 0.0, 0.0], 1.0)
		beadj = Bead([2.001, 0.0, 0.0], 1.0)

		R_pybrown = JO_2B_R_matrix(beadi, beadj) / self.R_pybrown_infty

		R_hydrolib_2_001 = np.array( [ [ 688.93245092110328, 0.0, 0.0, -687.17240456038189, 0.0, 0.0 ],
									   [ 0.0, 5.8638551300400064, 0.0, 0.0, -3.8868109413208707, 0.0 ],
									   [ 0.0, 0.0, 5.8638551301406192, 0.0, 0.0, -3.8868109413937710 ],
									   [ -687.17240456038189, 0.0, 0.0, 688.93245092110328, 0.0, 0.0 ],
									   [ 0.0, -3.8868109413208707, 0.0, 0.0, 5.8638551300400072, 0.0 ],
									   [ 0.0, 0.0, -3.8868109413937701, 0.0, 0.0, 5.8638551301406219 ] ] )

		R_hydrolib_2_001 /= self.R_hydrolib_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_hydrolib_2_001[i][j], delta = 0.005)

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------