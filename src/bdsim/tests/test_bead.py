# pyBD is a Brownian and Stokesian dynamics simulation tool
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
import copy as cp
import numpy as np
import sys
sys.path.insert(0, '.')
import unittest

from pyBrown.bead import Bead, overlap_pbc, distance_pbc

#-------------------------------------------------------------------------------

class TestBead(unittest.TestCase):

	def setUp(self):
		
		self.b = Bead([0.0, 0.0, 0.0], 1.0)

		self.b1 = Bead([0.1, 0.1, 0.1], 1.0)

		self.b2 = Bead([10.0, 10.0, 10.0], 1.0)

	#---------------------------------------------------------------------------

	def test_overlap(self):

		self.assertTrue( overlap_pbc(self.b, self.b1, 10.0) )

		self.assertFalse( overlap_pbc(self.b, self.b2, 100.0) )

		self.assertTrue( overlap_pbc(self.b, self.b2, 10.0) )

		self.assertTrue( overlap_pbc(self.b, self.b2, 2.0) )

	#---------------------------------------------------------------------------

	def test_distance(self):

		self.assertAlmostEqual( distance_pbc(self.b, self.b, 10.0), 0.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b1, self.b1, 10.0), 0.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b2, self.b2, 10.0), 0.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b1, 10.0), np.sqrt(3)*0.1, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b1, 10000.0), np.sqrt(3)*0.1, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b2, 100.0), np.sqrt(3)*10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b2, 20.0), np.sqrt(3)*10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b2, 11.0), np.sqrt(3), places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, self.b2, 10.0), 0.0, places = 7 )

	#---------------------------------------------------------------------------

	def test_distance_2(self):

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, 10.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, -10.0], 1.0), 100.0), 10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, 10.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, -10.0], 1.0), 15.0), 5.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, 10.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 0.0, -10.0], 1.0), 10.0), 0.0, places = 7 )

	#---------------------------------------------------------------------------

	def test_distance_3(self):

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, -10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, -10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, -10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, -10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, -10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, -10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([10.0, 0.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([-10.0, 0.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, 10.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b, Bead([0.0, -10.0, -10.0], 1.0), 5.0), 0.0, places = 7 )

	def test_translate_and_flux(self):

		b1 = Bead([-1.0, 0.0, 0.0], 0.0)

		vec = np.array([0.6, 0.0, 0.0])

		normal = np.array([ 1.0, 0.0, 0.0 ])

		normal_point = np.zeros(3)

		self.assertEqual( b1.translate_and_return_flux(vec, normal, normal_point), 0 )

		self.assertEqual( b1.translate_and_return_flux(vec, normal, normal_point), 1 )

		vec = np.array([0.0, 20.0, 0.0])

		self.assertEqual( b1.translate_and_return_flux(vec, normal, normal_point), 0 )

		normal = np.array([ 0.0, 1.0, 0.0 ])

		normal_point = np.array([0.0, 25.0, 0.0])

		self.assertEqual( b1.translate_and_return_flux(vec, normal, normal_point), 1 )

		vec = np.array([0.0, -400.0, 0.0])

		self.assertEqual( b1.translate_and_return_flux(vec, normal, normal_point), -1 )

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------