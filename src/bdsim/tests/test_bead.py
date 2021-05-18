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
import copy as cp
import numpy as np
import os.path
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from pyBrown.bead import Bead, overlap, overlap_pbc, distance, distance_pbc, pointer_pbc, compute_pointer_pbc_matrix, check_overlaps

#-------------------------------------------------------------------------------

def compute_pointer_pbc_matrix_python(beads, box_length):

	rij = np.zeros((len(beads), len(beads), 3))

	for i in range(1, len(beads)):
		for j in range(0, i):
			rij[i][j] = pointer_pbc(beads[i], beads[j], box_length)
			rij[j][i] = -rij[i][j]

	return rij

#-------------------------------------------------------------------------------

def check_overlaps_python(beads, box_length, overlap_treshold):

	overlaps = False

	for i in range(len(beads)-1):
		for j in range(i+1, len(beads)):
			pointer = beads[i].r - beads[j].r
			radii_sum = beads[i].a + beads[j].a
			radii_sum_pbc = box_length - radii_sum

			if ( pointer[0] > radii_sum and pointer[0] < radii_sum_pbc ) or ( pointer[0] < -radii_sum and pointer[0] > -radii_sum_pbc ):
				continue
			elif ( pointer[1] > radii_sum and pointer[1] < radii_sum_pbc ) or ( pointer[1] < -radii_sum and pointer[1] > -radii_sum_pbc ):
				continue
			elif ( pointer[2] > radii_sum and pointer[2] < radii_sum_pbc ) or ( pointer[2] < -radii_sum and pointer[2] > -radii_sum_pbc ):
				continue
			else:
				if overlap_pbc(beads[i], beads[j], box_length, overlap_treshold):
					return True

	return overlaps

#-------------------------------------------------------------------------------

class TestBead(unittest.TestCase):

	def setUp(self):
		
		self.b0 = Bead([0.0, 0.0, 0.0], 1.0)

		self.b1 = Bead([0.1, 0.1, 0.1], 1.0)

		self.b2 = Bead([10.0, 10.0, 10.0], 1.0)

		self.b3 = Bead([2.0, 0.0, 0.0], 1.0)

		self.b4 = Bead([4.0, 0.0, 0.0], 3.0)

	#---------------------------------------------------------------------------

	def test_if_translate_bead_by_zero_vector_makes_no_change(self):

		x0, y0, z0 = self.b0.r

		self.b0.translate([0.0, 0.0, 0.0])

		self.assertSequenceEqual([x0, y0, z0], list(self.b0.r))

	#---------------------------------------------------------------------------

	def test_if_translate_bead_by_unit_vector_x_is_correct(self):

		x0, y0, z0 = self.b0.r

		self.b0.translate([1.0, 0.0, 0.0])

		self.assertSequenceEqual([x0+1.0, y0, z0], list(self.b0.r))

	#---------------------------------------------------------------------------

	def test_if_translate_bead_by_unit_vector_y_is_correct(self):

		x0, y0, z0 = self.b0.r

		self.b0.translate([0.0, 1.0, 0.0])

		self.assertSequenceEqual([x0, y0+1.0, z0], list(self.b0.r))

	#---------------------------------------------------------------------------

	def test_if_translate_bead_by_unit_vector_z_is_correct(self):

		x0, y0, z0 = self.b0.r

		self.b0.translate([0.0, 0.0, 1.0])

		self.assertSequenceEqual([x0, y0, z0+1.0], list(self.b0.r))

	#---------------------------------------------------------------------------

	def test_if_staying_at_xy_plaen_results_in_zero_flux(self):

		normal = [0.0, 0.0, 1.0]

		plane_point = [0.0, 0.0, 0.0]

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 0.0], normal, plane_point), 0 )

	#---------------------------------------------------------------------------

	def test_if_going_through_xy_parallel_plane_results_in_positive_flux(self):

		normal = [0.0, 0.0, 1.0]

		plane_point = [0.0, 0.0, 1.0]

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 2.0], normal, plane_point), 1 )

	#---------------------------------------------------------------------------

	def test_if_going_through_xy_antiparallel_plane_results_in_negative_flux(self):

		normal = [0.0, 0.0, -1.0]

		plane_point = [0.0, 0.0, 1.0]

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 2.0], normal, plane_point), -1 )

	#---------------------------------------------------------------------------

	def test_if_going_through_yz_plane_in_two_steps_results_in_positive_flux(self):

		b_start = Bead([-1.0, 0.0, 0.0], 0.0)

		translation_vector = np.array([0.6, 0.0, 0.0])

		normal = [1.0, 0.0, 0.0]

		plane_point = [0.0, 0.0, 0.0]

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, normal, plane_point), 0 )

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, normal, plane_point), 1 )

	#---------------------------------------------------------------------------

	def test_if_going_parallel_to_plane_does_not_produce_flux(self):

		b_start = Bead([-1.0, 0.0, 0.0], 0.0)

		translation_vector = np.array([0.0, 20.0, 0.0])

		normal = [1.0, 0.0, 0.0]

		plane_point = [0.0, 0.0, 0.0]

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, normal, plane_point), 0 )

	#---------------------------------------------------------------------------

	def test_if_bead_overlaps_with_itself(self):

		self.assertTrue( overlap(self.b0, self.b0) )

	#---------------------------------------------------------------------------

	def test_if_bead_overlaps_with_itself_in_large_box(self):

		self.assertTrue( overlap_pbc(self.b1, self.b1, 10000.0) )

	#---------------------------------------------------------------------------

	def test_if_bead_overlaps_with_a_close_one_in_large_box(self):

		self.assertTrue( overlap_pbc(self.b0, self.b1, 10000.0) )

	#---------------------------------------------------------------------------

	def tesT_if_bead_overlaps_with_its_near_replica(self):

		self.assertTrue( overlap_pbc(self.b0, self.b2, 10.0) )

	#---------------------------------------------------------------------------

	def test_if_bead_oberlaps_with_its_fr_replica(self):

		self.assertTrue( overlap_pbc(self.b0, self.b2, 0.1) )

	#---------------------------------------------------------------------------

	def test_if_bead_does_not_overlap_with_separate_one(self):

		self.assertFalse( overlap_pbc(self.b0, self.b2, 100.0) )

	#---------------------------------------------------------------------------

	def test_if_beads_overlap_when_they_touch_each_other_in_large_box(self):

		self.assertTrue( overlap_pbc(self.b0, self.b3, 10000.0) )

	#---------------------------------------------------------------------------

	def test_distance_between_bead_and_itself(self):

		self.assertEqual( distance(self.b0, self.b0), 0.0 )

		self.assertEqual( distance(self.b1, self.b1), 0.0 )

		self.assertEqual( distance(self.b2, self.b2), 0.0 )

		self.assertEqual( distance(self.b3, self.b3), 0.0 )

		self.assertEqual( distance(self.b4, self.b4), 0.0 )

	#---------------------------------------------------------------------------

	def test_distance_between_bead_and_its_near_replica(self):

		self.assertEqual( distance_pbc(self.b0, self.b2, 10.0), 0.0 )

	#---------------------------------------------------------------------------

	def test_distance_between_bead_and_its_far_replica(self):

		self.assertAlmostEqual( distance_pbc(self.b0, self.b2, 0.1), 0.0, places = 7 )

	#---------------------------------------------------------------------------

	def test_distance_between_bead_and_bead_near_to_its_near_replica(self):

		self.assertAlmostEqual( distance_pbc(self.b0, self.b2, 11.0), np.sqrt(3), places = 7 )

	#---------------------------------------------------------------------------

	def test_distance_between_near_beads(self):

		self.assertAlmostEqual( distance(self.b0, self.b1), np.sqrt(3)*0.1, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, self.b1, 10.0), np.sqrt(3)*0.1, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, self.b1, 10000.0), np.sqrt(3)*0.1, places = 7 )

	#---------------------------------------------------------------------------

	def test_multiple_known_distances(self):

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 0.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, 10.0], 1.0), 100.0), 10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, -10.0], 1.0), 100.0), 10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 0.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, 10.0], 1.0), 15.0), 5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, -10.0], 1.0), 15.0), 5.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 0.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, 10.0], 1.0), 10.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 0.0, -10.0], 1.0), 10.0), 0.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, -10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, -10.0, 0.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, -10.0], 1.0), 100.0), np.sqrt(2)*10.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, -10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, -10.0, 0.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, -10.0], 1.0), 15.0), np.sqrt(2)*5.0, places = 7 )

		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, -10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, -10.0, 0.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([10.0, 0.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([-10.0, 0.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, 10.0, -10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, 10.0], 1.0), 5.0), 0.0, places = 7 )
		self.assertAlmostEqual( distance_pbc(self.b0, Bead([0.0, -10.0, -10.0], 1.0), 5.0), 0.0, places = 7 )

	#---------------------------------------------------------------------------

	def test_pointer_pbc_matrix(self):

		np.random.seed(0)

		box_length = 10.0

		beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(100) ]

		c_ish = compute_pointer_pbc_matrix(beads, box_length)

		python_ish = compute_pointer_pbc_matrix_python(beads, box_length)

		for i in range(len(beads)):
			for j in range(len(beads)):
				for k in range(3):

					self.assertEqual(c_ish[i][j][k], python_ish[i][j][k])

	#---------------------------------------------------------------------------

	def test_pointer_pbc_matrix(self):

		np.random.seed(0)

		box_length = 10.0

		overlap_treshold = 0.0

		for i in range(10000):

			beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(10) ]

			c_ish = check_overlaps(beads, box_length, overlap_treshold)

			python_ish = check_overlaps_python(beads, box_length, overlap_treshold)

			self.assertEqual(c_ish, python_ish)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------