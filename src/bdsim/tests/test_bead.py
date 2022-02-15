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

from pyBrown.bead import Bead, overlap, overlap_pbc, distance, distance_pbc, pointer_pbc, pole_pointer_pbc, pole_distance_pbc, compute_pointer_pbc_matrix, compute_pointer_immobile_pbc_matrix, check_overlaps, build_connection_matrix, angle_pbc
from pyBrown.plane import Plane

#-------------------------------------------------------------------------------

def compute_pointer_pbc_matrix_python(beads, box_length):

	rij = np.zeros((len(beads), len(beads), 3))

	for i in range(len(beads)-1):
		for j in range(i+1, len(beads)):
			rij[i][j] = pointer_pbc(beads[i], beads[j], box_length)
			rij[j][i] = -rij[i][j]

	return rij

#-------------------------------------------------------------------------------

def compute_pointer_immobile_pbc_matrix_python(mobile_beads, immobile_beads, box_length):

	rik = np.zeros((len(mobile_beads), len(immobile_beads), 3))

	for i in range(len(mobile_beads)):
		for k in range(len(immobile_beads)):
			rik[i][k] = pointer_pbc(mobile_beads[i], immobile_beads[k], box_length)

	return rik

#-------------------------------------------------------------------------------

def check_overlaps_python(beads, box_length, overlap_treshold, connection_matrix):

	overlaps = False

	for i in range(len(beads)-1):
		for j in range(i+1, len(beads)):
			pointer = beads[i].r - beads[j].r
			radii_sum = beads[i].a + beads[j].a
			radii_sum_pbc = box_length - radii_sum

			if connection_matrix[i][j]:
				continue

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

		plane = Plane(plane_point, normal)

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 0.0], plane), 0 )

	#---------------------------------------------------------------------------

	def test_if_going_through_xy_parallel_plane_results_in_positive_flux(self):

		normal = [0.0, 0.0, 1.0]

		plane_point = [0.0, 0.0, 1.0]

		plane = Plane(plane_point, normal)

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 2.0], plane), 1 )

	#---------------------------------------------------------------------------

	def test_if_going_through_xy_antiparallel_plane_results_in_negative_flux(self):

		normal = [0.0, 0.0, -1.0]

		plane_point = [0.0, 0.0, 1.0]

		plane = Plane(plane_point, normal)

		self.assertEqual( self.b0.translate_and_return_flux([0.0, 0.0, 2.0], plane), -1 )

	#---------------------------------------------------------------------------

	def test_if_going_through_yz_plane_in_two_steps_results_in_positive_flux(self):

		b_start = Bead([-1.0, 0.0, 0.0], 0.0)

		translation_vector = np.array([0.6, 0.0, 0.0])

		normal = [1.0, 0.0, 0.0]

		plane_point = [0.0, 0.0, 0.0]

		plane = Plane(plane_point, normal)

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, plane), 0 )

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, plane), 1 )

	#---------------------------------------------------------------------------

	def test_if_going_parallel_to_plane_does_not_produce_flux(self):

		b_start = Bead([-1.0, 0.0, 0.0], 0.0)

		translation_vector = np.array([0.0, 20.0, 0.0])

		normal = [1.0, 0.0, 0.0]

		plane_point = [0.0, 0.0, 0.0]

		plane = Plane(plane_point, normal)

		self.assertEqual( b_start.translate_and_return_flux(translation_vector, plane), 0 )

	#---------------------------------------------------------------------------

	def test_if_going_through_xy_plane_is_crossing(self):

		b_start = Bead([0.0, 0.0, -1.0], 0.0)

		translation_vector = np.array([0.0, 0.0, 2.0])

		normal = np.array([0.0, 0.0, 1.0])

		plane_point = np.array([0.0, 0.0, 0.0])

		plane = Plane(plane_point, normal)

		self.assertTrue( b_start.translate_and_check_for_plane_crossing(translation_vector, [plane]) )

	#---------------------------------------------------------------------------

	def test_if_going_through_xz_plane_is_crossing(self):

		b_start = Bead([0.0, 1.0, 0.0], 0.0)

		translation_vector = np.array([0.0, -2.0, 0.0])

		normal = np.array([0.0, -1.0, 0.0])

		plane_point = np.array([0.0, 0.0, 0.0])

		plane = Plane(plane_point, normal)

		self.assertTrue( b_start.translate_and_check_for_plane_crossing(translation_vector, [plane]) )

	#---------------------------------------------------------------------------

	def test_if_going_through_yz_plane_is_crossing(self):

		b_start = Bead([-1.0, 0.0, 0.0], 0.0)

		translation_vector = np.array([2.0, 0.0, 0.0])

		normal = np.array([1.0, 0.0, 0.0])

		plane_point = np.array([0.0, 0.0, 0.0])

		plane = Plane(plane_point, normal)

		self.assertTrue( b_start.translate_and_check_for_plane_crossing(translation_vector, [plane]) )

	#---------------------------------------------------------------------------

	def test_if_going_through_shifted_xy_plane_is_crossing(self):

		b_start = Bead([0.0, 0.0, 1.0], 0.0)

		translation_vector = np.array([0.0, 0.0, 2.0])

		normal = np.array([0.0, 0.0, 1.0])

		plane_point = np.array([0.0, 0.0, 0.0])

		plane = Plane(plane_point, normal)

		plane_point_shift = np.array([0.0, 0.0, 2.0])

		plane_shift = Plane(plane_point_shift, normal)

		self.assertFalse( b_start.translate_and_check_for_plane_crossing(translation_vector, [plane]) )

		self.assertFalse( b_start.translate_and_check_for_plane_crossing(-translation_vector, [plane]) )

		self.assertTrue( b_start.translate_and_check_for_plane_crossing(translation_vector, [plane_shift]) )

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

	def test_pole_pointer(self):

		beada = Bead([0.0, 0.0, 0.0], 1.0)

		beadb = Bead([0.0, 0.0, 2.0], 1.0)

		beadc = Bead([0.0, 0.0, 4.0], 1.0)

		beadd = Bead([0.0, 2.0, 2.0], 1.0)

		self.assertSequenceEqual( list(pole_pointer_pbc(beada, beadb, beadc, 1000.0)), [ 0.0, 0.0, 1.0 ] )

		self.assertEqual( pole_distance_pbc(beada, beadb, beadc, 1000.0), 1.0 )

		self.assertSequenceEqual( list(pole_pointer_pbc(beada, beadb, beadd, 1000.0)), [ 0.0, 2.0, -1.0] )

		self.assertEqual( pole_distance_pbc(beada, beadb, beadd, 1000.0), np.sqrt(5) )

	#---------------------------------------------------------------------------

	def test_pole_pointer2(self):

		beada = Bead([0.0, 0.0, 0.0], 0.5)

		beadb = Bead([0.0, 0.0, 2.0], 1.5)

		beadc = Bead([0.0, 0.0, 4.0], 0.75)

		beadd = Bead([0.0, 2.0, 2.0], 1.0)

		self.assertSequenceEqual( list(pole_pointer_pbc(beada, beadb, beadc, 1000.0)), [ 0.0, 0.0, 0.5 ] )

		self.assertEqual( pole_distance_pbc(beada, beadb, beadc, 1000.0), 0.5 )

		self.assertSequenceEqual( list(pole_pointer_pbc(beada, beadb, beadd, 1000.0)), [ 0.0, 2.0, -1.5] )

		self.assertEqual( pole_distance_pbc(beada, beadb, beadd, 1000.0), np.sqrt(6.25) )

	#---------------------------------------------------------------------------

	def test_pointer_pbc_matrix_vs_python(self):

		for s in range(100):

			np.random.seed(s)

			box_length = 10.0

			beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(100) ]

			c_ish = compute_pointer_pbc_matrix(beads, box_length)

			python_ish = compute_pointer_pbc_matrix_python(beads, box_length)

			for i in range(len(beads)):
				for j in range(len(beads)):
					for k in range(3):

						self.assertEqual(c_ish[i][j][k], python_ish[i][j][k])

	#---------------------------------------------------------------------------

	def test_pointer_immobile_pbc_matrix_vs_python(self):

		for s in range(100):

			np.random.seed(s)

			box_length = 10.0

			mobile_beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(20) ]

			immobile_beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(80) ]

			c_ish = compute_pointer_immobile_pbc_matrix(mobile_beads, immobile_beads, box_length)

			python_ish = compute_pointer_immobile_pbc_matrix_python(mobile_beads, immobile_beads, box_length)

			for i in range(len(mobile_beads)):
				for j in range(len(immobile_beads)):
					for k in range(3):

						self.assertEqual(c_ish[i][j][k], python_ish[i][j][k])

	#---------------------------------------------------------------------------

	def test_overlap_pbc(self):

		np.random.seed(0)

		box_length = 10.0

		overlap_treshold = 0.0

		for i in range(10000):

			beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0, bead_id = i+1) for i in range(10) ]

			connection_matrix = build_connection_matrix(beads)
			self.assertSequenceEqual(list(connection_matrix.flatten()), list(np.transpose(np.zeros((10,10)).flatten())))

			c_ish = check_overlaps(beads, box_length, overlap_treshold, connection_matrix)

			python_ish = check_overlaps_python(beads, box_length, overlap_treshold, connection_matrix)

			self.assertEqual(c_ish, python_ish)

	#---------------------------------------------------------------------------

	def test_overlap_pbc_with_random_connections(self):

		np.random.seed(0)

		box_length = 10.0

		overlap_treshold = 0.0

		for i in range(10000):

			beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0, bead_id = i+1) for i in range(10) ]

			for j in range(len(beads)-1):

				for k in range(j+1, len(beads)):

					if np.random.choice([True, False]):

						beads[j].bonded_with.append(beads[k].bead_id)
						beads[j].bonded_how[beads[k].bead_id] = [0.0, 0.0]

			connection_matrix = build_connection_matrix(beads)
			self.assertSequenceEqual(list(connection_matrix.flatten()), list(np.transpose(connection_matrix.flatten())))

			c_ish = check_overlaps(beads, box_length, overlap_treshold, connection_matrix)

			python_ish = check_overlaps_python(beads, box_length, overlap_treshold, connection_matrix)

			self.assertEqual(c_ish, python_ish)

	#---------------------------------------------------------------------------

	def test_overlap_vs_bonds(self):

		box_length = 10.0

		overlap_treshold = 0.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0, bead_id = 1), Bead([0.0, 0.0, 1.0], 1.0, bead_id = 2) ]

		connection_matrix = build_connection_matrix(beads)

		self.assertTrue( check_overlaps(beads, box_length, overlap_treshold, connection_matrix) )

		beads[0].bonded_with.append( beads[1].bead_id )

		connection_matrix = build_connection_matrix(beads)

		self.assertFalse( check_overlaps(beads, box_length, overlap_treshold, connection_matrix) )

	#---------------------------------------------------------------------------

	def test_angle_pbc_0(self):

		bead1 = Bead([0.0, 0.0, 0.0], 1.0)
		bead2 = Bead([0.0, 0.0, 2.5], 1.0)
		bead3 = Bead([0.0, 0.0, 1.0], 1.0)

		box_length = 1000.0

		self.assertEqual(angle_pbc(bead2.r-bead1.r, bead3.r-bead2.r), 0.0)

	#---------------------------------------------------------------------------

	def test_angle_pbc_45(self):

		bead1 = Bead([0.0, 0.0, 0.0], 1.0)
		bead2 = Bead([1.0, 1.0, 0.0], 1.0)
		bead3 = Bead([0.0, 1.0, 0.0], 1.0)

		self.assertAlmostEqual(angle_pbc(bead2.r-bead1.r, bead3.r-bead2.r), 45.0, places = 7)

	#---------------------------------------------------------------------------

	def test_angle_pbc_90(self):

		bead1 = Bead([0.0, 0.0, 0.0], 1.0)
		bead2 = Bead([0.0, 0.0, 2.5], 1.0)
		bead3 = Bead([0.0, 1.0, 2.5], 1.0)

		self.assertEqual(angle_pbc(bead2.r-bead1.r, bead3.r-bead2.r), 90.0)

	#---------------------------------------------------------------------------

	def test_angle_pbc_180(self):

		bead1 = Bead([0.0, 0.0, 0.0], 1.0)
		bead2 = Bead([0.0, 0.0, 2.5], 1.0)
		bead3 = Bead([0.0, 0.0, 6.0], 1.0)

		self.assertEqual(angle_pbc(bead2.r-bead1.r, bead3.r-bead2.r), 180.0)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------