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
from pybrown.diffusion import JO_2B_R_matrix, RPY_M_matrix, JO_R_lubrication_correction_matrix

#-------------------------------------------------------------------------------

def clear_line(line):

	while line[0] == ' ' or line[0] == '\t' or line[0] == '[': line = line[1:]
	while line[-1] == ' ' or line[-1] == ']': line = line[:-1]

	return line

#-------------------------------------------------------------------------------

def add_line_to_list(line):

	row = []

	for element in line.split(): row.append( float(element) )

	return row

#-------------------------------------------------------------------------------

def array_from_string(string):
	lines = string.split('\n')

	array = []

	for line in lines:
		line = clear_line(line)
		# print( '<<{}>>'.format(line) )
		array.append( add_line_to_list(line) )

	return np.array(array)

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

		beads = [beadi, beadj]
		pointers = compute_pointer_pbc_matrix(beads, 1000000.0)
		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = 1000000.0, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_3[i][j], delta = 0.001)

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

		beads = [beadi, beadj]
		pointers = compute_pointer_pbc_matrix(beads, 1000000.0)
		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = 1000000.0, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_2_5[i][j], delta = 0.001)

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

		beads = [beadi, beadj]
		pointers = compute_pointer_pbc_matrix(beads, 1000000.0)
		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = 1000000.0, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_2_1[i][j], delta = 0.002)

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

		beads = [beadi, beadj]
		pointers = compute_pointer_pbc_matrix(beads, 1000000.0)
		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = 1000000.0, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_2_01[i][j], delta = 0.005)

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

		beads = [beadi, beadj]
		pointers = compute_pointer_pbc_matrix(beads, 1000000.0)
		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = 1000000.0, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_2_001[i][j], delta = 0.005)

	#-------------------------------------------------------------------------------

	def test_JO_R_3inline_3000(self):

		insanely_large_box_size = 1000000.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([3000.0, 0.0, 0.0], 1.0), Bead([6000.0, 0.0, 0.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		R_hydrolib_3inline_3000 = np.array( [ [ 2.7279085850957747, 0.0, 0.0, -1.3636135438672267E-003, 0.0, 0.0, -6.8129532706100399E-004, 0.0, 0.0 ],
										   	  [ 0.0, 2.7279079460403417, 0.0, 0.0, -6.8189183253035759E-004, 0.0, 0.0, -3.4081802663550831E-004, 0.0 ],
										   	  [ 0.0, 0.0, 2.7279079460403417, 0.0, 0.0, -6.8189183246217689E-004, 0.0, 0.0, -3.4081802660144344E-004 ],
										   	  [ -1.3636135438672269E-003, 0.0, 0.0, 2.7279090965787858, 0.0, 0.0, -1.3636135438672274E-003, 0.0, 0.0 ],
										   	  [ 0.0, -6.8189183253035759E-004, 0.0, 0.0, 2.7279080739110815, 0.0, 0.0, -6.8189183253035770E-004, 0.0 ],
										   	  [ 0.0, 0.0, -6.8189183246217700E-004, 0.0, 0.0, 2.7279080739110815, 0.0, 0.0, -6.8189183246217689E-004 ],
										   	  [ -6.8129532706100410E-004, 0.0, 0.0, -1.3636135438672274E-003, 0.0, 0.0, 2.7279085850957743, 0.0, 0.0 ],
										   	  [ 0.0, -3.4081802663550825E-004, 0.0, 0.0, -6.8189183253035749E-004, 0.0, 0.0, 2.7279079460403417, 0.0 ],
										   	  [ 0.0, 0.0, -3.4081802660144344E-004, 0.0, 0.0, -6.8189183246217667E-004, 0.0, 0.0, 2.7279079460403417 ] ] )

		R_hydrolib_3inline_3000 /= self.R_hydrolib_infty

		for i in range(9):
			for j in range(9):
				self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_3inline_3000[i][j], delta = 0.0000001)

	#-------------------------------------------------------------------------------

	def test_3inline_3_hydrolib(self):

		insanely_large_box_size = 1000000.0

		# 2 beads separated by 3.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 3.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_hydrolib = ' 2.9791344182688091        2.0536680767759284E-021   3.3964340629756637E-023 -0.81066685570698527       -9.7410224887806375E-021  -2.2873814939236211E-023\n\
						   2.0536680767759265E-021   2.9791344182688091        4.4536721082844076E-033  -9.7410224887806421E-021 -0.81066685570698527       -4.9884896427160042E-033\n\
						   3.3964340629756072E-023   4.4536029986745556E-033   3.7312230404791031       -2.2873814939235041E-023  -4.9886366421239579E-033  -1.8263361551445891\n\
						  -0.81066685570698538       -9.7410224887806436E-021  -2.2873814939234791E-023   2.9791344182688086        2.0536680767759242E-021   3.3964340629755320E-023\n\
						  -9.7410224887806375E-021 -0.81066685570698538       -4.9885151147116695E-033   2.0536680767759261E-021   2.9791344182688086        4.4535564477191326E-033\n\
						  -2.2873814939235441E-023  -4.9884878880461526E-033  -1.8263361551445891        3.3964340629755367E-023   4.4537319218976483E-033   3.7312230404791040'

		M_inv_hydrolib = array_from_string(M_inv_hydrolib) / self.R_hydrolib_infty

		R_hydrolib = ' 2.9791344182688091        2.0536680767759284E-021   3.3964340629756637E-023 -0.81066685570698527       -9.7410224887806375E-021  -2.2873814939236211E-023\n\
					   2.0536680767759265E-021   2.9791344182688091        4.4536721082844076E-033  -9.7410224887806421E-021 -0.81066685570698527       -4.9884896427160042E-033\n\
					   3.3964340629756072E-023   4.4536029986745556E-033   3.7312230404791031       -2.2873814939235041E-023  -4.9886366421239579E-033  -1.8263361551445891\n\
					  -0.81066685570698538       -9.7410224887806436E-021  -2.2873814939234791E-023   2.9791344182688086        2.0536680767759242E-021   3.3964340629755320E-023\n\
					  -9.7410224887806375E-021 -0.81066685570698538       -4.9885151147116695E-033   2.0536680767759261E-021   2.9791344182688086        4.4535564477191326E-033\n\
					  -2.2873814939235441E-023  -4.9884878880461526E-033  -1.8263361551445891        3.3964340629755367E-023   4.4537319218976483E-033   3.7312230404791040'

		R_hydrolib = array_from_string(R_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_hydrolib[i][j], R_pybrown_tot[i][j], delta = 0.001)

		R_diff_coupling = M_inv_hydrolib - R_pybrown_rpy

		# 2 beads separated by 6.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 6.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_doubledist_hydrolib = ' 2.7745807365547943       -2.0475446836836327E-022   3.3988786072209832E-025 -0.35346335089271053        4.2942448913989604E-023  -1.3147912146442580E-025\n\
									 -2.0475446836836339E-022   2.7745807365547943        2.9237902399708226E-037   4.2942448913989604E-023 -0.35346335089271053       -1.3144254960278847E-037\n\
									  3.3988786072206581E-025   2.9238095737917875E-037   2.9118709268376257       -1.3147912146441836E-025  -1.3144437105508625E-037 -0.71666450564426687\n\
									 -0.35346335089271064        4.2942448913989587E-023  -1.3147912146442293E-025   2.7745807365547939       -2.0475446836836332E-022   3.3988786072207518E-025\n\
									  4.2942448913989604E-023 -0.35346335089271064       -1.3143345001541023E-037  -2.0475446836836332E-022   2.7745807365547939        2.9237798877128998E-037\n\
									 -1.3147912146441551E-025  -1.3144273061619844E-037 -0.71666450564426687        3.3988786072207417E-025   2.9238026549421340E-037   2.9118709268376279'

		M_inv_doubledist_hydrolib = array_from_string(M_inv_doubledist_hydrolib) / self.R_hydrolib_infty

		R_doubledist_hydrolib = ' 2.7745807365547943       -2.0475446836836327E-022   3.3988786072209832E-025 -0.35346335089271053        4.2942448913989604E-023  -1.3147912146442580E-025\n\
								 -2.0475446836836339E-022   2.7745807365547943        2.9237902399708226E-037   4.2942448913989604E-023 -0.35346335089271053       -1.3144254960278847E-037\n\
								  3.3988786072206581E-025   2.9238095737917875E-037   2.9118709268376257       -1.3147912146441836E-025  -1.3144437105508625E-037 -0.71666450564426687\n\
								 -0.35346335089271064        4.2942448913989587E-023  -1.3147912146442293E-025   2.7745807365547939       -2.0475446836836332E-022   3.3988786072207518E-025\n\
								  4.2942448913989604E-023 -0.35346335089271064       -1.3143345001541023E-037  -2.0475446836836332E-022   2.7745807365547939        2.9237798877128998E-037\n\
								 -1.3147912146441551E-025  -1.3144273061619844E-037 -0.71666450564426687        3.3988786072207417E-025   2.9238026549421340E-037   2.9118709268376279'

		R_doubledist_hydrolib = array_from_string(R_doubledist_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_doubledist_hydrolib[i][j], R_pybrown_tot[i][j], delta = 0.001)

		R_diff_doubledist_coupling = M_inv_doubledist_hydrolib - R_pybrown_rpy

		# 3 beads separated by 3.0, 6.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 3.0], 1.0), Bead([0.0, 0.0, 6.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_composed_hydrolib = ' 2.9938007440202359       -1.4221193986985605E-021   4.0176986132101057E-023 -0.75775272664156246       -1.0051409654346127E-020  -3.8049928633603320E-024 -0.20916792627982284        1.9092186828165809E-020  -3.7109611548424532E-023\n\
								   -1.4221193986985550E-021   2.9938007440202359        4.0816032681085875E-033  -1.0051409655411461E-020 -0.75775272664156246       -7.3577621996136054E-033   1.9092186828165806E-020 -0.20916792627982284        1.8671785019477006E-033\n\
								    4.0176986132102614E-023   4.0815037882471763E-033   3.7529350622857884       -3.4856132703740066E-023  -2.4254386263690307E-033  -1.7116189224268237       -3.7109611548427441E-023   1.8672439010511666E-033 -0.27692827268312131\n\
								   -0.75775272664156279       -1.0051409655411455E-020  -3.4856132703740953E-023   3.2016238839106577        5.6948035470688249E-021   8.6650080881031912E-023 -0.75775272664156279       -1.0051409655411443E-020  -3.4856132703737368E-023\n\
								   -1.0051409654346116E-020 -0.75775272664156279       -2.4254327673185718E-033   5.6948035470688234E-021   3.2016238839106577        6.0004684571711215E-033  -1.0051409654346127E-020 -0.75775272664156279       -2.4256045904018573E-033\n\
								   -3.8049928633602680E-024  -7.3575137939359744E-033  -1.7116189224268235        8.6650080881032618E-023   6.0002683871499938E-033   4.6310954487288170       -3.8049928633565395E-024  -7.3579898188378681E-033  -1.7116189224268241\n\
								   -0.20916792627982278        1.9092186828165812E-020  -3.7109611548426348E-023 -0.75775272664156279       -1.0051409654346128E-020  -3.8049928633572213E-024   2.9938007440202372       -1.4221193986985579E-021   4.0176986132101474E-023\n\
								    1.9092186828165800E-020 -0.20916792627982284        1.8671189127883727E-033  -1.0051409655411446E-020 -0.75775272664156257       -7.3577079093488754E-033  -1.4221193986985533E-021   2.9938007440202372        4.0817604572598994E-033\n\
								   -3.7109611548424897E-023   1.8671511918657083E-033 -0.27692827268312192       -3.4856132703737721E-023  -2.4257348048610032E-033  -1.7116189224268241        4.0176986132101098E-023   4.0819254033853757E-033   3.7529350622857867'

		M_inv_composed_hydrolib = array_from_string(M_inv_composed_hydrolib) / self.R_hydrolib_infty

		R_composed_hydrolib = ' 2.9938007440202359       -1.4221193986985605E-021   4.0176986132101057E-023 -0.75775272664156246       -1.0051409654346127E-020  -3.8049928633603320E-024 -0.20916792627982284        1.9092186828165809E-020  -3.7109611548424532E-023\n\
							   -1.4221193986985550E-021   2.9938007440202359        4.0816032681085875E-033  -1.0051409655411461E-020 -0.75775272664156246       -7.3577621996136054E-033   1.9092186828165806E-020 -0.20916792627982284        1.8671785019477006E-033\n\
							    4.0176986132102614E-023   4.0815037882471763E-033   3.7529350622857884       -3.4856132703740066E-023  -2.4254386263690307E-033  -1.7116189224268237       -3.7109611548427441E-023   1.8672439010511666E-033 -0.27692827268312131\n\
							   -0.75775272664156279       -1.0051409655411455E-020  -3.4856132703740953E-023   3.2016238839106577        5.6948035470688249E-021   8.6650080881031912E-023 -0.75775272664156279       -1.0051409655411443E-020  -3.4856132703737368E-023\n\
							   -1.0051409654346116E-020 -0.75775272664156279       -2.4254327673185718E-033   5.6948035470688234E-021   3.2016238839106577        6.0004684571711215E-033  -1.0051409654346127E-020 -0.75775272664156279       -2.4256045904018573E-033\n\
							   -3.8049928633602680E-024  -7.3575137939359744E-033  -1.7116189224268235        8.6650080881032618E-023   6.0002683871499938E-033   4.6310954487288170       -3.8049928633565395E-024  -7.3579898188378681E-033  -1.7116189224268241\n\
							   -0.20916792627982278        1.9092186828165812E-020  -3.7109611548426348E-023 -0.75775272664156279       -1.0051409654346128E-020  -3.8049928633572213E-024   2.9938007440202372       -1.4221193986985579E-021   4.0176986132101474E-023\n\
							    1.9092186828165800E-020 -0.20916792627982284        1.8671189127883727E-033  -1.0051409655411446E-020 -0.75775272664156257       -7.3577079093488754E-033  -1.4221193986985533E-021   2.9938007440202372        4.0817604572598994E-033\n\
							   -3.7109611548424897E-023   1.8671511918657083E-033 -0.27692827268312192       -3.4856132703737721E-023  -2.4257348048610032E-033  -1.7116189224268241        4.0176986132101098E-023   4.0819254033853757E-033   3.7529350622857867'

		R_composed_hydrolib = array_from_string(R_composed_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		R_diff_3B_coupling = np.zeros((9,9))

		R_diff_3B_coupling[0][0] = R_diff_coupling[0][0] + R_diff_doubledist_coupling[0][0]
		R_diff_3B_coupling[1][1] = R_diff_coupling[1][1] + R_diff_doubledist_coupling[1][1]
		R_diff_3B_coupling[2][2] = R_diff_coupling[2][2] + R_diff_doubledist_coupling[2][2]
		R_diff_3B_coupling[3][3] = R_diff_coupling[3][3] + R_diff_coupling[0][0]
		R_diff_3B_coupling[4][4] = R_diff_coupling[4][4] + R_diff_coupling[1][1]
		R_diff_3B_coupling[5][5] = R_diff_coupling[5][5] + R_diff_coupling[2][2]
		R_diff_3B_coupling[6][6] = R_diff_coupling[3][3] + R_diff_doubledist_coupling[3][3]
		R_diff_3B_coupling[7][7] = R_diff_coupling[4][4] + R_diff_doubledist_coupling[4][4]
		R_diff_3B_coupling[8][8] = R_diff_coupling[5][5] + R_diff_doubledist_coupling[5][5]
		R_diff_3B_coupling[0][3] = R_diff_3B_coupling[3][0] = R_diff_coupling[0][3]
		R_diff_3B_coupling[1][4] = R_diff_3B_coupling[4][1] = R_diff_coupling[1][4]
		R_diff_3B_coupling[2][5] = R_diff_3B_coupling[5][2] = R_diff_coupling[2][5]
		R_diff_3B_coupling[0][6] = R_diff_3B_coupling[6][0] = R_diff_doubledist_coupling[0][3]
		R_diff_3B_coupling[1][7] = R_diff_3B_coupling[7][1] = R_diff_doubledist_coupling[1][4]
		R_diff_3B_coupling[2][8] = R_diff_3B_coupling[8][2] = R_diff_doubledist_coupling[2][5]
		R_diff_3B_coupling[6][3] = R_diff_3B_coupling[3][6] = R_diff_coupling[0][3]
		R_diff_3B_coupling[7][4] = R_diff_3B_coupling[4][7] = R_diff_coupling[1][4]
		R_diff_3B_coupling[8][5] = R_diff_3B_coupling[5][8] = R_diff_coupling[2][5]

		for i in range(9):
			for j in range(9):
				self.assertAlmostEqual(R_composed_hydrolib[i][j]-M_inv_composed_hydrolib[i][j], R_pybrown_tot[i][j]-R_pybrown_rpy[i][j]-R_diff_3B_coupling[i][j], delta = 0.002)

	#-------------------------------------------------------------------------------

	def test_3inline_2_1_hydrolib(self):

		insanely_large_box_size = 1000000.0

		# 2 beads separated by 2.1

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 2.1], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_hydrolib = ' 3.7668415631628123        4.7625408589996756E-019  -1.2366469849859567E-020  -1.7669913022463506       -4.6449840298635491E-019   1.2374070221930988E-020\n\
						   4.7625408589996871E-019   3.7668415631628123        2.2523943398667538E-029  -4.6449840298635597E-019  -1.7669913022463506       -2.2541043158307981E-029\n\
						  -1.2366469849859942E-020   2.2524050273351493E-029   8.0687214463513843        1.2374070221931075E-020  -2.2541037823878329E-029  -6.2933383208308804\n\
						  -1.7669913022463508       -4.6449840298635453E-019   1.2374070221931204E-020   3.7668415631628123        4.7625408589996737E-019  -1.2366469849859390E-020\n\
						  -4.6449840298635597E-019  -1.7669913022463508       -2.2540858253495230E-029   4.7625408589996881E-019   3.7668415631628123        2.2524108113263745E-029\n\
						   1.2374070221931508E-020  -2.2541007778985234E-029  -6.2933383208308813       -1.2366469849859493E-020   2.2524118894319930E-029   8.0687214463513861'

		M_inv_hydrolib = array_from_string(M_inv_hydrolib) / self.R_hydrolib_infty

		R_hydrolib = ' 3.7998794346160971        4.7625408589996756E-019  -1.2366469849859567E-020  -1.7999650596843935       -4.6449840298635491E-019   1.2374070221930988E-020\n\
					   4.7625408589996871E-019   3.7998794346160971        2.2523943398667538E-029  -4.6449840298635597E-019  -1.7999650596843935       -2.2541043158307981E-029\n\
					  -1.2366469849859942E-020   2.2524050273351493E-029   10.998747952544798        1.2374070221931075E-020  -2.2541037823878329E-029  -9.2231504632368395\n\
					  -1.7999650596843935       -4.6449840298635453E-019   1.2374070221931204E-020   3.7998794346160971        4.7625408589996737E-019  -1.2366469849859390E-020\n\
					  -4.6449840298635597E-019  -1.7999650596843935       -2.2540858253495230E-029   4.7625408589996881E-019   3.7998794346160971        2.2524108113263745E-029\n\
					   1.2374070221931508E-020  -2.2541007778985234E-029  -9.2231504632368413       -1.2366469849859493E-020   2.2524118894319930E-029   10.998747952544800'

		R_hydrolib = array_from_string(R_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_hydrolib[i][j], R_pybrown_tot[i][j], delta = 0.005)

		R_diff_coupling = M_inv_hydrolib - R_pybrown_rpy

		# 2 beads separated by 4.2

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 4.2], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_doubledist_hydrolib = ' 2.8329156873365959       -2.0485716977518635E-021   4.8715626081712323E-024 -0.52646555760744007        3.8530395058022157E-022  -2.7027304520768638E-024\n\
									 -2.0485716977518624E-021   2.8329156873365959        1.0664670146512150E-035   3.8530395058022133E-022 -0.52646555760744007       -1.2722435817626480E-035\n\
									  4.8715626081712279E-024   1.0664863493266723E-035   3.1388970274233179       -2.7027304520769075E-024  -1.2719458367646211E-035  -1.0931280265814245\n\
									 -0.52646555760743996        3.8530395058022133E-022  -2.7027304520769391E-024   2.8329156873365968       -2.0485716977518635E-021   4.8715626081710626E-024\n\
									  3.8530395058022147E-022 -0.52646555760743996       -1.2720068752808857E-035  -2.0485716977518639E-021   2.8329156873365968        1.0664441517133729E-035\n\
									 -2.7027304520770383E-024  -1.2722413737319046E-035  -1.0931280265814249        4.8715626081710641E-024   1.0663777509190060E-035   3.1388970274233157'

		M_inv_doubledist_hydrolib = array_from_string(M_inv_doubledist_hydrolib) / self.R_hydrolib_infty

		R_doubledist_hydrolib = ' 2.8329156873365959       -2.0485716977518635E-021   4.8715626081712323E-024 -0.52646555760744007        3.8530395058022157E-022  -2.7027304520768638E-024\n\
								 -2.0485716977518624E-021   2.8329156873365959        1.0664670146512150E-035   3.8530395058022133E-022 -0.52646555760744007       -1.2722435817626480E-035\n\
								  4.8715626081712279E-024   1.0664863493266723E-035   3.1388970274233179       -2.7027304520769075E-024  -1.2719458367646211E-035  -1.0931280265814245\n\
								 -0.52646555760743996        3.8530395058022133E-022  -2.7027304520769391E-024   2.8329156873365968       -2.0485716977518635E-021   4.8715626081710626E-024\n\
								  3.8530395058022147E-022 -0.52646555760743996       -1.2720068752808857E-035  -2.0485716977518639E-021   2.8329156873365968        1.0664441517133729E-035\n\
								 -2.7027304520770383E-024  -1.2722413737319046E-035  -1.0931280265814249        4.8715626081710641E-024   1.0663777509190060E-035   3.1388970274233157'

		R_doubledist_hydrolib = array_from_string(R_doubledist_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_doubledist_hydrolib[i][j], R_pybrown_tot[i][j], delta = 0.005)

		R_diff_doubledist_coupling = M_inv_doubledist_hydrolib - R_pybrown_rpy

		# 3 beads separated by 2.1, 4.2

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 2.1], 1.0), Bead([0.0, 0.0, 4.2], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		R_pybrown_rpy = np.linalg.inv(M_pybrown_rpy)
		R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
		R_pybrown_tot = ( R_pybrown_lub_corr + R_pybrown_rpy ) / self.R_pybrown_infty

		M_inv_composed_hydrolib = ' 3.7845911310388765        3.5174505272787338E-019  -1.2954202827927385E-020  -1.6772576506473340       -3.7615390971915059E-018   6.2618640603592588E-021 -0.24489941621229036        3.5669990383914660E-018   6.7624636901585833E-021\n\
								    3.5174505272787473E-019   3.7845911310388765        2.1824805145433573E-029  -3.7615390971889984E-018  -1.6772576506473340       -2.1831863550075441E-029   3.5669990383914668E-018 -0.24489941621229017       -1.7016034326125065E-031\n\
								   -1.2954202827927155E-020   2.1824627747244744E-029   8.1062376286879285        5.7293407316730908E-021  -1.9250194881200115E-029  -5.9325643315580940        6.7624636901581169E-021  -1.7059117634033403E-031 -0.52247385567006344\n\
								   -1.6772576506473340       -3.7615390971889961E-018   5.7293407316737890E-021   4.7341611482990507        7.3287382188761050E-018  -1.1591703519050175E-020  -1.6772576506473342       -3.7615390971889961E-018   5.7293407316723882E-021\n\
								   -3.7615390971915067E-018  -1.6772576506473338       -1.9250163103209029E-029   7.3287382188761081E-018   4.7341611482990507        3.8815177323886322E-029  -3.7615390971915067E-018  -1.6772576506473340       -1.9249903068170164E-029\n\
								    6.2618640603591903E-021  -2.1832044452496351E-029  -5.9325643315580923       -1.1591703519050254E-020   3.8815753357998989E-029   12.824348997625904        6.2618640603592354E-021  -2.1831364143677783E-029  -5.9325643315580852\n\
								   -0.24489941621229036        3.5669990383914676E-018   6.7624636901584313E-021  -1.6772576506473342       -3.7615390971915052E-018   6.2618640603586419E-021   3.7845911310388778        3.5174505272787208E-019  -1.2954202827926149E-020\n\
								    3.5669990383914668E-018 -0.24489941621229044       -1.7003326622626134E-031  -3.7615390971889968E-018  -1.6772576506473342       -2.1831956977333957E-029   3.5174505272787304E-019   3.7845911310388778        2.1824730136939258E-029\n\
								    6.7624636901589075E-021  -1.6994372950424958E-031 -0.52247385567006377        5.7293407316725206E-021  -1.9250232200588735E-029  -5.9325643315580834       -1.2954202827926362E-020   2.1824597317211212E-029   8.1062376286879232'

		M_inv_composed_hydrolib = array_from_string(M_inv_composed_hydrolib) / self.R_hydrolib_infty

		R_composed_hydrolib = ' 3.8176290024921609        3.5174505272787338E-019  -1.2954202827927385E-020  -1.7102314080853767       -3.7615390971915059E-018   6.2618640603592588E-021 -0.24489941621229036        3.5669990383914660E-018   6.7624636901585833E-021\n\
							    3.5174505272787473E-019   3.8176290024921609        2.1824805145433573E-029  -3.7615390971889984E-018  -1.7102314080853767       -2.1831863550075441E-029   3.5669990383914668E-018 -0.24489941621229017       -1.7016034326125065E-031\n\
							   -1.2954202827927155E-020   2.1824627747244744E-029   11.036264134881343        5.7293407316730908E-021  -1.9250194881200115E-029  -8.8623764739640531        6.7624636901581169E-021  -1.7059117634033403E-031 -0.52247385567006344\n\
							   -1.7102314080853767       -3.7615390971889961E-018   5.7293407316737890E-021   4.8002368912056204        7.3287382188761050E-018  -1.1591703519050175E-020  -1.7102314080853769       -3.7615390971889961E-018   5.7293407316723882E-021\n\
							   -3.7615390971915067E-018  -1.7102314080853764       -1.9250163103209029E-029   7.3287382188761081E-018   4.8002368912056204        3.8815177323886322E-029  -3.7615390971915067E-018  -1.7102314080853767       -1.9249903068170164E-029\n\
							    6.2618640603591903E-021  -2.1832044452496351E-029  -8.8623764739640514       -1.1591703519050254E-020   3.8815753357998989E-029   18.684402010012736        6.2618640603592354E-021  -2.1831364143677783E-029  -8.8623764739640460\n\
							   -0.24489941621229036        3.5669990383914676E-018   6.7624636901584313E-021  -1.7102314080853769       -3.7615390971915052E-018   6.2618640603586419E-021   3.8176290024921626        3.5174505272787208E-019  -1.2954202827926149E-020\n\
							    3.5669990383914668E-018 -0.24489941621229044       -1.7003326622626134E-031  -3.7615390971889968E-018  -1.7102314080853769       -2.1831956977333957E-029   3.5174505272787304E-019   3.8176290024921626        2.1824730136939258E-029\n\
							    6.7624636901589075E-021  -1.6994372950424958E-031 -0.52247385567006377        5.7293407316725206E-021  -1.9250232200588735E-029  -8.8623764739640443       -1.2954202827926362E-020   2.1824597317211212E-029   11.036264134881339'

		R_composed_hydrolib = array_from_string(R_composed_hydrolib) / self.R_hydrolib_infty
		R_pybrown_rpy /= self.R_pybrown_infty

		R_diff_3B_coupling = np.zeros((9,9))

		R_diff_3B_coupling[0][0] = R_diff_coupling[0][0] + R_diff_doubledist_coupling[0][0]
		R_diff_3B_coupling[1][1] = R_diff_coupling[1][1] + R_diff_doubledist_coupling[1][1]
		R_diff_3B_coupling[2][2] = R_diff_coupling[2][2] + R_diff_doubledist_coupling[2][2]
		R_diff_3B_coupling[3][3] = R_diff_coupling[3][3] + R_diff_coupling[0][0]
		R_diff_3B_coupling[4][4] = R_diff_coupling[4][4] + R_diff_coupling[1][1]
		R_diff_3B_coupling[5][5] = R_diff_coupling[5][5] + R_diff_coupling[2][2]
		R_diff_3B_coupling[6][6] = R_diff_coupling[3][3] + R_diff_doubledist_coupling[3][3]
		R_diff_3B_coupling[7][7] = R_diff_coupling[4][4] + R_diff_doubledist_coupling[4][4]
		R_diff_3B_coupling[8][8] = R_diff_coupling[5][5] + R_diff_doubledist_coupling[5][5]
		R_diff_3B_coupling[0][3] = R_diff_3B_coupling[3][0] = R_diff_coupling[0][3]
		R_diff_3B_coupling[1][4] = R_diff_3B_coupling[4][1] = R_diff_coupling[1][4]
		R_diff_3B_coupling[2][5] = R_diff_3B_coupling[5][2] = R_diff_coupling[2][5]
		R_diff_3B_coupling[0][6] = R_diff_3B_coupling[6][0] = R_diff_doubledist_coupling[0][3]
		R_diff_3B_coupling[1][7] = R_diff_3B_coupling[7][1] = R_diff_doubledist_coupling[1][4]
		R_diff_3B_coupling[2][8] = R_diff_3B_coupling[8][2] = R_diff_doubledist_coupling[2][5]
		R_diff_3B_coupling[6][3] = R_diff_3B_coupling[3][6] = R_diff_coupling[0][3]
		R_diff_3B_coupling[7][4] = R_diff_3B_coupling[4][7] = R_diff_coupling[1][4]
		R_diff_3B_coupling[8][5] = R_diff_3B_coupling[5][8] = R_diff_coupling[2][5]

		for i in range(9):
			for j in range(9):
				self.assertAlmostEqual(R_composed_hydrolib[i][j]-M_inv_composed_hydrolib[i][j], R_pybrown_tot[i][j]-R_pybrown_rpy[i][j]-R_diff_3B_coupling[i][j], delta = 0.005)

	#-------------------------------------------------------------------------------

	def test_pecnut_RPY(self):

		insanely_large_box_size = 1000000.0

		beads = [ Bead([-5.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([7.0, 0.0, 0.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut = np.array( [ [ 0.05305169, 0.0, 0.0, 0.01549108, 0.0, 0.0, 0.00660075, 0.0, 0.0 ],
										[ 0.0, 0.05305169, 0.0, 0.0, 0.00816995, 0.0, 0.0, 0.00333108,  0.0 ],
										[ 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.00816995, 0.0, 0.0, 0.00333108 ],
										[ 0.01549108, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.01121354, 0.0, 0.0 ],
										[ 0.0, 0.00816995, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.00576144,  0.0 ],
										[ 0.0, 0.0, 0.00816995, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.00576144 ],
										[ 0.00660075, 0.0, 0.0, 0.01121354, 0.0, 0.0, 0.05305169, 0.0, 0.0 ],
										[ 0.0, 0.00333108, 0.0, 0.0, 0.00576144, 0.0, 0.0, 0.05305169, 0.0 ],
										[ 0.0, 0.0, 0.00333108, 0.0, 0.0, 0.00576144, 0.0, 0.0, 0.05305169 ] ] )

		for i in range(9):
			for j in range(9):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut[i][j], places = 7)

	#-------------------------------------------------------------------------------

	def test_pecnut_2inline_3_and_more(self):

		insanely_large_box_size = 1000000.0

		# 2 beads separated by 3.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 3.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut_string = '[[ 0.05305169  0.          0.          0.01424535  0.          0.          0.          0.          0.          0.         -0.00442097  0.        ]\n\
									[ 0.          0.05305169  0.          0.          0.01424535  0.          0.          0.          0.          0.00442097  0.          0.        ]\n\
									[ 0.          0.          0.05305169  0.          0.          0.02456095  0.          0.          0.          0.          0.          0.        ]\n\
									[ 0.01424535  0.          0.          0.05305169  0.          0.          0.          0.00442097  0.          0.          0.          0.        ]\n\
									[ 0.          0.01424535  0.          0.          0.05305169  0.         -0.00442097  0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.          0.02456095  0.          0.          0.05305169  0.          0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.          0.          0.         -0.00442097  0.          0.03978874  0.          0.         -0.00073683  0.          0.        ]\n\
									[ 0.          0.          0.          0.00442097  0.          0.          0.          0.03978874  0.          0.         -0.00073683  0.        ]\n\
									[ 0.          0.          0.          0.          0.          0.          0.          0.          0.03978874  0.          0.          0.00147366]\n\
									[ 0.          0.00442097  0.          0.          0.          0.         -0.00073683  0.          0.          0.03978874  0.          0.        ]\n\
									[-0.00442097  0.          0.          0.          0.          0.          0.         -0.00073683  0.          0.          0.03978874  0.        ]\n\
									[ 0.          0.          0.          0.          0.          0.          0.          0.          0.00147366  0.          0.          0.03978874]]'

		M_infinity_inv_pecnut_string = '[[  2.05522514e+01   0.00000000e+00   0.00000000e+00  -5.58251899e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.03204556e-01   0.00000000e+00   0.00000000e+00   2.31634292e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51212594e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.09491457e-01   0.00000000e+00]\n\
								 		[  0.00000000e+00   2.05522514e+01   0.00000000e+00   0.00000000e+00  -5.58251899e+00   0.00000000e+00  -7.03204556e-01   0.00000000e+00   0.00000000e+00  -2.31634292e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51212594e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.09491457e-01]\n\
								 		[  0.00000000e+00   0.00000000e+00   2.53845828e+01   0.00000000e+00   0.00000000e+00  -1.22527323e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.10916471e+00   0.00000000e+00  -2.10916471e+00   0.00000000e+00   0.00000000e+00  -3.63374114e+00   0.00000000e+00  -3.63374114e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[ -5.58251899e+00   0.00000000e+00   0.00000000e+00   2.05522514e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.31634292e+00   0.00000000e+00   0.00000000e+00  -7.03204556e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.09491457e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51212594e-01   0.00000000e+00]\n\
								 		[  0.00000000e+00  -5.58251899e+00   0.00000000e+00   0.00000000e+00   2.05522514e+01   0.00000000e+00   2.31634292e+00   0.00000000e+00   0.00000000e+00   7.03204556e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.09491457e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51212594e-01]\n\
								 		[  0.00000000e+00   0.00000000e+00  -1.22527323e+01   0.00000000e+00   0.00000000e+00   2.53845828e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.63374114e+00   0.00000000e+00   3.63374114e+00   0.00000000e+00   0.00000000e+00   2.10916471e+00   0.00000000e+00   2.10916471e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00  -7.03204556e-01   0.00000000e+00   0.00000000e+00   2.31634292e+00   0.00000000e+00   2.55329794e+01   0.00000000e+00   0.00000000e+00   5.50116090e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.08276632e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68873309e+00]\n\
								 		[  7.03204556e-01   0.00000000e+00   0.00000000e+00  -2.31634292e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.55329794e+01   0.00000000e+00   0.00000000e+00   5.50116090e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.08276632e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68873309e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51672642e+01   0.00000000e+00   0.00000000e+00  -9.32120897e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00  -2.31634292e+00   0.00000000e+00   0.00000000e+00   7.03204556e-01   0.00000000e+00   5.50116090e-01   0.00000000e+00   0.00000000e+00   2.55329794e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68873309e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.08276632e-02]\n\
								 		[  2.31634292e+00   0.00000000e+00   0.00000000e+00  -7.03204556e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.50116090e-01   0.00000000e+00   0.00000000e+00   2.55329794e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68873309e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.08276632e-02   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.32120897e-01   0.00000000e+00   0.00000000e+00   2.51672642e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00  -2.10916471e+00   0.00000000e+00   0.00000000e+00   3.63374114e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.16680582e+01   0.00000000e+00   7.22688351e-01   0.00000000e+00   0.00000000e+00   1.71228719e+00   0.00000000e+00   1.88467706e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453699e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72389875e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00  -2.10916471e+00   0.00000000e+00   0.00000000e+00   3.63374114e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.22688351e-01   0.00000000e+00   2.16680582e+01   0.00000000e+00   0.00000000e+00   1.88467706e+00   0.00000000e+00   1.71228719e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  2.51212594e-01   0.00000000e+00   0.00000000e+00  -5.09491457e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.08276632e-02   0.00000000e+00   0.00000000e+00   1.68873309e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11413470e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.25638159e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   2.51212594e-01   0.00000000e+00   0.00000000e+00  -5.09491457e-01   0.00000000e+00   1.08276632e-02   0.00000000e+00   0.00000000e+00  -1.68873309e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11413470e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.25638159e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00  -3.63374114e+00   0.00000000e+00   0.00000000e+00   2.10916471e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.71228719e+00   0.00000000e+00   1.88467706e+00   0.00000000e+00   0.00000000e+00   2.16680582e+01   0.00000000e+00   7.22688351e-01   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72389875e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453699e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 		[  0.00000000e+00   0.00000000e+00  -3.63374114e+00   0.00000000e+00   0.00000000e+00   2.10916471e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.88467706e+00   0.00000000e+00   1.71228719e+00   0.00000000e+00   0.00000000e+00   7.22688351e-01   0.00000000e+00   2.16680582e+01   0.00000000e+00   0.00000000e+00]\n\
								 		[  5.09491457e-01   0.00000000e+00   0.00000000e+00  -2.51212594e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68873309e+00   0.00000000e+00   0.00000000e+00  -1.08276632e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.25638159e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11413470e+01   0.00000000e+00]\n\
								 		[  0.00000000e+00   5.09491457e-01   0.00000000e+00   0.00000000e+00  -2.51212594e-01   0.00000000e+00  -1.68873309e+00   0.00000000e+00   0.00000000e+00   1.08276632e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.25638159e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11413470e+01]]'

		R_2B_exact_pecnut_string = '[[  3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00]\n\
									[  0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03]\n\
									[  0.00000000e+00   0.00000000e+00   4.10639028e-01   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00]\n\
									[ -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00]\n\
									[  0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02]\n\
									[  0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   4.10639028e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02]\n\
									[  9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02]\n\
									[  1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.75114170e-01   0.00000000e+00   1.65641365e-01   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.65641365e-01   0.00000000e+00   1.75114170e-01   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00]\n\
									[  3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00]\n\
									[  0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02]\n\
									[  0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00   1.75114170e-01   0.00000000e+00   1.65641365e-01   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00   1.65641365e-01   0.00000000e+00   1.75114170e-01   0.00000000e+00   0.00000000e+00]\n\
									[  5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00]\n\
									[  0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01]]'

		R_grand_pecnut_string = '[[ 20.58563305   0.           0.          -5.60163294   0.           0.           0.           0.71309732   0.           0.           2.32965592   0.           0.           0.           0.           0.2879775    0.           0.           0.           0.           0.51534355   0.        ]\n\
								 [  0.          20.58563305   0.           0.          -5.60163294   0.          -0.71309732   0.           0.          -2.32965592   0.           0.           0.           0.           0.           0.           0.2879775    0.           0.           0.           0.           0.51534355]\n\
								 [  0.           0.          25.7952218    0.           0.         -12.63249246   0.           0.           0.           0.           0.           0.          -2.36474088   0.          -2.36474088   0.           0.          -3.84726842   0.          -3.84726842   0.           0.        ]\n\
								 [ -5.60163294   0.           0.          20.58563305   0.           0.           0.          -2.32965592   0.           0.          -0.71309732   0.           0.           0.           0.          -0.51534355   0.           0.           0.           0.          -0.2879775    0.        ]\n\
								 [  0.          -5.60163294   0.           0.          20.58563305   0.           2.32965592   0.           0.           0.71309732   0.           0.           0.           0.           0.           0.          -0.51534355   0.           0.           0.           0.          -0.2879775 ]\n\
								 [  0.           0.         -12.63249246   0.           0.          25.7952218    0.           0.           0.           0.           0.           0.           3.84726842   0.           3.84726842   0.           0.           2.36474088   0.           2.36474088   0.           0.        ]\n\
								 [  0.          -0.71309732   0.           0.           2.32965592   0.          25.58477028   0.           0.           0.54449255   0.           0.           0.           0.           0.           0.           0.05552948   0.           0.           0.           0.          -1.7116061 ]\n\
								 [  0.71309732   0.           0.          -2.32965592   0.           0.           0.          25.58477028   0.           0.           0.54449255   0.           0.           0.           0.          -0.05552948   0.           0.           0.           0.           1.7116061    0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.          25.18198015   0.           0.          -0.93356454   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
								 [  0.          -2.32965592   0.           0.           0.71309732   0.           0.54449255   0.           0.          25.58477028   0.           0.           0.           0.           0.           0.          -1.7116061    0.           0.           0.           0.           0.05552948]\n\
								 [  2.32965592   0.           0.          -0.71309732   0.           0.           0.           0.54449255   0.           0.          25.58477028   0.           0.           0.           0.           1.7116061    0.           0.           0.           0.          -0.05552948   0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.          -0.93356454   0.           0.          25.18198015   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
								 [  0.           0.          -2.36474088   0.           0.           3.84726842   0.           0.           0.           0.           0.           0.          21.84317239   0.           0.88832972   0.           0.           1.82250358   0.           1.99484192   0.           0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          20.95484268   0.           0.           0.           0.          -0.17233834   0.           0.           0.        ]\n\
								 [  0.           0.          -2.36474088   0.           0.           3.84726842   0.           0.           0.           0.           0.           0.           0.88832972   0.          21.84317239   0.           0.           1.99484192   0.           1.82250358   0.           0.        ]\n\
								 [  0.2879775    0.           0.          -0.51534355   0.           0.           0.          -0.05552948   0.           0.           1.7116061    0.           0.           0.           0.          21.24847193   0.           0.           0.           0.          -1.28155297   0.        ]\n\
								 [  0.           0.2879775    0.           0.          -0.51534355   0.           0.05552948   0.           0.          -1.7116061    0.           0.           0.           0.           0.           0.          21.24847193   0.           0.           0.           0.          -1.28155297]\n\
								 [  0.           0.          -3.84726842   0.           0.           2.36474088   0.           0.           0.           0.           0.           0.           1.82250358   0.           1.99484192   0.           0.          21.84317239   0.           0.88832972   0.           0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          -0.17233834   0.           0.           0.           0.          20.95484268   0.           0.           0.        ]\n\
								 [  0.           0.          -3.84726842   0.           0.           2.36474088   0.           0.           0.           0.           0.           0.           1.99484192   0.           1.82250358   0.           0.           0.88832972   0.          21.84317239   0.           0.        ]\n\
								 [  0.51534355   0.           0.          -0.2879775    0.           0.           0.           1.7116061    0.           0.          -0.05552948   0.           0.           0.           0.          -1.28155297   0.           0.           0.           0.          21.24847193   0.        ]\n\
								 [  0.           0.51534355   0.           0.          -0.2879775    0.          -1.7116061    0.           0.           0.05552948   0.           0.           0.           0.           0.           0.          -1.28155297   0.           0.           0.           0.          21.24847193]]'

		M_infinity_pecnut = array_from_string(M_infinity_pecnut_string)[:6,:6]
		M_infinity_inv_pecnut = array_from_string(M_infinity_inv_pecnut_string)[:6,:6]
		R_2B_exact_pecnut = array_from_string(R_2B_exact_pecnut_string)[:6,:6]
		R_grand_pecnut = array_from_string(R_grand_pecnut_string)[:6,:6]

		R_pybrown = np.linalg.inv(M_pybrown_rpy) + JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_grand_pecnut[i][j], delta = 0.005)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut[i][j], delta = 0.005)

		R_diff_infinity_coupling = M_infinity_inv_pecnut - np.linalg.inv(M_infinity_pecnut)

		# 2 beads separated by 6.0
		# pecnut gives no lubrication correction for that distance

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 6.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut_doubledist_string = '[[  5.30516925e-02   0.00000000e+00   0.00000000e+00   6.75426070e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.10524266e-03   0.00000000e+00]\n\
											   [  0.00000000e+00   5.30516925e-02   0.00000000e+00   0.00000000e+00   6.75426070e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10524266e-03   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   5.30516925e-02   0.00000000e+00   0.00000000e+00   1.30173024e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  6.75426070e-03   0.00000000e+00   0.00000000e+00   5.30516925e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10524266e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   6.75426070e-03   0.00000000e+00   0.00000000e+00   5.30516925e-02   0.00000000e+00  -1.10524266e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   1.30173024e-02   0.00000000e+00   0.00000000e+00   5.30516925e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.10524266e-03   0.00000000e+00   3.97887358e-02   0.00000000e+00   0.00000000e+00  -9.21035550e-05   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10524266e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.97887358e-02   0.00000000e+00   0.00000000e+00  -9.21035550e-05   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.97887358e-02   0.00000000e+00   0.00000000e+00   1.84207110e-04]\n\
											   [  0.00000000e+00   1.10524266e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.21035550e-05   0.00000000e+00   0.00000000e+00   3.97887358e-02   0.00000000e+00   0.00000000e+00]\n\
											   [ -1.10524266e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.21035550e-05   0.00000000e+00   0.00000000e+00   3.97887358e-02   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.84207110e-04   0.00000000e+00   0.00000000e+00   3.97887358e-02]]'

		M_infinity_inv_pecnut_doubledist_string = '[[  1.91716153e+01   0.00000000e+00   0.00000000e+00  -2.44228342e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   0.00000000e+00   5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.83810821e-02   0.00000000e+00]\n\
												   [  0.00000000e+00   1.91716153e+01   0.00000000e+00   0.00000000e+00  -2.44228342e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.83810821e-02]\n\
												   [  0.00000000e+00   0.00000000e+00   2.01177090e+01   0.00000000e+00   0.00000000e+00  -4.95051995e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00]\n\
												   [ -2.44228342e+00   0.00000000e+00   0.00000000e+00   1.91716153e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00]\n\
												   [  0.00000000e+00  -2.44228342e+00   0.00000000e+00   0.00000000e+00   1.91716153e+01   0.00000000e+00   5.32780191e-01   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -7.61444930e-03]\n\
												   [  0.00000000e+00   0.00000000e+00  -4.95051995e+00   0.00000000e+00   0.00000000e+00   2.01177090e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   5.32780191e-01   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.05938279e-01]\n\
												   [  6.93532222e-02   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51332799e+01   0.00000000e+00   0.00000000e+00  -1.16357777e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.04357373e-04]\n\
												   [  5.32780191e-01   0.00000000e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.16357777e-01   0.00000000e+00   0.00000000e+00   2.51332799e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09787264e+01   0.00000000e+00   3.47739715e-02   0.00000000e+00   0.00000000e+00   2.32212289e-01   0.00000000e+00   2.37599108e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09439524e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.38681904e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.47739715e-02   0.00000000e+00   2.09787264e+01   0.00000000e+00   0.00000000e+00   2.37599108e-01   0.00000000e+00   2.32212289e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  7.61444930e-03   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00]\n\
												   [  0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01]\n\
												   [  0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.32212289e-01   0.00000000e+00   2.37599108e-01   0.00000000e+00   0.00000000e+00   2.09787264e+01   0.00000000e+00   3.47739715e-02   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.38681904e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09439524e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.37599108e-01   0.00000000e+00   2.32212289e-01   0.00000000e+00   0.00000000e+00   3.47739715e-02   0.00000000e+00   2.09787264e+01   0.00000000e+00   0.00000000e+00]\n\
												   [  2.83810821e-02   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00]\n\
												   [  0.00000000e+00   2.83810821e-02   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01]]'

		R_grand_pecnut_doubledist_string = '[[  1.91716153e+01   0.00000000e+00   0.00000000e+00  -2.44228342e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   0.00000000e+00   5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.83810821e-02   0.00000000e+00]\n\
											[  0.00000000e+00   1.91716153e+01   0.00000000e+00   0.00000000e+00  -2.44228342e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.83810821e-02]\n\
											[  0.00000000e+00   0.00000000e+00   2.01177090e+01   0.00000000e+00   0.00000000e+00  -4.95051995e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00]\n\
											[ -2.44228342e+00   0.00000000e+00   0.00000000e+00   1.91716153e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00]\n\
											[  0.00000000e+00  -2.44228342e+00   0.00000000e+00   0.00000000e+00   1.91716153e+01   0.00000000e+00   5.32780191e-01   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -7.61444930e-03]\n\
											[  0.00000000e+00   0.00000000e+00  -4.95051995e+00   0.00000000e+00   0.00000000e+00   2.01177090e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   5.32780191e-01   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.05938279e-01]\n\
											[  6.93532222e-02   0.00000000e+00   0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51332799e+01   0.00000000e+00   0.00000000e+00  -1.16357777e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00  -5.32780191e-01   0.00000000e+00   0.00000000e+00   6.93532222e-02   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.04357373e-04]\n\
											[  5.32780191e-01   0.00000000e+00   0.00000000e+00  -6.93532222e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.01345004e-02   0.00000000e+00   0.00000000e+00   2.51497024e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.16357777e-01   0.00000000e+00   0.00000000e+00   2.51332799e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09787264e+01   0.00000000e+00   3.47739715e-02   0.00000000e+00   0.00000000e+00   2.32212289e-01   0.00000000e+00   2.37599108e-01   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09439524e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.38681904e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -2.06411716e-01   0.00000000e+00   0.00000000e+00   7.75205628e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.47739715e-02   0.00000000e+00   2.09787264e+01   0.00000000e+00   0.00000000e+00   2.37599108e-01   0.00000000e+00   2.32212289e-01   0.00000000e+00   0.00000000e+00]\n\
											[  7.61444930e-03   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00]\n\
											[  0.00000000e+00   7.61444930e-03   0.00000000e+00   0.00000000e+00  -2.83810821e-02   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01]\n\
											[  0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.32212289e-01   0.00000000e+00   2.37599108e-01   0.00000000e+00   0.00000000e+00   2.09787264e+01   0.00000000e+00   3.47739715e-02   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.38681904e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09439524e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -7.75205628e-01   0.00000000e+00   0.00000000e+00   2.06411716e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.37599108e-01   0.00000000e+00   2.32212289e-01   0.00000000e+00   0.00000000e+00   3.47739715e-02   0.00000000e+00   2.09787264e+01   0.00000000e+00   0.00000000e+00]\n\
											[  2.83810821e-02   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05938279e-01   0.00000000e+00   0.00000000e+00  -9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01   0.00000000e+00]\n\
											[  0.00000000e+00   2.83810821e-02   0.00000000e+00   0.00000000e+00  -7.61444930e-03   0.00000000e+00  -2.05938279e-01   0.00000000e+00   0.00000000e+00   9.04357373e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.20898662e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09480072e+01]]'

		M_infinity_pecnut_doubledist = array_from_string(M_infinity_pecnut_doubledist_string)[:6,:6]
		M_infinity_inv_pecnut_doubledist = array_from_string(M_infinity_inv_pecnut_doubledist_string)[:6,:6]
		R_grand_pecnut_doubledist = array_from_string(R_grand_pecnut_doubledist_string)[:6,:6]

		R_pybrown = np.linalg.inv(M_pybrown_rpy) + JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_grand_pecnut_doubledist[i][j], delta = 0.005)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_infinity_inv_pecnut_doubledist[i][j], R_grand_pecnut_doubledist[i][j], delta = 0.005)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut_doubledist[i][j], delta = 0.005)

		R_diff_infinity_doubledist_coupling = M_infinity_inv_pecnut_doubledist - np.linalg.inv(M_infinity_pecnut_doubledist)

		# 3 beads separated by 3.0, 6.0

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 3.0], 1.0), Bead([0.0, 0.0, 6.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut_composed_string = '[[ 0.05305169  0.          0.          0.01424535  0.          0.          0.00675426  0.          0.          0.          0.          0.        ]\n\
											 [ 0.          0.05305169  0.          0.          0.01424535  0.          0.          0.00675426  0.          0.          0.          0.        ]\n\
											 [ 0.          0.          0.05305169  0.          0.          0.02456095  0.          0.          0.0130173   0.          0.          0.        ]\n\
											 [ 0.01424535  0.          0.          0.05305169  0.          0.          0.01424535  0.          0.          0.          0.00442097  0.        ]\n\
											 [ 0.          0.01424535  0.          0.          0.05305169  0.          0.          0.01424535  0.         -0.00442097  0.          0.        ]\n\
											 [ 0.          0.          0.02456095  0.          0.          0.05305169  0.          0.          0.02456095  0.          0.          0.        ]\n\
											 [ 0.00675426  0.          0.          0.01424535  0.          0.          0.05305169  0.          0.          0.          0.00110524  0.        ]\n\
											 [ 0.          0.00675426  0.          0.          0.01424535  0.          0.          0.05305169  0.         -0.00110524  0.          0.        ]\n\
											 [ 0.          0.          0.0130173   0.          0.          0.02456095  0.          0.          0.05305169  0.          0.          0.        ]\n\
											 [ 0.          0.          0.          0.         -0.00442097  0.          0.         -0.00110524  0.          0.03978874  0.          0.        ]\n\
											 [ 0.          0.          0.          0.00442097  0.          0.          0.00110524  0.          0.          0.          0.03978874  0.        ]\n\
											 [ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.          0.          0.03978874]]'

		M_infinity_inv_pecnut_composed_string = '[[  2.06563410e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00]\n\
												 [  0.00000000e+00   2.06563410e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02]\n\
												 [  0.00000000e+00   0.00000000e+00   2.55581573e+01   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.11350696e+00   0.00000000e+00  -2.11350696e+00   0.00000000e+00   0.00000000e+00  -3.94277928e+00   0.00000000e+00  -3.94277928e+00   0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00]\n\
												 [ -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.20502332e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   4.45824155e-17   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.45674151e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00]\n\
												 [  0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.20502332e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00  -5.32932034e-19   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.93278283e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92139155e-01]\n\
												 [  0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00   3.10568540e+01   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.58314623e+00   0.00000000e+00   3.58314623e+00   0.00000000e+00   0.00000000e+00   2.39265463e-16   0.00000000e+00   5.72218691e-16   0.00000000e+00   0.00000000e+00  -3.58314623e+00   0.00000000e+00  -3.58314623e+00   0.00000000e+00   0.00000000e+00]\n\
												 [ -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.06563410e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00]\n\
												 [  0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.06563410e+01   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.54312001e-01]\n\
												 [  0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00   2.55581573e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   3.94277928e+00   0.00000000e+00   3.94277928e+00   0.00000000e+00   0.00000000e+00   2.11350696e+00   0.00000000e+00   2.11350696e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02]\n\
												 [  7.09538314e-01   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51675321e+01   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -7.24168089e-17   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.59742750e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.13827075e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.69172452e+00]\n\
												 [  2.49610961e+00   0.00000000e+00   0.00000000e+00   7.48412285e-17   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.59742750e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.13827075e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   2.52015627e+01   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02]\n\
												 [  8.85238492e-02   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   2.51675321e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00  -2.11350696e+00   0.00000000e+00   0.00000000e+00   3.58314623e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.16689931e+01   0.00000000e+00   7.23622490e-01   0.00000000e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   0.00000000e+00   9.56768656e-02   0.00000000e+00   9.96454715e-02   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453706e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00  -2.11350696e+00   0.00000000e+00   0.00000000e+00   3.58314623e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.23622490e-01   0.00000000e+00   2.16689931e+01   0.00000000e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   0.00000000e+00   9.96454715e-02   0.00000000e+00   9.56768656e-02   0.00000000e+00   0.00000000e+00]\n\
												 [  2.54312001e-01   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00]\n\
												 [  0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02]\n\
												 [  0.00000000e+00   0.00000000e+00  -3.94277928e+00   0.00000000e+00   0.00000000e+00  -7.67228906e-17   0.00000000e+00   0.00000000e+00   3.94277928e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   0.00000000e+00   2.24797528e+01   0.00000000e+00   1.53296461e+00   0.00000000e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09467882e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00  -3.94277928e+00   0.00000000e+00   0.00000000e+00  -1.07836508e-16   0.00000000e+00   0.00000000e+00   3.94277928e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   0.00000000e+00   1.53296461e+00   0.00000000e+00   2.24797528e+01   0.00000000e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  5.49522173e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00  -1.13827075e-02   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13386866e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   1.13827075e-02   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13386866e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00]\n\
												 [  0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00  -3.58314623e+00   0.00000000e+00   0.00000000e+00   2.11350696e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.56768656e-02   0.00000000e+00   9.96454715e-02   0.00000000e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   0.00000000e+00   2.16689931e+01   0.00000000e+00   7.23622490e-01   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453706e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												 [  0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00  -3.58314623e+00   0.00000000e+00   0.00000000e+00   2.11350696e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.96454715e-02   0.00000000e+00   9.56768656e-02   0.00000000e+00   0.00000000e+00   1.90706701e+00   0.00000000e+00   1.73470979e+00   0.00000000e+00   0.00000000e+00   7.23622490e-01   0.00000000e+00   2.16689931e+01   0.00000000e+00   0.00000000e+00]\n\
												 [  4.25494712e-02   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00]\n\
												 [  0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01]]'

		R_2B_exact_pecnut_composed_string = '[[  3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   4.10639028e-01   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [ -1.91139529e-02   0.00000000e+00   0.00000000e+00   6.67633335e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00]\n\
											 [  0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   6.67633335e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03]\n\
											 [  0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   8.21278057e-01   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   4.10639028e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   1.03581762e-01   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.94036332e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02]\n\
											 [  1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   1.03581762e-01   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.94036332e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   2.94318406e-02   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.75114170e-01   0.00000000e+00   1.65641365e-01   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00  -2.55576172e-01   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.65641365e-01   0.00000000e+00   1.75114170e-01   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00   3.50228340e-01   0.00000000e+00   3.31282729e-01   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.89456109e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13527285e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00   3.31282729e-01   0.00000000e+00   3.50228340e-01   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00  -8.94036332e-02   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.14249790e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00]\n\
											 [  0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   8.94036332e-02   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.14249790e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10216391e-01   0.00000000e+00   1.10164859e-01   0.00000000e+00   0.00000000e+00   1.75114170e-01   0.00000000e+00   1.65641365e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.13527285e-01   0.00000000e+00   0.00000000e+00   2.55576172e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.10164859e-01   0.00000000e+00   1.10216391e-01   0.00000000e+00   0.00000000e+00   1.65641365e-01   0.00000000e+00   1.75114170e-01   0.00000000e+00   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00]\n\
											 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01]]'

		R_grand_pecnut_composed_string = '[[  2.06897227e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00]\n\
										  [  0.00000000e+00   2.06897227e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02]\n\
										  [  0.00000000e+00   0.00000000e+00   2.59687963e+01   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.36908313e+00   0.00000000e+00  -2.36908313e+00   0.00000000e+00   0.00000000e+00  -4.15630656e+00   0.00000000e+00  -4.15630656e+00   0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00]\n\
										  [ -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.21169965e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   4.45824155e-17   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.45674151e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00]\n\
										  [  0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.21169965e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00  -5.32932034e-19   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.93278283e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.97991247e-01]\n\
										  [  0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00   3.18781321e+01   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.79667352e+00   0.00000000e+00   3.79667352e+00   0.00000000e+00   0.00000000e+00   2.39265463e-16   0.00000000e+00   5.72218691e-16   0.00000000e+00   0.00000000e+00  -3.79667352e+00   0.00000000e+00  -3.79667352e+00   0.00000000e+00   0.00000000e+00]\n\
										  [ -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.06897227e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00]\n\
										  [  0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.06897227e+01   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91076909e-01]\n\
										  [  0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00   2.59687963e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   4.15630656e+00   0.00000000e+00   4.15630656e+00   0.00000000e+00   0.00000000e+00   2.36908313e+00   0.00000000e+00   2.36908313e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02]\n\
										  [  7.19431080e-01   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51822480e+01   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -7.24168089e-17   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.60778568e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00786341e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.71459752e+00]\n\
										  [  2.50942260e+00   0.00000000e+00   0.00000000e+00   7.48412285e-17   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.60778568e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.00786341e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   2.52309945e+01   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02]\n\
										  [  8.85238492e-02   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   2.51822480e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00  -2.36908313e+00   0.00000000e+00   0.00000000e+00   3.79667352e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.18441073e+01   0.00000000e+00   8.89263855e-01   0.00000000e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   0.00000000e+00   9.56768656e-02   0.00000000e+00   9.96454715e-02   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09548434e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00  -2.36908313e+00   0.00000000e+00   0.00000000e+00   3.79667352e+00   0.00000000e+00   0.00000000e+00   6.40368455e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.89263855e-01   0.00000000e+00   2.18441073e+01   0.00000000e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   0.00000000e+00   9.96454715e-02   0.00000000e+00   9.56768656e-02   0.00000000e+00   0.00000000e+00]\n\
										  [  2.91076909e-01   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00]\n\
										  [  0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02]\n\
										  [  0.00000000e+00   0.00000000e+00  -4.15630656e+00   0.00000000e+00   0.00000000e+00  -7.67228906e-17   0.00000000e+00   0.00000000e+00   4.15630656e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   0.00000000e+00   2.28299811e+01   0.00000000e+00   1.86424733e+00   0.00000000e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09657338e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00  -4.15630656e+00   0.00000000e+00   0.00000000e+00  -1.07836508e-16   0.00000000e+00   0.00000000e+00   4.15630656e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   0.00000000e+00   1.86424733e+00   0.00000000e+00   2.28299811e+01   0.00000000e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  5.55374265e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00  -1.00786341e-01   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.15529364e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   1.00786341e-01   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.15529364e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00]\n\
										  [  0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00  -3.79667352e+00   0.00000000e+00   0.00000000e+00   2.36908313e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.56768656e-02   0.00000000e+00   9.96454715e-02   0.00000000e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   0.00000000e+00   2.18441073e+01   0.00000000e+00   8.89263855e-01   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09548434e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
										  [  0.00000000e+00   0.00000000e+00  -6.40368455e-02   0.00000000e+00   0.00000000e+00  -3.79667352e+00   0.00000000e+00   0.00000000e+00   2.36908313e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.96454715e-02   0.00000000e+00   9.56768656e-02   0.00000000e+00   0.00000000e+00   2.01723187e+00   0.00000000e+00   1.84492618e+00   0.00000000e+00   0.00000000e+00   8.89263855e-01   0.00000000e+00   2.18441073e+01   0.00000000e+00   0.00000000e+00]\n\
										  [  4.25494712e-02   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00]\n\
										  [  0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01]]'

		M_infinity_pecnut_composed = array_from_string( M_infinity_pecnut_composed_string )[:9,:9]
		M_infinity_inv_pecnut_composed = array_from_string( M_infinity_inv_pecnut_composed_string )[:9,:9]
		R_2B_exact_pecnut_composed = array_from_string( R_2B_exact_pecnut_composed_string )[:9,:9]
		R_grand_pecnut_composed = array_from_string(R_grand_pecnut_composed_string)[:9,:9]

		R_pybrown = np.linalg.inv(M_pybrown_rpy) + JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut_composed[i][j], delta = 0.005)

		R_diff_3B_coupling = np.zeros((9,9))

		R_diff_3B_coupling[0][0] = R_diff_infinity_coupling[0][0]
		R_diff_3B_coupling[1][1] = R_diff_infinity_coupling[1][1]
		R_diff_3B_coupling[2][2] = R_diff_infinity_coupling[2][2]
		R_diff_3B_coupling[3][3] = R_diff_infinity_coupling[3][3]
		R_diff_3B_coupling[4][4] = R_diff_infinity_coupling[4][4]
		R_diff_3B_coupling[5][5] = R_diff_infinity_coupling[5][5]
		R_diff_3B_coupling[6][6] = R_diff_infinity_coupling[3][3]
		R_diff_3B_coupling[7][7] = R_diff_infinity_coupling[4][4]
		R_diff_3B_coupling[8][8] = R_diff_infinity_coupling[5][5]
		R_diff_3B_coupling[0][3] = R_diff_3B_coupling[3][0] = R_diff_infinity_coupling[0][3]
		R_diff_3B_coupling[1][4] = R_diff_3B_coupling[1][4] = R_diff_infinity_coupling[1][4]
		R_diff_3B_coupling[2][5] = R_diff_3B_coupling[2][5] = R_diff_infinity_coupling[2][5]
		R_diff_3B_coupling[0][6] = R_diff_3B_coupling[6][0] = 0.0
		R_diff_3B_coupling[1][7] = R_diff_3B_coupling[7][1] = 0.0
		R_diff_3B_coupling[2][8] = R_diff_3B_coupling[8][2] = 0.0
		R_diff_3B_coupling[6][3] = R_diff_3B_coupling[3][6] = R_diff_infinity_coupling[0][3]
		R_diff_3B_coupling[7][4] = R_diff_3B_coupling[4][7] = R_diff_infinity_coupling[1][4]
		R_diff_3B_coupling[8][5] = R_diff_3B_coupling[5][8] = R_diff_infinity_coupling[2][5]

		for i in range(9):
			for j in range(9):
				pass
				# if i != j: self.assertAlmostEqual(R_2B_exact_pecnut_composed[i][j] + R_diff_3B_coupling[i][j], JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)[i][j], delta = 0.005)

	#-------------------------------------------------------------------------------

	def test_pecnut_2inline_2_1_and_more(self):

		insanely_large_box_size = 1000000.0

		# 2 beads separated by 2.1

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 2.1], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut_string = '[[ 0.05305169  0.          0.          0.02181127  0.          0.          0.          0.          0.          0.         -0.00902239  0.        ]\n\
									[ 0.          0.05305169  0.          0.          0.02181127  0.          0.          0.          0.          0.00902239  0.          0.        ]\n\
									[ 0.          0.          0.05305169  0.          0.          0.03216553  0.          0.          0.          0.          0.          0.        ]\n\
									[ 0.02181127  0.          0.          0.05305169  0.          0.          0.          0.00902239  0.          0.          0.          0.        ]\n\
									[ 0.          0.02181127  0.          0.          0.05305169  0.         -0.00902239  0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.          0.03216553  0.          0.          0.05305169  0.          0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.          0.          0.         -0.00902239  0.          0.03978874  0.          0.         -0.00214819  0.          0.        ]\n\
									[ 0.          0.          0.          0.00902239  0.          0.          0.          0.03978874  0.          0.         -0.00214819  0.        ]\n\
									[ 0.          0.          0.          0.          0.          0.          0.          0.          0.03978874  0.          0.          0.00429638]\n\
									[ 0.          0.00902239  0.          0.          0.          0.         -0.00214819  0.          0.          0.03978874  0.          0.        ]\n\
									[-0.00902239  0.          0.          0.          0.          0.          0.         -0.00214819  0.          0.          0.03978874  0.        ]\n\
									[ 0.          0.          0.          0.          0.          0.          0.          0.          0.00429638  0.          0.          0.03978874]]'

		M_infinity_inv_pecnut_string = '[[ 24.64609336   0.           0.         -10.90100074   0.           0.           0.           3.47362703   0.           0.           6.24200906   0.           0.           0.           0.           2.03346768   0.           0.           0.           0.           2.90203876   0.        ]\n\
										[  0.          24.64609336   0.           0.         -10.90100074   0.          -3.47362703   0.           0.          -6.24200906   0.           0.           0.           0.           0.           0.           2.03346768   0.           0.           0.           0.           2.90203876]\n\
										[  0.           0.          38.62162871   0.           0.         -26.40925458   0.           0.           0.           0.           0.           0.          -7.97240905   0.          -7.97240905   0.           0.         -10.01632968   0.         -10.01632968   0.           0.        ]\n\
										[-10.90100074   0.           0.          24.64609336   0.           0.           0.          -6.24200906   0.           0.          -3.47362703   0.           0.           0.           0.          -2.90203876   0.           0.           0.           0.          -2.03346768   0.        ]\n\
										[  0.         -10.90100074   0.           0.          24.64609336   0.           6.24200906   0.           0.           3.47362703   0.           0.           0.           0.           0.           0.          -2.90203876   0.           0.           0.           0.          -2.03346768]\n\
										[  0.           0.         -26.40925458   0.           0.          38.62162871   0.           0.           0.           0.           0.           0.          10.01632968   0.          10.01632968   0.           0.           7.97240905   0.           7.97240905   0.           0.        ]\n\
										[  0.          -3.47362703   0.           0.           6.24200906   0.          27.96926192   0.           0.           2.44791735   0.           0.           0.           0.           0.           0.          -0.65568355   0.           0.           0.           0.          -5.62708053]\n\
										[  3.47362703   0.           0.          -6.24200906   0.           0.           0.          27.96926192   0.           0.           2.44791735   0.           0.           0.           0.           0.65568355   0.           0.           0.           0.           5.62708053   0.        ]\n\
										[  0.           0.           0.           0.           0.           0.           0.           0.          25.42923635   0.           0.          -2.74584131   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
										[  0.          -6.24200906   0.           0.           3.47362703   0.           2.44791735   0.           0.          27.96926192   0.           0.           0.           0.           0.           0.          -5.62708053   0.           0.           0.           0.          -0.65568355]\n\
										[  6.24200906   0.           0.          -3.47362703   0.           0.           0.           2.44791735   0.           0.          27.96926192   0.           0.           0.           0.           5.62708053   0.           0.           0.           0.           0.65568355   0.        ]\n\
										[  0.           0.           0.           0.           0.           0.           0.           0.          -2.74584131   0.           0.          25.42923635   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
										[  0.           0.          -7.97240905   0.           0.          10.01632968   0.           0.           0.           0.           0.           0.          24.30315988   0.           3.30886246   0.           0.           4.54592067   0.           5.57401951   0.           0.        ]\n\
										[  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          20.99429742   0.           0.           0.           0.          -1.02809884   0.           0.           0.        ]\n\
										[  0.           0.          -7.97240905   0.           0.          10.01632968   0.           0.           0.           0.           0.           0.           3.30886246   0.          24.30315988   0.           0.           5.57401951   0.           4.54592067   0.           0.        ]\n\
										[  2.03346768   0.           0.          -2.90203876   0.           0.           0.           0.65568355   0.           0.           5.62708053   0.           0.           0.           0.          22.39844117   0.           0.           0.           0.          -1.33668465   0.        ]\n\
										[  0.           2.03346768   0.           0.          -2.90203876   0.          -0.65568355   0.           0.          -5.62708053   0.           0.           0.           0.           0.           0.          22.39844117   0.           0.           0.           0.          -1.33668465]\n\
										[  0.           0.         -10.01632968   0.           0.           7.97240905   0.           0.           0.           0.           0.           0.           4.54592067   0.           5.57401951   0.           0.          24.30315988   0.           3.30886246   0.           0.        ]\n\
										[  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          -1.02809884   0.           0.           0.           0.          20.99429742   0.           0.           0.        ]\n\
										[  0.           0.         -10.01632968   0.           0.           7.97240905   0.           0.           0.           0.           0.           0.           5.57401951   0.           4.54592067   0.           0.           3.30886246   0.          24.30315988   0.           0.        ]\n\
										[  2.90203876   0.           0.          -2.03346768   0.           0.           0.           5.62708053   0.           0.           0.65568355   0.           0.           0.           0.          -1.33668465   0.           0.           0.           0.          22.39844117   0.        ]\n\
										[  0.           2.90203876   0.           0.          -2.03346768   0.          -5.62708053   0.           0.          -0.65568355   0.           0.           0.           0.           0.           0.          -1.33668465   0.           0.           0.           0.          22.39844117]]'

		R_2B_exact_pecnut_string = '[[  1.61144665   0.           0.          -1.53733885   0.           0.           0.           1.48384431   0.           0.           1.51325676   0.           0.           0.           0.           1.10893458   0.           0.           0.           0.           0.8555353    0.        ]\n\
									[  0.           1.61144665   0.           0.          -1.53733885   0.          -1.48384431   0.           0.          -1.51325676   0.           0.           0.           0.           0.           0.           1.10893458   0.           0.           0.           0.           0.8555353 \n\
									[  0.           0.          37.37928146   0.           0.         -37.32247382   0.           0.           0.           0.           0.           0.         -22.08052525   0.         -22.08052525   0.           0.         -21.97352373   0.         -21.97352373   0.           0.        \n\
									[ -1.53733885   0.           0.           1.61144665   0.           0.           0.          -1.51325676   0.           0.          -1.48384431   0.           0.           0.           0.          -0.8555353    0.           0.           0.           0.          -1.10893458   0.        \n\
									[  0.          -1.53733885   0.           0.           1.61144665   0.           1.51325676   0.           0.           1.48384431   0.           0.           0.           0.           0.           0.          -0.8555353    0.           0.           0.           0.          -1.10893458\n\
									[  0.           0.         -37.32247382   0.           0.          37.37928146   0.           0.           0.           0.           0.           0.          21.97352373   0.          21.97352373   0.           0.          22.08052525   0.          22.08052525   0.           0.        \n\
									[  0.          -1.48384431   0.           0.           1.51325676   0.           3.02004514   0.           0.           0.10911332   0.           0.           0.           0.           0.           0.           0.52170821   0.           0.           0.           0.          -2.1219513 \n\
									[  1.48384431   0.           0.          -1.51325676   0.           0.           0.           3.02004514   0.           0.           0.10911332   0.           0.           0.           0.          -0.52170821   0.           0.           0.           0.           2.1219513    0.        \n\
									[  0.           0.           0.           0.           0.           0.           0.           0.           0.41494041   0.           0.          -0.1904008    0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        \n\
									[  0.          -1.51325676   0.           0.           1.48384431   0.           0.10911332   0.           0.           3.02004514   0.           0.           0.           0.           0.           0.          -2.1219513    0.           0.           0.           0.           0.52170821\n\
									[  1.51325676   0.           0.          -1.48384431   0.           0.           0.           0.10911332   0.           0.           3.02004514   0.           0.           0.           0.           2.1219513    0.           0.           0.           0.          -0.52170821   0.        \n\
									[  0.           0.           0.           0.           0.           0.           0.           0.          -0.1904008    0.           0.           0.41494041   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        \n\
									[  0.           0.         -22.08052525   0.           0.          21.97352373   0.           0.           0.           0.           0.           0.          13.23764797   0.          13.03339741   0.           0.          12.86541526   0.          12.9066069    0.           0.        \n\
									[  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.20425055   0.           0.           0.           0.          -0.04119164   0.           0.           0.        \n\
									[  0.           0.         -22.08052525   0.           0.          21.97352373   0.           0.           0.           0.           0.           0.          13.03339741   0.          13.23764797   0.           0.          12.9066069    0.          12.86541526   0.           0.        \n\
									[  1.10893458   0.           0.          -0.8555353    0.           0.           0.          -0.52170821   0.           0.           2.1219513    0.           0.           0.           0.           2.33902258   0.           0.           0.           0.          -0.80502526   0.        \n\
									[  0.           1.10893458   0.           0.          -0.8555353    0.           0.52170821   0.           0.          -2.1219513    0.           0.           0.           0.           0.           0.           2.33902258   0.           0.           0.           0.          -0.80502526\n\
									[  0.           0.         -21.97352373   0.           0.          22.08052525   0.           0.           0.           0.           0.           0.          12.86541526   0.          12.9066069    0.           0.          13.23764797   0.          13.03339741   0.           0.        \n\
									[  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          -0.04119164   0.           0.           0.           0.           0.20425055   0.           0.           0.        \n\
									[  0.           0.         -21.97352373   0.           0.          22.08052525   0.           0.           0.           0.           0.           0.          12.9066069    0.          12.86541526   0.           0.          13.03339741   0.          13.23764797   0.           0.        \n\
									[  0.8555353    0.           0.          -1.10893458   0.           0.           0.           2.1219513    0.           0.          -0.52170821   0.           0.           0.           0.          -0.80502526   0.           0.           0.           0.           2.33902258   0.        ]\n\
									[  0.           0.8555353    0.           0.          -1.10893458   0.          -2.1219513    0.           0.           0.52170821   0.           0.           0.           0.           0.           0.          -0.80502526   0.           0.           0.           0.           2.33902258]]'

		R_grand_pecnut_string = '[[ 26.25754001   0.           0.         -12.43833959   0.           0.           0.           4.95747135   0.           0.           7.75526582   0.           0.           0.           0.           3.14240226   0.           0.           0.           0.           3.75757406   0.        ]\n\
								 [  0.          26.25754001   0.           0.         -12.43833959   0.          -4.95747135   0.           0.          -7.75526582   0.           0.           0.           0.           0.           0.           3.14240226   0.           0.           0.           0.           3.75757406]\n\
								 [  0.           0.          76.00091018   0.           0.         -63.7317284    0.           0.           0.           0.           0.           0.         -30.0529343    0.         -30.0529343    0.           0.         -31.98985341   0.         -31.98985341   0.           0.        ]\n\
								 [-12.43833959   0.           0.          26.25754001   0.           0.           0.          -7.75526582   0.           0.          -4.95747135   0.           0.           0.           0.          -3.75757406   0.           0.           0.           0.          -3.14240226   0.        ]\n\
								 [  0.         -12.43833959   0.           0.          26.25754001   0.           7.75526582   0.           0.           4.95747135   0.           0.           0.           0.           0.           0.          -3.75757406   0.           0.           0.           0.          -3.14240226]\n\
								 [  0.           0.         -63.7317284    0.           0.          76.00091018   0.           0.           0.           0.           0.           0.          31.98985341   0.          31.98985341   0.           0.          30.0529343    0.          30.0529343    0.           0.        ]\n\
								 [  0.          -4.95747135   0.           0.           7.75526582   0.          30.98930706   0.           0.           2.55703067   0.           0.           0.           0.           0.           0.          -0.13397534   0.           0.           0.           0.          -7.74903184]\n\
								 [  4.95747135   0.           0.          -7.75526582   0.           0.           0.          30.98930706   0.           0.           2.55703067   0.           0.           0.           0.           0.13397534   0.           0.           0.           0.           7.74903184   0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.          25.84417675   0.           0.          -2.93624211   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
								 [  0.          -7.75526582   0.           0.           4.95747135   0.           2.55703067   0.           0.          30.98930706   0.           0.           0.           0.           0.           0.          -7.74903184   0.           0.           0.           0.          -0.13397534]\n\
								 [  7.75526582   0.           0.          -4.95747135   0.           0.           0.           2.55703067   0.           0.          30.98930706   0.           0.           0.           0.           7.74903184   0.           0.           0.           0.           0.13397534   0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.          -2.93624211   0.           0.          25.84417675   0.           0.           0.           0.           0.           0.           0.           0.           0.           0.        ]\n\
								 [  0.           0.         -30.0529343    0.           0.          31.98985341   0.           0.           0.           0.           0.           0.          37.54080784   0.          16.34225987   0.           0.          17.41133593   0.          18.48062641   0.           0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          21.19854797   0.           0.           0.           0.          -1.06929048   0.           0.           0.        ]\n\
								 [  0.           0.         -30.0529343    0.           0.          31.98985341   0.           0.           0.           0.           0.           0.          16.34225987   0.          37.54080784   0.           0.          18.48062641   0.          17.41133593   0.           0.        ]\n\
								 [  3.14240226   0.           0.          -3.75757406   0.           0.           0.           0.13397534   0.           0.           7.74903184   0.           0.           0.           0.          24.73746376   0.           0.           0.           0.          -2.14170992   0.        ]\n\
								 [  0.           3.14240226   0.           0.          -3.75757406   0.          -0.13397534   0.           0.          -7.74903184   0.           0.           0.           0.           0.           0.          24.73746376   0.           0.           0.           0.          -2.14170992]\n\
								 [  0.           0.         -31.98985341   0.           0.          30.0529343    0.           0.           0.           0.           0.           0.          17.41133593   0.          18.48062641   0.           0.          37.54080784   0.          16.34225987   0.           0.        ]\n\
								 [  0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.          -1.06929048   0.           0.           0.           0.          21.19854797   0.           0.           0.        ]\n\
								 [  0.           0.         -31.98985341   0.           0.          30.0529343    0.           0.           0.           0.           0.           0.          18.48062641   0.          17.41133593   0.           0.          16.34225987   0.          37.54080784   0.           0.        ]\n\
								 [  3.75757406   0.           0.          -3.14240226   0.           0.           0.           7.74903184   0.           0.           0.13397534   0.           0.           0.           0.          -2.14170992   0.           0.           0.           0.          24.73746376   0.        ]\n\
								 [  0.           3.75757406   0.           0.          -3.14240226   0.          -7.74903184   0.           0.          -0.13397534   0.           0.           0.           0.           0.           0.          -2.14170992   0.           0.           0.           0.          24.73746376]]'

		M_infinity_pecnut = array_from_string(M_infinity_pecnut_string)[:6,:6]
		M_infinity_inv_pecnut = array_from_string(M_infinity_inv_pecnut_string)[:6,:6]
		R_2B_exact_pecnut = array_from_string(R_2B_exact_pecnut_string)[:6,:6]
		R_grand_pecnut = array_from_string(R_grand_pecnut_string)[:6,:6]

		R_pybrown = np.linalg.inv(M_pybrown_rpy) + JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)

		# for i in range(6):
		# 	for j in range(6):
		# 		# self.assertAlmostEqual(R_pybrown[i][j], R_grand_pecnut[i][j], delta = 0.005)
		# 		print('[{}][{}]: {} vs. {}'.format(i, j, R_pybrown[i][j], R_grand_pecnut[i][j]))

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut[i][j], delta = 0.005)

		R_diff_infinity_coupling = M_infinity_inv_pecnut - np.linalg.inv(M_infinity_pecnut)

		# 2 beads separated by 4.2
		# pecnut gives no lubrication correction for that distance

		beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 4.2], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut_doubledist_string = '[[ 0.05305169  0.          0.          0.00983154  0.          0.          0.          0.          0.          0.         -0.0022556   0.        ]\n\
											   [ 0.          0.05305169  0.          0.          0.00983154  0.          0.          0.          0.          0.0022556   0.          0.        ]\n\
											   [ 0.          0.          0.05305169  0.          0.          0.01823095  0.          0.          0.          0.          0.          0.        ]\n\
											   [ 0.00983154  0.          0.          0.05305169  0.          0.          0.          0.0022556   0.          0.          0.          0.        ]\n\
											   [ 0.          0.00983154  0.          0.          0.05305169  0.         -0.0022556   0.          0.          0.          0.          0.        ]\n\
											   [ 0.          0.          0.01823095  0.          0.          0.05305169  0.          0.          0.          0.          0.          0.        ]\n\
											   [ 0.          0.          0.          0.         -0.0022556   0.          0.03978874  0.          0.         -0.00026852  0.          0.        ]\n\
											   [ 0.          0.          0.          0.0022556   0.          0.          0.          0.03978874  0.          0.         -0.00026852  0.        ]\n\
											   [ 0.          0.          0.          0.          0.          0.          0.          0.          0.03978874  0.          0.          0.00053705]\n\
											   [ 0.          0.0022556   0.          0.          0.          0.         -0.00026852  0.          0.          0.03978874  0.          0.        ]\n\
											   [-0.0022556   0.          0.          0.          0.          0.          0.         -0.00026852  0.          0.          0.03978874  0.        ]\n\
											   [ 0.          0.          0.          0.          0.          0.          0.          0.          0.00053705  0.          0.          0.03978874]]'

		M_infinity_inv_pecnut_doubledist_string = '[[  1.95714051e+01   0.00000000e+00   0.00000000e+00  -3.63645242e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.17162750e-01   0.00000000e+00   0.00000000e+00   1.11224956e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.51901580e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.22531656e-01   0.00000000e+00]\n\
												   [  0.00000000e+00   1.95714051e+01   0.00000000e+00   0.00000000e+00  -3.63645242e+00   0.00000000e+00  -2.17162750e-01   0.00000000e+00   0.00000000e+00  -1.11224956e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.51901580e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.22531656e-01]\n\
												   [  0.00000000e+00   0.00000000e+00   2.16583682e+01   0.00000000e+00   0.00000000e+00  -7.53071198e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.56489529e-01   0.00000000e+00  -6.56489529e-01   0.00000000e+00   0.00000000e+00  -1.64971010e+00   0.00000000e+00  -1.64971010e+00   0.00000000e+00   0.00000000e+00]\n\
												   [ -3.63645242e+00   0.00000000e+00   0.00000000e+00   1.95714051e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.11224956e+00   0.00000000e+00   0.00000000e+00  -2.17162750e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.22531656e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.51901580e-02   0.00000000e+00]\n\
												   [  0.00000000e+00  -3.63645242e+00   0.00000000e+00   0.00000000e+00   1.95714051e+01   0.00000000e+00   1.11224956e+00   0.00000000e+00   0.00000000e+00   2.17162750e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.22531656e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.51901580e-02]\n\
												   [  0.00000000e+00   0.00000000e+00  -7.53071198e+00   0.00000000e+00   0.00000000e+00   2.16583682e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.64971010e+00   0.00000000e+00   1.64971010e+00   0.00000000e+00   0.00000000e+00   6.56489529e-01   0.00000000e+00   6.56489529e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00  -2.17162750e-01   0.00000000e+00   0.00000000e+00   1.11224956e+00   0.00000000e+00   2.52142923e+01   0.00000000e+00   0.00000000e+00   1.82315850e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56871869e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.03092075e-01]\n\
												   [  2.17162750e-01   0.00000000e+00   0.00000000e+00  -1.11224956e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.52142923e+01   0.00000000e+00   0.00000000e+00   1.82315850e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56871869e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03092075e-01   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51373208e+01   0.00000000e+00   0.00000000e+00  -3.39290044e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00  -1.11224956e+00   0.00000000e+00   0.00000000e+00   2.17162750e-01   0.00000000e+00   1.82315850e-01   0.00000000e+00   0.00000000e+00   2.52142923e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.03092075e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56871869e-03]\n\
												   [  1.11224956e+00   0.00000000e+00   0.00000000e+00  -2.17162750e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.82315850e-01   0.00000000e+00   0.00000000e+00   2.52142923e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03092075e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56871869e-03   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.39290044e-01   0.00000000e+00   0.00000000e+00   2.51373208e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -6.56489529e-01   0.00000000e+00   0.00000000e+00   1.64971010e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11057152e+01   0.00000000e+00   1.61715133e-01   0.00000000e+00   0.00000000e+00   6.52821443e-01   0.00000000e+00   6.84872560e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09440001e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.20511174e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -6.56489529e-01   0.00000000e+00   0.00000000e+00   1.64971010e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.61715133e-01   0.00000000e+00   2.11057152e+01   0.00000000e+00   0.00000000e+00   6.84872560e-01   0.00000000e+00   6.52821443e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  4.51901580e-02   0.00000000e+00   0.00000000e+00  -1.22531656e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56871869e-03   0.00000000e+00   0.00000000e+00   6.03092075e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09750835e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79240296e-01   0.00000000e+00]\n\
												   [  0.00000000e+00   4.51901580e-02   0.00000000e+00   0.00000000e+00  -1.22531656e-01   0.00000000e+00   5.56871869e-03   0.00000000e+00   0.00000000e+00  -6.03092075e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09750835e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79240296e-01]\n\
												   [  0.00000000e+00   0.00000000e+00  -1.64971010e+00   0.00000000e+00   0.00000000e+00   6.56489529e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.52821443e-01   0.00000000e+00   6.84872560e-01   0.00000000e+00   0.00000000e+00   2.11057152e+01   0.00000000e+00   1.61715133e-01   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.20511174e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09440001e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
												   [  0.00000000e+00   0.00000000e+00  -1.64971010e+00   0.00000000e+00   0.00000000e+00   6.56489529e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.84872560e-01   0.00000000e+00   6.52821443e-01   0.00000000e+00   0.00000000e+00   1.61715133e-01   0.00000000e+00   2.11057152e+01   0.00000000e+00   0.00000000e+00]\n\
												   [  1.22531656e-01   0.00000000e+00   0.00000000e+00  -4.51901580e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03092075e-01   0.00000000e+00   0.00000000e+00  -5.56871869e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79240296e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09750835e+01   0.00000000e+00]\n\
												   [  0.00000000e+00   1.22531656e-01   0.00000000e+00   0.00000000e+00  -4.51901580e-02   0.00000000e+00  -6.03092075e-01   0.00000000e+00   0.00000000e+00   5.56871869e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79240296e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09750835e+01]]'

		R_2B_exact_pecnut_doubledist_string = '[[  3.75630809e-03   0.00000000e+00   0.00000000e+00  -1.37201117e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.48806399e-06   0.00000000e+00   0.00000000e+00   4.42589051e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.05863507e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.22789237e-04   0.00000000e+00]\n\
											   [  0.00000000e+00   3.75630809e-03   0.00000000e+00   0.00000000e+00  -1.37201117e-03   0.00000000e+00  -3.48806399e-06   0.00000000e+00   0.00000000e+00  -4.42589051e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.05863507e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.22789237e-04]\n\
											   [  0.00000000e+00   0.00000000e+00   3.12795722e-02   0.00000000e+00   0.00000000e+00  -2.28500165e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.74520041e-02   0.00000000e+00  -1.74520041e-02   0.00000000e+00   0.00000000e+00  -1.00702570e-02   0.00000000e+00  -1.00702570e-02   0.00000000e+00   0.00000000e+00]\n\
											   [ -1.37201117e-03   0.00000000e+00   0.00000000e+00   3.75630809e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.42589051e-04   0.00000000e+00   0.00000000e+00  -3.48806399e-06   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.22789237e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.05863507e-03   0.00000000e+00]\n\
											   [  0.00000000e+00  -1.37201117e-03   0.00000000e+00   0.00000000e+00   3.75630809e-03   0.00000000e+00   4.42589051e-04   0.00000000e+00   0.00000000e+00   3.48806399e-06   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.22789237e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.05863507e-03]\n\
											   [  0.00000000e+00   0.00000000e+00  -2.28500165e-02   0.00000000e+00   0.00000000e+00   3.12795722e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00702570e-02   0.00000000e+00   1.00702570e-02   0.00000000e+00   0.00000000e+00   1.74520041e-02   0.00000000e+00   1.74520041e-02   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00  -3.48806399e-06   0.00000000e+00   0.00000000e+00   4.42589051e-04   0.00000000e+00   2.87831545e-03   0.00000000e+00   0.00000000e+00  -1.85732698e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.34144141e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.72439235e-04]\n\
											   [  3.48806399e-06   0.00000000e+00   0.00000000e+00  -4.42589051e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.87831545e-03   0.00000000e+00   0.00000000e+00  -1.85732698e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.34144141e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.72439235e-04   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.76701482e-04   0.00000000e+00   0.00000000e+00  -2.68821315e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00  -4.42589051e-04   0.00000000e+00   0.00000000e+00   3.48806399e-06   0.00000000e+00  -1.85732698e-04   0.00000000e+00   0.00000000e+00   2.87831545e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.72439235e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.34144141e-03]\n\
											   [  4.42589051e-04   0.00000000e+00   0.00000000e+00  -3.48806399e-06   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.85732698e-04   0.00000000e+00   0.00000000e+00   2.87831545e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.72439235e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.34144141e-03   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.68821315e-05   0.00000000e+00   0.00000000e+00   8.76701482e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00  -1.74520041e-02   0.00000000e+00   0.00000000e+00   1.00702570e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.06635350e-02   0.00000000e+00   1.00034484e-02   0.00000000e+00   0.00000000e+00   3.75850358e-03   0.00000000e+00   3.75753872e-03   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.60086652e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.64860386e-07   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00  -1.74520041e-02   0.00000000e+00   0.00000000e+00   1.00702570e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00034484e-02   0.00000000e+00   1.06635350e-02   0.00000000e+00   0.00000000e+00   3.75753872e-03   0.00000000e+00   3.75850358e-03   0.00000000e+00   0.00000000e+00]\n\
											   [  4.05863507e-03   0.00000000e+00   0.00000000e+00  -5.22789237e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.34144141e-03   0.00000000e+00   0.00000000e+00   6.72439235e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.40455031e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.97939350e-04   0.00000000e+00]\n\
											   [  0.00000000e+00   4.05863507e-03   0.00000000e+00   0.00000000e+00  -5.22789237e-04   0.00000000e+00   3.34144141e-03   0.00000000e+00   0.00000000e+00  -6.72439235e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.40455031e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.97939350e-04]\n\
											   [  0.00000000e+00   0.00000000e+00  -1.00702570e-02   0.00000000e+00   0.00000000e+00   1.74520041e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.75850358e-03   0.00000000e+00   3.75753872e-03   0.00000000e+00   0.00000000e+00   1.06635350e-02   0.00000000e+00   1.00034484e-02   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.64860386e-07   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.60086652e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											   [  0.00000000e+00   0.00000000e+00  -1.00702570e-02   0.00000000e+00   0.00000000e+00   1.74520041e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.75753872e-03   0.00000000e+00   3.75850358e-03   0.00000000e+00   0.00000000e+00   1.00034484e-02   0.00000000e+00   1.06635350e-02   0.00000000e+00   0.00000000e+00]\n\
											   [  5.22789237e-04   0.00000000e+00   0.00000000e+00  -4.05863507e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.72439235e-04   0.00000000e+00   0.00000000e+00  -3.34144141e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.97939350e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.40455031e-03   0.00000000e+00]\n\
											   [  0.00000000e+00   5.22789237e-04   0.00000000e+00   0.00000000e+00  -4.05863507e-03   0.00000000e+00  -6.72439235e-04   0.00000000e+00   0.00000000e+00   3.34144141e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.97939350e-04   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.40455031e-03]]'

		R_grand_pecnut_doubledist_string = '[[  1.95751614e+01   0.00000000e+00   0.00000000e+00  -3.63782443e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.17166238e-01   0.00000000e+00   0.00000000e+00   1.11269215e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92487930e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.23054445e-01   0.00000000e+00]\n\
											[  0.00000000e+00   1.95751614e+01   0.00000000e+00   0.00000000e+00  -3.63782443e+00   0.00000000e+00  -2.17166238e-01   0.00000000e+00   0.00000000e+00  -1.11269215e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92487930e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.23054445e-01]\n\
											[  0.00000000e+00   0.00000000e+00   2.16896478e+01   0.00000000e+00   0.00000000e+00  -7.55356200e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.73941533e-01   0.00000000e+00  -6.73941533e-01   0.00000000e+00   0.00000000e+00  -1.65978036e+00   0.00000000e+00  -1.65978036e+00   0.00000000e+00   0.00000000e+00]\n\
											[ -3.63782443e+00   0.00000000e+00   0.00000000e+00   1.95751614e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.11269215e+00   0.00000000e+00   0.00000000e+00  -2.17166238e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.23054445e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.92487930e-02   0.00000000e+00]\n\
											[  0.00000000e+00  -3.63782443e+00   0.00000000e+00   0.00000000e+00   1.95751614e+01   0.00000000e+00   1.11269215e+00   0.00000000e+00   0.00000000e+00   2.17166238e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.23054445e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.92487930e-02]\n\
											[  0.00000000e+00   0.00000000e+00  -7.55356200e+00   0.00000000e+00   0.00000000e+00   2.16896478e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.65978036e+00   0.00000000e+00   1.65978036e+00   0.00000000e+00   0.00000000e+00   6.73941533e-01   0.00000000e+00   6.73941533e-01   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00  -2.17166238e-01   0.00000000e+00   0.00000000e+00   1.11269215e+00   0.00000000e+00   2.52171706e+01   0.00000000e+00   0.00000000e+00   1.82130117e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.91016010e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.03764515e-01]\n\
											[  2.17166238e-01   0.00000000e+00   0.00000000e+00  -1.11269215e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.52171706e+01   0.00000000e+00   0.00000000e+00   1.82130117e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.91016010e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03764515e-01   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51381975e+01   0.00000000e+00   0.00000000e+00  -3.39316926e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00  -1.11269215e+00   0.00000000e+00   0.00000000e+00   2.17166238e-01   0.00000000e+00   1.82130117e-01   0.00000000e+00   0.00000000e+00   2.52171706e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -6.03764515e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.91016010e-03]\n\
											[  1.11269215e+00   0.00000000e+00   0.00000000e+00  -2.17166238e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.82130117e-01   0.00000000e+00   0.00000000e+00   2.52171706e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03764515e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.91016010e-03   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.39316926e-01   0.00000000e+00   0.00000000e+00   2.51381975e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -6.73941533e-01   0.00000000e+00   0.00000000e+00   1.65978036e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11163787e+01   0.00000000e+00   1.71718581e-01   0.00000000e+00   0.00000000e+00   6.56579946e-01   0.00000000e+00   6.88630099e-01   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09446602e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.20501525e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -6.73941533e-01   0.00000000e+00   0.00000000e+00   1.65978036e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.71718581e-01   0.00000000e+00   2.11163787e+01   0.00000000e+00   0.00000000e+00   6.88630099e-01   0.00000000e+00   6.56579946e-01   0.00000000e+00   0.00000000e+00]\n\
											[  4.92487930e-02   0.00000000e+00   0.00000000e+00  -1.23054445e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.91016010e-03   0.00000000e+00   0.00000000e+00   6.03764515e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09834881e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79938235e-01   0.00000000e+00]\n\
											[  0.00000000e+00   4.92487930e-02   0.00000000e+00   0.00000000e+00  -1.23054445e-01   0.00000000e+00   8.91016010e-03   0.00000000e+00   0.00000000e+00  -6.03764515e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09834881e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79938235e-01]\n\
											[  0.00000000e+00   0.00000000e+00  -1.65978036e+00   0.00000000e+00   0.00000000e+00   6.73941533e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.56579946e-01   0.00000000e+00   6.88630099e-01   0.00000000e+00   0.00000000e+00   2.11163787e+01   0.00000000e+00   1.71718581e-01   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.20501525e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09446602e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
											[  0.00000000e+00   0.00000000e+00  -1.65978036e+00   0.00000000e+00   0.00000000e+00   6.73941533e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.88630099e-01   0.00000000e+00   6.56579946e-01   0.00000000e+00   0.00000000e+00   1.71718581e-01   0.00000000e+00   2.11163787e+01   0.00000000e+00   0.00000000e+00]\n\
											[  1.23054445e-01   0.00000000e+00   0.00000000e+00  -4.92487930e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.03764515e-01   0.00000000e+00   0.00000000e+00  -8.91016010e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79938235e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09834881e+01   0.00000000e+00]\n\
											[  0.00000000e+00   1.23054445e-01   0.00000000e+00   0.00000000e+00  -4.92487930e-02   0.00000000e+00  -6.03764515e-01   0.00000000e+00   0.00000000e+00   8.91016010e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.79938235e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09834881e+01]]'

		M_infinity_pecnut_doubledist = array_from_string(M_infinity_pecnut_doubledist_string)[:6,:6]
		M_infinity_inv_pecnut_doubledist = array_from_string(M_infinity_inv_pecnut_doubledist_string)[:6,:6]
		R_2B_exact_pecnut_doubledist = array_from_string(R_2B_exact_pecnut_doubledist_string)[:6,:6]
		R_grand_pecnut_doubledist = array_from_string(R_grand_pecnut_doubledist_string)[:6,:6]

		R_pybrown = np.linalg.inv(M_pybrown_rpy) + JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(R_pybrown[i][j], R_grand_pecnut_doubledist[i][j], delta = 0.005)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut_doubledist[i][j], delta = 0.005)

		R_diff_infinity_doubledist_coupling = M_infinity_inv_pecnut_doubledist - np.linalg.inv(M_infinity_pecnut_doubledist)

		# more needed

	#-------------------------------------------------------------------------------

	def test_pecnut_RPY_3inline_3(self):

		insanely_large_box_size = 1000000.0

		beads = [ Bead([-3.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([3.0, 0.0, 0.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)

		M_infinity_pecnut = np.array( [ [ 0.05305169, 0.0, 0.0, 0.02456095, 0.0, 0.0, 0.0130173, 0.0, 0.0 ],
										[ 0.0, 0.05305169, 0.0, 0.0, 0.01424535, 0.0, 0.0, 0.00675426, 0.0 ],
										[ 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.01424535, 0.0, 0.0, 0.00675426 ],
										[ 0.02456095, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.02456095, 0.0, 0.0 ],
										[ 0.0, 0.01424535, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.01424535, 0.0 ],
										[ 0.0, 0.0, 0.01424535, 0.0, 0.0, 0.05305169, 0.0, 0.0, 0.01424535 ],
										[ 0.0130173, 0.0, 0.0, 0.02456095, 0.0, 0.0, 0.05305169, 0.0, 0.0 ],
										[ 0.0, 0.00675426, 0.0, 0.0, 0.01424535, 0.0, 0.0, 0.05305169, 0.0 ],
										[ 0.0, 0.0, 0.00675426, 0.0, 0.0, 0.01424535, 0.0, 0.0, 0.05305169 ] ] )

		for i in range(9):
			for j in range(9):
				self.assertAlmostEqual(M_pybrown_rpy[i][j], M_infinity_pecnut[i][j], places = 7)	

	#-------------------------------------------------------------------------------

	def test_pecnut_full_3inline_3(self):

		insanely_large_box_size = 1000000.0

		beads = [ Bead([-3.0, 0.0, 0.0], 1.0), Bead([0.0, 0.0, 0.0], 1.0), Bead([3.0, 0.0, 0.0], 1.0) ]
		pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		M_infinity_pecnut_string = '[[ 0.05305169  0.          0.          0.02456095  0.          0.          0.0130173   0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.05305169  0.          0.          0.01424535  0.          0.          0.00675426  0.          0.          0.          0.        ]\n\
									[ 0.          0.          0.05305169  0.          0.          0.01424535  0.          0.          0.00675426  0.          0.          0.        ]\n\
									[ 0.02456095  0.          0.          0.05305169  0.          0.          0.02456095  0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.01424535  0.          0.          0.05305169  0.          0.          0.01424535  0.          0.          0.          0.00442097]\n\
									[ 0.          0.          0.01424535  0.          0.          0.05305169  0.          0.          0.01424535  0.         -0.00442097  0.        ]\n\
									[ 0.0130173   0.          0.          0.02456095  0.          0.          0.05305169  0.          0.          0.          0.          0.        ]\n\
									[ 0.          0.00675426  0.          0.          0.01424535  0.          0.          0.05305169  0.          0.          0.          0.00110524]\n\
									[ 0.          0.          0.00675426  0.          0.          0.01424535  0.          0.          0.05305169  0.         -0.00110524  0.        ]\n\
									[ 0.          0.          0.          0.          0.          0.          0.          0.          0.          0.03978874  0.          0.        ]\n\
									[ 0.          0.          0.          0.          0.         -0.00442097  0.          0.         -0.00110524  0.          0.03978874  0.        ]\n\
									[ 0.          0.          0.          0.          0.00442097  0.          0.          0.00110524  0.          0.          0.          0.03978874]]'

		M_infinity_inv_pecnut = '[[  2.55581573e+01   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.88710420e+00   0.00000000e+00  -7.73597239e-01   0.00000000e+00   0.00000000e+00   5.38593665e+00   0.00000000e+00  -1.44315738e+00   0.00000000e+00   0.00000000e+00   8.74759578e-02   0.00000000e+00  -2.34391122e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.06563410e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   2.06563410e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00]\n\
								 [ -1.13468776e+01   0.00000000e+00   0.00000000e+00   3.10568540e+01   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89466878e+00   0.00000000e+00   1.31152255e+00   0.00000000e+00   0.00000000e+00   2.23489657e-16   0.00000000e+00  -7.37187641e-18   0.00000000e+00   0.00000000e+00   4.89466878e+00   0.00000000e+00  -1.31152255e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.20502332e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   5.32932034e-19   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.91175850e-18   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.20502332e+01   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00  -9.90005418e-17   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.62542038e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00]\n\
								 [ -2.09961760e+00   0.00000000e+00   0.00000000e+00  -1.13468776e+01   0.00000000e+00   0.00000000e+00   2.55581573e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.74759578e-02   0.00000000e+00   2.34391122e-02   0.00000000e+00   0.00000000e+00  -5.38593665e+00   0.00000000e+00   1.44315738e+00   0.00000000e+00   0.00000000e+00  -2.88710420e+00   0.00000000e+00   7.73597239e-01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.06563410e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.20921999e+00   0.00000000e+00   0.00000000e+00   2.06563410e+01   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51675321e+01   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   7.09538314e-01   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   2.52015627e+01   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00  -7.48412285e-17   0.00000000e+00   0.00000000e+00   2.49610961e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.59742750e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.13827075e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.49610961e+00   0.00000000e+00   0.00000000e+00   7.24168089e-17   0.00000000e+00   0.00000000e+00  -2.49610961e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.59742750e+01   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.13827075e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00  -9.29089869e-01   0.00000000e+00   0.00000000e+00   2.51675321e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.29517917e+00   0.00000000e+00   0.00000000e+00   7.09538314e-01   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.29517917e+00   0.00000000e+00   0.00000000e+00  -7.09538314e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.67374832e-01   0.00000000e+00   0.00000000e+00   2.55338141e+01   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  2.88710420e+00   0.00000000e+00   0.00000000e+00  -4.89466878e+00   0.00000000e+00   0.00000000e+00  -8.74759578e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.22956686e+01   0.00000000e+00  -3.61811245e-01   0.00000000e+00   0.00000000e+00   3.38627827e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   0.00000000e+00   1.81972375e-01   0.00000000e+00  -4.98227358e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   2.11418817e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -7.73597239e-01   0.00000000e+00   0.00000000e+00   1.31152255e+00   0.00000000e+00   0.00000000e+00   2.34391122e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.61811245e-01   0.00000000e+00   2.10423177e+01   0.00000000e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   8.31413144e-02   0.00000000e+00   0.00000000e+00  -4.98227358e-02   0.00000000e+00   9.38135593e-03   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   2.54312001e-01   0.00000000e+00   0.00000000e+00  -4.92139155e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453706e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03]\n\
								 [  5.38593665e+00   0.00000000e+00   0.00000000e+00   4.83707030e-16   0.00000000e+00   0.00000000e+00  -5.38593665e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.38627827e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   0.00000000e+00   2.38073391e+01   0.00000000e+00  -7.66482303e-01   0.00000000e+00   0.00000000e+00   3.38627827e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00   0.00000000e+00  -1.13827075e-02   0.00000000e+00   0.00000000e+00   1.68323559e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13386866e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -1.44315738e+00   0.00000000e+00   0.00000000e+00  -1.25825873e-16   0.00000000e+00   0.00000000e+00   1.44315738e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   8.31413144e-02   0.00000000e+00   0.00000000e+00  -7.66482303e-01   0.00000000e+00   2.11521665e+01   0.00000000e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   8.31413144e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   5.49522173e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.49522173e-01   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   1.13827075e-02   0.00000000e+00   0.00000000e+00  -1.68323559e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.13386866e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09467882e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01]\n\
								 [  8.74759578e-02   0.00000000e+00   0.00000000e+00   4.89466878e+00   0.00000000e+00   0.00000000e+00  -2.88710420e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.81972375e-01   0.00000000e+00  -4.98227358e-02   0.00000000e+00   0.00000000e+00   3.38627827e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   0.00000000e+00   2.22956686e+01   0.00000000e+00  -3.61811245e-01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   1.69172452e+00   0.00000000e+00   0.00000000e+00  -1.09930645e-02   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -2.34391122e-02   0.00000000e+00   0.00000000e+00  -1.31152255e+00   0.00000000e+00   0.00000000e+00   7.73597239e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.98227358e-02   0.00000000e+00   9.38135593e-03   0.00000000e+00   0.00000000e+00  -9.53533505e-01   0.00000000e+00   8.31413144e-02   0.00000000e+00   0.00000000e+00  -3.61811245e-01   0.00000000e+00   2.10423177e+01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.92139155e-01   0.00000000e+00   0.00000000e+00  -2.54312001e-01   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00  -1.69172452e+00   0.00000000e+00   0.00000000e+00   1.09930645e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.24632325e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.11418817e+01   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72357218e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09453706e+01]]'

		R_2B_exact_pecnut_string = '[[  4.10639028e-01   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.49123543e-01   0.00000000e+00  -9.35473714e-02   0.00000000e+00   0.00000000e+00   2.91683696e-01   0.00000000e+00  -7.81564108e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[ -3.79760138e-01   0.00000000e+00   0.00000000e+00   8.21278057e-01   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91683696e-01   0.00000000e+00   7.81564108e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.91683696e-01   0.00000000e+00  -7.81564108e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   6.67633335e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   6.67633335e-02   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.79760138e-01   0.00000000e+00   0.00000000e+00   4.10639028e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91683696e-01   0.00000000e+00   7.81564108e-02   0.00000000e+00   0.00000000e+00  -3.49123543e-01   0.00000000e+00   9.35473714e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.91139529e-02   0.00000000e+00   0.00000000e+00   3.33816668e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   2.94318406e-02   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   1.03581762e-01   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.94036332e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00]\n\
									[  0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   1.03581762e-01   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.94036332e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.44363935e-03   0.00000000e+00   0.00000000e+00   1.47159203e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.33129971e-02   0.00000000e+00   0.00000000e+00   9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.33129971e-02   0.00000000e+00   0.00000000e+00  -9.89276573e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.62354107e-03   0.00000000e+00   0.00000000e+00   5.17908809e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  3.49123543e-01   0.00000000e+00   0.00000000e+00  -2.91683696e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.18563800e-01   0.00000000e+00  -8.28206823e-02   0.00000000e+00   0.00000000e+00   2.05621957e-01   0.00000000e+00  -5.50824295e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[ -9.35473714e-02   0.00000000e+00   0.00000000e+00   7.81564108e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.28206823e-02   0.00000000e+00   3.16645404e-02   0.00000000e+00   0.00000000e+00  -5.50824295e-02   0.00000000e+00   1.48108241e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   3.67649077e-02   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  2.91683696e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91683696e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05621957e-01   0.00000000e+00  -5.50824295e-02   0.00000000e+00   0.00000000e+00   6.37127599e-01   0.00000000e+00  -1.65641365e-01   0.00000000e+00   0.00000000e+00   2.05621957e-01   0.00000000e+00  -5.50824295e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00  -8.94036332e-02   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.14249790e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[ -7.81564108e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.81564108e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.50824295e-02   0.00000000e+00   1.48108241e-02   0.00000000e+00   0.00000000e+00  -1.65641365e-01   0.00000000e+00   6.33290808e-02   0.00000000e+00   0.00000000e+00  -5.50824295e-02   0.00000000e+00   1.48108241e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.85209181e-03   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   8.94036332e-02   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.14249790e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.89456109e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   2.91683696e-01   0.00000000e+00   0.00000000e+00  -3.49123543e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.05621957e-01   0.00000000e+00  -5.50824295e-02   0.00000000e+00   0.00000000e+00   3.18563800e-01   0.00000000e+00  -8.28206823e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.28730056e-02   0.00000000e+00   0.00000000e+00  -4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00  -7.81564108e-02   0.00000000e+00   0.00000000e+00   9.35473714e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.50824295e-02   0.00000000e+00   1.48108241e-02   0.00000000e+00   0.00000000e+00  -8.28206823e-02   0.00000000e+00   3.16645404e-02   0.00000000e+00   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.85209181e-03   0.00000000e+00   0.00000000e+00  -3.67649077e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.28730056e-02   0.00000000e+00   0.00000000e+00   4.47018166e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.51713804e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.07124895e-01   0.00000000e+00]\n\
									[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.15315976e-05   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.47280544e-03]]'

		R_grand_pecnut_string = '[[  2.59687963e+01   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00  -2.09961760e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.23622774e+00   0.00000000e+00  -8.67144611e-01   0.00000000e+00   0.00000000e+00   5.67762035e+00   0.00000000e+00  -1.52131379e+00   0.00000000e+00   0.00000000e+00   8.74759578e-02   0.00000000e+00  -2.34391122e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.06897227e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   2.06897227e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00]\n\
								 [ -1.17266378e+01   0.00000000e+00   0.00000000e+00   3.18781321e+01   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.18635247e+00   0.00000000e+00   1.38967896e+00   0.00000000e+00   0.00000000e+00   2.23489657e-16   0.00000000e+00  -7.37187641e-18   0.00000000e+00   0.00000000e+00   5.18635247e+00   0.00000000e+00  -1.38967896e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.21169965e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   5.32932034e-19   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.91175850e-18   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.21169965e+01   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00  -9.90005418e-17   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.62542038e-17   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00]\n\
								 [ -2.09961760e+00   0.00000000e+00   0.00000000e+00  -1.17266378e+01   0.00000000e+00   0.00000000e+00   2.59687963e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.74759578e-02   0.00000000e+00   2.34391122e-02   0.00000000e+00   0.00000000e+00  -5.67762035e+00   0.00000000e+00   1.52131379e+00   0.00000000e+00   0.00000000e+00  -3.23622774e+00   0.00000000e+00   8.67144611e-01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.06897227e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -1.46572236e+00   0.00000000e+00   0.00000000e+00  -5.22833394e+00   0.00000000e+00   0.00000000e+00   2.06897227e+01   0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.51822480e+01   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00   8.85238492e-02   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   7.19431080e-01   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   2.52309945e+01   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00  -7.48412285e-17   0.00000000e+00   0.00000000e+00   2.50942260e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.60778568e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00786341e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.50942260e+00   0.00000000e+00   0.00000000e+00   7.24168089e-17   0.00000000e+00   0.00000000e+00  -2.50942260e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.60778568e+01   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.00786341e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -8.21056163e-02   0.00000000e+00   0.00000000e+00  -9.30533508e-01   0.00000000e+00   0.00000000e+00   2.51822480e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00  -8.85238492e-02   0.00000000e+00   0.00000000e+00  -2.30849217e+00   0.00000000e+00   0.00000000e+00   7.19431080e-01   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   8.85238492e-02   0.00000000e+00   0.00000000e+00   2.30849217e+00   0.00000000e+00   0.00000000e+00  -7.19431080e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.35580018e-02   0.00000000e+00   0.00000000e+00   5.61751291e-01   0.00000000e+00   0.00000000e+00   2.55856050e+01   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  3.23622774e+00   0.00000000e+00   0.00000000e+00  -5.18635247e+00   0.00000000e+00   0.00000000e+00  -8.74759578e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.26142324e+01   0.00000000e+00  -4.44631927e-01   0.00000000e+00   0.00000000e+00   3.59190023e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   0.00000000e+00   1.81972375e-01   0.00000000e+00  -4.98227358e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   2.12490066e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -8.67144611e-01   0.00000000e+00   0.00000000e+00   1.38967896e+00   0.00000000e+00   0.00000000e+00   2.34391122e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.44631927e-01   0.00000000e+00   2.10739822e+01   0.00000000e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   9.79521385e-02   0.00000000e+00   0.00000000e+00  -4.98227358e-02   0.00000000e+00   9.38135593e-03   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   2.91076909e-01   0.00000000e+00   0.00000000e+00  -4.97991247e-01   0.00000000e+00   0.00000000e+00  -4.25494712e-02   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09548434e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03]\n\
								 [  5.67762035e+00   0.00000000e+00   0.00000000e+00   4.83707030e-16   0.00000000e+00   0.00000000e+00  -5.67762035e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   3.59190023e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   0.00000000e+00   2.44444667e+01   0.00000000e+00  -9.32123667e-01   0.00000000e+00   0.00000000e+00   3.59190023e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00   0.00000000e+00  -1.00786341e-01   0.00000000e+00   0.00000000e+00   1.70610860e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.15529364e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -1.52131379e+00   0.00000000e+00   0.00000000e+00  -1.25825873e-16   0.00000000e+00   0.00000000e+00   1.52131379e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   9.79521385e-02   0.00000000e+00   0.00000000e+00  -9.32123667e-01   0.00000000e+00   2.12154956e+01   0.00000000e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   9.79521385e-02   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   5.55374265e-01   0.00000000e+00   0.00000000e+00   3.68892681e-17   0.00000000e+00   0.00000000e+00  -5.55374265e-01   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   1.00786341e-01   0.00000000e+00   0.00000000e+00  -1.70610860e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.15529364e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09657338e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01]\n\
								 [  8.74759578e-02   0.00000000e+00   0.00000000e+00   5.18635247e+00   0.00000000e+00   0.00000000e+00  -3.23622774e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   1.81972375e-01   0.00000000e+00  -4.98227358e-02   0.00000000e+00   0.00000000e+00   3.59190023e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   0.00000000e+00   2.26142324e+01   0.00000000e+00  -4.44631927e-01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   9.33973638e-02   0.00000000e+00   0.00000000e+00   1.71459752e+00   0.00000000e+00   0.00000000e+00  -5.56948811e-02   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00   0.00000000e+00   0.00000000e+00]\n\
								 [ -2.34391122e-02   0.00000000e+00   0.00000000e+00  -1.38967896e+00   0.00000000e+00   0.00000000e+00   8.67144611e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.98227358e-02   0.00000000e+00   9.38135593e-03   0.00000000e+00   0.00000000e+00  -1.00861593e+00   0.00000000e+00   9.79521385e-02   0.00000000e+00   0.00000000e+00  -4.44631927e-01   0.00000000e+00   2.10739822e+01   0.00000000e+00   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   4.25494712e-02   0.00000000e+00   0.00000000e+00   4.97991247e-01   0.00000000e+00   0.00000000e+00  -2.91076909e-01   0.00000000e+00  -9.33973638e-02   0.00000000e+00   0.00000000e+00  -1.71459752e+00   0.00000000e+00   0.00000000e+00   5.56948811e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -4.89922688e-02   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.27149463e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.12490066e+01   0.00000000e+00]\n\
								 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -3.96860587e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.72305687e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.09548434e+01]]'

		M_infinity_pecnut = array_from_string(M_infinity_pecnut_string)
		M_infinity_inv_pecnut = array_from_string(M_infinity_inv_pecnut)
		R_2B_exact_pecnut = array_from_string(R_2B_exact_pecnut_string)
		R_grand_pecnut = array_from_string(R_grand_pecnut_string)

		M_infinity_inv_pecnut_contracted = M_infinity_inv_pecnut[:9,:9]
		R_2B_exact_pecnut_contracted = R_2B_exact_pecnut[:9,:9]
		R_grand_pecnut_contracted = R_grand_pecnut[:9,:9]

		# print( R_2B_exact_pecnut_contracted )
		# print( JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False) )

		# 1/0

		for i in range(len(R_grand_pecnut)):
			for j in range(len(R_grand_pecnut)):
				self.assertAlmostEqual(R_grand_pecnut[i][j], M_infinity_inv_pecnut[i][j] + R_2B_exact_pecnut[i][j], places = 5)


	# def test_JO_R_3inline_3(self):

	# 	insanely_large_box_size = 1000000.0)

	# 	beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([3.0, 0.0, 0.0], 1.0), Bead([6.0, 0.0, 0.0], 1.0) ]
	# 	pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

	# 	M_pybrown_rpy = RPY_M_matrix(beads, pointers)
	# 	R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = False)
	# 	R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

	# 	R_hydrolib_3inline_3 = np.array( [ [ 3.7529350624336963, 0.0, 0.0, -1.7116189225981648, 0.0, 0.0, -0.27692827267588010, 0.0, 0.0 ],
	# 									   [ 0.0, 2.9938007440202492, 0.0, 0.0, -0.75775272664134452, 0.0, 0.0, -0.20916792627928388, 0.0 ],
	# 									   [ 0.0, 0.0, 2.9938007439852643, 0.0, 0.0, -0.75775272657922133, 0.0, 0.0, -0.20916792627095443 ],
	# 									   [ -1.7116189225981653, 0.0, 0.0, 4.6310954490348335, 0.0, 0.0, -1.7116189225981642, 0.0, 0.0 ],
	# 									   [ 0.0, -0.75775272664134463, 0.0, 0.0, 3.2016238839106537, 0.0, 0.0, -0.75775272664134430, 0.0 ],
	# 									   [ 0.0, 0.0, -0.75775272657922121, 0.0, 0.0, 3.2016238838475726, 0.0, 0.0, -0.75775272657922133 ],
	# 									   [ -0.27692827267588022, 0.0, 0.0, -1.7116189225981646, 0.0, 0.0, 3.7529350624336963, 0.0, 0.0 ],
	# 									   [ 0.0, -0.20916792627928388, 0.0, 0.0, -0.75775272664134463, 0.0, 0.0, 2.9938007440202483, 0.0 ],
	# 									   [ 0.0, 0.0, -0.20916792627095429, 0.0, 0.0, -0.75775272657922177, 0.0, 0.0, 2.9938007439852639 ] ] )

	# 	R_hydrolib_3inline_3 /= self.R_hydrolib_infty

	# 	dR12 = ( JO_2B_R_matrix(beads[0], beads[1]) - np.linalg.inv( RPY_M_matrix([beads[0], beads[1]], compute_pointer_pbc_matrix([beads[0], beads[1]], 1000000.0)) ) ) / self.R_pybrown_infty
	# 	dR13 = ( JO_2B_R_matrix(beads[0], beads[2]) - np.linalg.inv( RPY_M_matrix([beads[0], beads[2]], compute_pointer_pbc_matrix([beads[0], beads[2]], 1000000.0)) ) ) / self.R_pybrown_infty
	# 	dR23 = ( JO_2B_R_matrix(beads[1], beads[2]) - np.linalg.inv( RPY_M_matrix([beads[1], beads[2]], compute_pointer_pbc_matrix([beads[1], beads[2]], 1000000.0)) ) ) / self.R_pybrown_infty
	# 	# print(dR12)
	# 	# print(dR13)
	# 	# print(dR23)
	# 	print(R_pybrown_lub_corr/self.R_pybrown_infty)
	# 	print(np.linalg.inv(M_pybrown_rpy)/self.R_pybrown_infty)

	# 	# print('pyBrown:\n{}'.format(R_pybrown_tot))
	# 	print('hydrolib:\n{}'.format(R_hydrolib_3inline_3))
	# 	1/0

	# 	for i in range(9):
	# 		for j in range(9):
	# 			self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_3inline_3[i][j], delta = 0.025)

	# def test_JO_R_3inline_2_1(self):

		# insanely_large_box_size = 1000000.0

		# beads = [ Bead([0.0, 0.0, 0.0], 1.0), Bead([2.1, 0.0, 0.0], 1.0), Bead([4.2, 0.0, 0.0], 1.0) ]
		# pointers = compute_pointer_pbc_matrix(beads, insanely_large_box_size)

		# M_pybrown_rpy = RPY_M_matrix(beads, pointers)
		# print( np.linalg.inv(M_pybrown_rpy) / self.R_pybrown_infty )
		# print( JO_2B_R_matrix(beads[0], beads[2]) / self.R_pybrown_infty )
		# print( np.linalg.inv(RPY_M_matrix([ beads[0], beads[2] ], compute_pointer_pbc_matrix([ beads[0], beads[2] ], insanely_large_box_size))) / self.R_pybrown_infty )
		# 1/0
		# R_pybrown_lub_corr = JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff = insanely_large_box_size, cichocki_correction = True)
		# R_pybrown_tot = ( R_pybrown_lub_corr + np.linalg.inv(M_pybrown_rpy) ) / self.R_pybrown_infty

		# full, multipole order 1, no lubrication
		# R_hydrolib_3inline_2_1 = np.array( [ [ 5.8257147072732396, 0.0, 0.0, 0.0, 0.0, 0.0, -3.5477023932754559, 0.0, 0.0, 0.0, 0.0, 0.0, -0.67602449028508182, 0.0, 0.0, 0.0, 0.0, 0.0 ],
		# 									 [ 0.0, 3.6796272463930726, 0.0, 0.0, 0.0, 0.56571935644572513, 0.0, -1.5734468590732698, 0.0, 0.0, 0.0, 1.0166131458580756, 0.0, -0.24154179479807453, 0.0, 0.0, 0.0, 1.4014244051598401E-002 ],
		# 									 [ 0.0, 0.0, 3.6796272463102073, 0.0, -0.56571935640308946, 0.0, 0.0, 0.0, -1.5734468589664299, 0.0, -1.0166131458353538, 0.0, 0.0, 0.0, -0.24154179479206847, 0.0, -1.4014244051355726E-002, 0.0 ],
		# 									 [ 0.0, 0.0, 0.0, 3.7116574497574129, 0.0, 0.0, 0.0, 0.0, 0.0, -0.40124635934215841, 0.0, 0.0, 0.0, 0.0, 0.0, -3.7666888594849061E-002, 0.0, 0.0 ],
		# 									 [ 0.0, 0.0, -0.56571935640308935, 0.0, 4.0930792100619220, 0.0, 0.0, 0.0, 0.94652774495011249, 0.0, 0.40080224007586790, 0.0, 0.0, 0.0, 1.4014244051355974E-002, 0.0, 5.9097712329172006E-002, 0.0 ],
		# 									 [ 0.0, 0.56571935644572502, 0.0, 0.0, 0.0, 4.0930792100805018, 0.0, -0.94652774497761472, 0.0, 0.0, 0.0, 0.40080224014093180, 0.0, -1.4014244051598323E-002, 0.0, 0.0, 0.0, 5.9097712346172289E-002 ],
		# 									 [ -3.5477023932754554, 0.0, 0.0, 0.0, 0.0, 0.0, 8.1198071625007344, 0.0, 0.0, 0.0, 0.0, 0.0, -3.5477023932754554, 0.0, 0.0, 0.0, 0.0, 0.0 ],
		# 									 [ 0.0, -1.5734468590732695, 0.0, 0.0, 0.0, -0.94652774497761483, 0.0, 4.5190830082420259, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5734468590732700, 0.0, 0.0, 0.0, 0.94652774497761483 ],
		# 									 [ 0.0, 0.0, -1.5734468589664290, 0.0, 0.94652774495011205, 0.0, 0.0, 0.0, 4.5190830080828190, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5734468589664286, 0.0, -0.94652774495011227, 0.0 ],
		# 									 [ 0.0, 0.0, 0.0, -0.40124635934215841, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7832995782481875, 0.0, 0.0, 0.0, 0.0, 0.0, -0.40124635934215841, 0.0, 0.0 ], 
		# 									 [ 0.0, 0.0, -1.0166131458353538, 0.0, 0.40080224007586790, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5607825418266597, 0.0, 0.0, 0.0, 1.0166131458353540, 0.0, 0.40080224007586801, 0.0 ],
		# 									 [ 0.0, 1.0166131458580756, 0.0, 0.0, 0.0, 0.40080224014093174, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5607825418503101, 0.0, -1.0166131458580758, 0.0, 0.0, 0.0, 0.40080224014093191 ],
		# 									 [ -0.67602449028508116, 0.0, 0.0, 0.0, 0.0, 0.0, -3.5477023932754550, 0.0, 0.0, 0.0, 0.0, 0.0, 5.8257147072732396, 0.0, 0.0, 0.0, 0.0, 0.0 ],
		# 									 [ 0.0, -0.24154179479807472, 0.0, 0.0, 0.0, -1.4014244051598367E-002, 0.0, -1.5734468590732700, 0.0, 0.0, 0.0, -1.0166131458580756, 0.0, 3.6796272463930735, 0.0, 0.0, 0.0, -0.56571935644572524 ],
		# 									 [ 0.0, 0.0, -0.24154179479206803, 0.0, 1.4014244051355830E-002, 0.0, 0.0, 0.0, -1.5734468589664288, 0.0, 1.0166131458353540, 0.0, 0.0, 0.0, 3.6796272463102073, 0.0, 0.56571935640308957, 0.0 ],
		# 									 [ 0.0, 0.0, 0.0, -3.7666888594849048E-002, 0.0, 0.0, 0.0, 0.0, 0.0, -0.40124635934215841, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7116574497574124, 0.0, 0.0 ],
		# 									 [ 0.0, 0.0, -1.4014244051355748E-002, 0.0, 5.9097712329172054E-002, 0.0, 0.0, 0.0, -0.94652774495011227, 0.0, 0.40080224007586784, 0.0, 0.0, 0.0, 0.56571935640308935, 0.0, 4.0930792100619220, 0.0 ],
		# 									 [ 0.0, 1.4014244051598422E-002, 0.0, 0.0, 0.0, 5.9097712346172199E-002, 0.0, 0.94652774497761505, 0.0, 0.0, 0.0, 0.40080224014093174, 0.0, -0.56571935644572524, 0.0, 0.0, 0.0, 4.0930792100805018 ] ] )

		# multipole order 1
		# R_hydrolib_3inline_2_1 = np.array( [ [ 11.079848365917112, 0.0, 0.0, -8.7946976170685467, 0.0, 0.0, -0.67602449028508182, 0.0, 0.0 ],
		# 									 [ 0.0, 3.8171986600626728, 0.0, 0.0, -1.7093942529760759, 0.0, 0.0, -0.24154179479807453, 0.0 ],
		# 									 [ 0.0, 0.0, 3.8171986599798076, 0.0, 0.0, -1.7093942528692361, 0.0, 0.0, -0.24154179479206847 ],
		# 									 [ -8.7946976170685467, 0.0, 0.0, 18.628074479788481, 0.0, 0.0, -8.7946976170685467, 0.0, 0.0 ],
		# 									 [ 0.0, -1.7093942529760757, 0.0, 0.0, 4.7942258355812264, 0.0, 0.0, -1.7093942529760764, 0.0 ],
		# 									 [ 0.0, 0.0, -1.7093942528692354, 0.0, 0.0, 4.7942258354220195, 0.0, 0.0, -1.7093942528692350 ],
		# 									 [ -0.67602449028508116, 0.0, 0.0, -8.7946976170685467, 0.0, 0.0, 11.079848365917112, 0.0, 0.0 ],
		# 									 [ 0.0, -0.24154179479807472, 0.0, 0.0, -1.7093942529760762, 0.0, 0.0, 3.8171986600626737, 0.0 ],
		# 									 [ 0.0, 0.0, -0.24154179479206803, 0.0, 0.0, -1.7093942528692352, 0.0, 0.0, 3.8171986599798076 ] ] )

		# multipole order 2
		# R_hydrolib_3inline_2_1 = np.array( [ [ 11.043006752001110, 0.0, 0.0, -9.3465688534047793, 0.0, 0.0, -3.9434196031939456E-002, 0.0, 0.0 ],
		# 									 [ 0.0, 3.8181778933677069, 0.0, 0.0, -1.7110644362440994, 0.0, 0.0, -0.24600066269559842, 0.0 ],
		# 									 [ 0.0, 0.0, 3.8181778933207622, 0.0, 0.0, -1.7110644361868184, 0.0, 0.0, -0.24600066267315820 ],
		# 									 [ -9.3465688534047775, 0.0, 0.0, 19.642144785208259, 0.0, 0.0, -9.3465688534047811, 0.0, 0.0 ],
		# 									 [ 0.0, -1.7110644362440997, 0.0, 0.0, 4.8047438758424246, 0.0, 0.0, -1.7110644362440997, 0.0 ],
		# 									 [ 0.0, 0.0, -1.7110644361868184, 0.0, 0.0, 4.8047438757779704, 0.0, 0.0, -1.7110644361868179 ],
		# 									 [ -3.9434196031937700E-002, 0.0, 0.0, -9.3465688534047828, 0.0, 0.0, 11.043006752001114, 0.0, 0.0 ],
		# 									 [ 0.0, -0.24600066269559887, 0.0, 0.0, -1.7110644362440990, 0.0, 0.0, 3.8181778933677060, 0.0 ],
		# 									 [ 0.0, 0.0, -0.24600066267315826, 0.0, 0.0, -1.7110644361868175, 0.0, 0.0, 3.8181778933207613 ] ] )

		# multipole order 3
		# R_hydrolib_3inline_2_1 = np.array( [ [ 11.036264134593569, 0.0, 0.0, -8.8623764781197192, 0.0, 0.0, -0.52247385115128109, 0.0, 0.0 ],
		# 									 [ 0.0, 3.8176290024892876, 0.0, 0.0, -1.7102314080213143, 0.0, 0.0, -0.24489941627605169, 0.0 ],
		# 									 [ 0.0, 0.0, 3.8176290024415125, 0.0, 0.0, -1.7102314079458654, 0.0, 0.0, -0.24489941626514644 ],
		# 									 [ -8.8623764781197174, 0.0, 0.0, 18.684402018144972, 0.0, 0.0, -8.8623764781197245, 0.0, 0.0 ],
		# 									 [ 0.0, -1.7102314080213148, 0.0, 0.0, 4.8002368910874047, 0.0, 0.0, -1.7102314080213143, 0.0 ],
		# 									 [ 0.0, 0.0, -1.7102314079458645, 0.0, 0.0, 4.8002368909719566, 0.0, 0.0, -1.7102314079458654 ],
		# 									 [ -0.52247385115128153, 0.0, 0.0, -8.8623764781197227, 0.0, 0.0, 11.036264134593571, 0.0, 0.0 ],
		# 									 [ 0.0, -0.24489941627605169, 0.0, 0.0, -1.7102314080213152, 0.0, 0.0, 3.8176290024892876, 0.0 ],
		# 									 [ 0.0, 0.0, -0.24489941626514655, 0.0, 0.0, -1.7102314079458649, 0.0, 0.0, 3.8176290024415134 ] ] )

		# R_hydrolib_3inline_2_1 /= self.R_hydrolib_infty

		# ###
		# M_hydrolib_3inline_2_1 = np.zeros((9, 9))
		# Mbig = np.linalg.inv(R_hydrolib_3inline_2_1)
		# M_hydrolib_3inline_2_1[0][0] = Mbig[0][0]
		# M_hydrolib_3inline_2_1[1][1] = Mbig[1][1]
		# M_hydrolib_3inline_2_1[2][2] = Mbig[2][2]
		# M_hydrolib_3inline_2_1[3][3] = Mbig[6][6]
		# M_hydrolib_3inline_2_1[4][4] = Mbig[7][7]
		# M_hydrolib_3inline_2_1[5][5] = Mbig[8][8]
		# M_hydrolib_3inline_2_1[6][6] = Mbig[12][12]
		# M_hydrolib_3inline_2_1[7][7] = Mbig[13][13]
		# M_hydrolib_3inline_2_1[8][8] = Mbig[14][14]
		# M_hydrolib_3inline_2_1[0][3] = M_hydrolib_3inline_2_1[3][0] = Mbig[0][6]
		# M_hydrolib_3inline_2_1[1][4] = M_hydrolib_3inline_2_1[4][1] = Mbig[1][7]
		# M_hydrolib_3inline_2_1[2][5] = M_hydrolib_3inline_2_1[5][2] = Mbig[2][8]
		# M_hydrolib_3inline_2_1[0][6] = M_hydrolib_3inline_2_1[6][0] = Mbig[0][12]
		# M_hydrolib_3inline_2_1[1][7] = M_hydrolib_3inline_2_1[7][1] = Mbig[1][13]
		# M_hydrolib_3inline_2_1[2][8] = M_hydrolib_3inline_2_1[8][2] = Mbig[2][14]
		# M_hydrolib_3inline_2_1[6][3] = M_hydrolib_3inline_2_1[3][6] = Mbig[12][6]
		# M_hydrolib_3inline_2_1[7][4] = M_hydrolib_3inline_2_1[4][7] = Mbig[13][7]
		# M_hydrolib_3inline_2_1[8][5] = M_hydrolib_3inline_2_1[5][8] = Mbig[14][8]
		# print(M_hydrolib_3inline_2_1)
		# print(M_pybrown_rpy/M_pybrown_rpy[0][0])
		# 1/0
		# ###

		# for i in range(9):
		# 	for j in range(9):
		# 		print('pb{} vs h{}'.format(R_pybrown_tot[i][j], R_hydrolib_3inline_2_1[i][j]))
		# 		# self.assertAlmostEqual(R_pybrown_tot[i][j], R_hydrolib_3inline_2_1[i][j], delta = 0.02)

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------