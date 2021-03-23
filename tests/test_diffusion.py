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

import unittest

import sys
sys.path.insert(0, '.')
import numpy as np
import copy as cp

from pyBD.diffusion import O, O_python, Oii_pbc_smith, Oii_pbc_smith_python, Oij_pbc_smith, Oij_pbc_smith_python
from pyBD.diffusion import Q, Q_python, Qii_pbc_smith, Qii_pbc_smith_python, Qij_pbc_smith, Qij_pbc_smith_python
from pyBD.diffusion import Mij_rpy, Mij_rpy_python
from pyBD.diffusion import Mii_rpy_smith, Mii_rpy_smith_python
from pyBD.diffusion import Mij_rpy_smith, Mij_rpy_smith_python
from pyBD.diffusion import X_f_poly, X_f_poly_python, Y_f_poly, Y_f_poly_python
from pyBD.diffusion import X_g_poly, X_g_poly_python, Y_g_poly, Y_g_poly_python
from pyBD.diffusion import XA11, XA11_python, YA11, YA11_python, XA12, XA12_python, YA12, YA12_python
from pyBD.diffusion import R_jeffrey, R_jeffrey_python
from pyBD.diffusion import M_rpy_smith, M_rpy_smith_python

from pyBD.bead import Bead, pointer_pbc

#-------------------------------------------------------------------------------

class TestDiffusion(unittest.TestCase):

	# def test_O_python_vs_c(self):

	# 	for i in range(-5, 6):
	# 		for j in range(-5, 6):
	# 			for k in range(-5, 6):

	# 				if i == 0 and j == 0 and k == 0: continue

	# 				r = np.array([i, j, k])

	# 				c_ish = O(r)
	# 				python_ish = O_python(r)

	# 				for ii in range(3):
	# 					for jj in range(3):
	# 						self.assertAlmostEqual( c_ish[ii][jj], python_ish[ii][jj], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Q_python_vs_c(self):

	# 	for i in range(-5, 6):
	# 		for j in range(-5, 6):
	# 			for k in range(-5, 6):

	# 				if i == 0 and j == 0 and k == 0: continue

	# 				r = np.array([i, j, k])

	# 				c_ish = Q(r)
	# 				python_ish = Q_python(r)

	# 				for ii in range(3):
	# 					for jj in range(3):
	# 						self.assertAlmostEqual( c_ish[ii][jj], python_ish[ii][jj], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Oii_python_vs_c(self):

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 750.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Oii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Oii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 75.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Oii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Oii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 75000.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Oii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Oii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Qii_python_vs_c(self):

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 750.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Qii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Qii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 75.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Qii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Qii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# 	for m in range(6):
	# 		for n in range(6):
	# 			a = 51.0
	# 			L = 75000.0
	# 			alpha = np.sqrt(np.pi)

	# 			c_ish = Qii_pbc_smith(a, L, alpha, m, n)
	# 			python_ish = Qii_pbc_smith_python(a, L, alpha, m, n)

	# 			for i in range(3):
	# 				for j in range(3):
	# 					self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Oij_python_vs_c(self):

	# 	for m in range(6):
	# 		for n in range(6):
	# 			for sigmax in [0.1, 0.5, 0.9]:
	# 				for sigmay in [0.1, 0.5, 0.9]:
	# 					for sigmaz in [0.1, 0.5, 0.9]:

	# 						sigma = np.array([sigmax, sigmay, sigmaz])
	# 						alpha = np.sqrt(np.pi)

	# 						c_ish = Oij_pbc_smith(sigma, alpha, m, n)
	# 						python_ish = Oij_pbc_smith_python(sigma, alpha, m, n)

	# 						for i in range(3):
	# 							for j in range(3):
	# 								self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Qij_python_vs_c(self):

	# 	for m in range(6):
	# 		for n in range(6):
	# 			for sigmax in [0.1, 0.5, 0.9]:
	# 				for sigmay in [0.1, 0.5, 0.9]:
	# 					for sigmaz in [0.1, 0.5, 0.9]:

	# 						sigma = np.array([sigmax, sigmay, sigmaz])
	# 						alpha = np.sqrt(np.pi)

	# 						c_ish = Qij_pbc_smith(sigma, alpha, m, n)
	# 						python_ish = Qij_pbc_smith_python(sigma, alpha, m, n)

	# 						for i in range(3):
	# 							for j in range(3):
	# 								self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Mij_python_vs_c(self):

	# 	for ai in [0.1, 1.0, 10.0]:
	# 		for aj in [0.1, 1.0, 10.0]:
	# 			for rx in [0, -0.1, 0.1, 10, -10, -100, 100]:
	# 				for ry in [0, -0.1, 0.1, 10, -10, -100, 100]:
	# 					for rz in [0, -0.1, 0.1, 10, -10, -100, 100]:

	# 						if rx==0 and ry==0 and rz==0: continue

	# 						r = np.array([rx, ry, rz])

	# 						c_ish = Mij_rpy(ai, aj, r)
	# 						python_ish = Mij_rpy_python(ai, aj, r)

	# 						for i in range(3):
	# 							for j in range(3):
	# 								self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Mii_rpy_smith_python_vs_c(self):

	# 	for a in [0.1, 1.0, 10.0]:
	# 		for L in [10, 100, 1000]:
	# 				for m in [0, 1, 2, 3, 4, 5]:
	# 					for n in [0, 1, 2, 3, 4, 5]:

	# 						alpha = np.sqrt(np.pi)

	# 						c_ish = Mii_rpy_smith(a, L, alpha, m, n)
	# 						python_ish = Mii_rpy_smith_python(a, L, alpha, m, n)

	# 						for i in range(3):
	# 							for j in range(3):
	# 								self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Mij_rpy_smith_python_vs_c(self):

	# 	for ai in [0.1, 1.0, 10.0]:
	# 		for aj in [0.1, 1.0, 10.0]:
	# 			for L in [10, 100, 1000]:
	# 				for rx in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 					for ry in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 						for rz in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 							for m in [0, 1, 2, 3, 4, 5]:
	# 								for n in [0, 1, 2, 3, 4, 5]:

	# 									if rx==0 and ry==0 and rz==0: continue

	# 									r = np.array([rx, ry, rz])

	# 									alpha = np.sqrt(np.pi)

	# 									c_ish = Mij_rpy_smith(ai, aj, r, L, alpha, m, n)
	# 									python_ish = Mij_rpy_smith_python(ai, aj, r, L, alpha, m, n)

	# 									for i in range(3):
	# 										for j in range(3):
	# 											self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_Mij_rpy_smith_symmetry(self):

	# 	for ai in [0.1, 1.0, 10.0]:
	# 		for aj in [0.1, 1.0, 10.0]:
	# 			for L in [10, 100, 1000]:
	# 				for rx in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 					for ry in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 						for rz in [0.0, 0.1*L, -0.1*L, 0.9*L, -0.9*L]:
	# 							for m in [0, 1, 2, 3, 4, 5]:
	# 								for n in [0, 1, 2, 3, 4, 5]:

	# 									if rx==0 and ry==0 and rz==0: continue

	# 									alpha = np.sqrt(np.pi)

	# 									r = np.array([rx, ry, rz])

	# 									vij = Mij_rpy_smith(ai, aj, r, L, alpha, m, n)
	# 									vji = Mij_rpy_smith(aj, ai, r, L, alpha, m, n)

	# 									for i in range(3):
	# 										for j in range(3):
	# 											self.assertAlmostEqual( vij[i][j], vji[i][j], places = 7 )

	# 									vij = Mij_rpy_smith_python(ai, aj, r, L, alpha, m, n)
	# 									vji = Mij_rpy_smith_python(aj, ai, r, L, alpha, m, n)

	# 									for i in range(3):
	# 										for j in range(3):
	# 											self.assertAlmostEqual( vij[i][j], vji[i][j], places = 7 )

	# #---------------------------------------------------------------------------

	# def test_X_Y_polys(self):

	# 	for l in [0.1, 0.5, 1.0, 2.0, 10.0]:

	# 		for rank in range(12):

	# 			self.assertAlmostEqual( X_f_poly(l, rank), X_f_poly_python(l, rank), places = 7 )

	# 			self.assertAlmostEqual( Y_f_poly(l, rank), Y_f_poly_python(l, rank), places = 7 )

	# 			if rank in [1, 2, 3]:

	# 				self.assertAlmostEqual( X_g_poly(l, rank), X_g_poly_python(l, rank), places = 7 )

	# 				if rank != 1: self.assertAlmostEqual( Y_g_poly(l, rank), Y_g_poly_python(l, rank), places = 7 )

	# #---------------------------------------------------------------------------

	# def test_XA_YA(self):

	# 	for l in [0.1, 0.5, 1.0, 2.0, 10.0]:

	# 		for s in np.linspace(2.001, 10.0, 10):

	# 			self.assertAlmostEqual( XA11(s, l), XA11_python(s, l), places = 7 )

	# 			self.assertAlmostEqual( YA11(s, l), YA11_python(s, l), places = 7 )

	# 			self.assertAlmostEqual( XA12(s, l), XA12_python(s, l), places = 7 )

	# 			self.assertAlmostEqual( YA12(s, l), YA12_python(s, l), places = 7 )

	# #---------------------------------------------------------------------------

	# def test_R_jeffrey(self):

	# 	for ai in [0.1, 0.5, 1.0]:

	# 		for aj in [0.1, 0.5, 1.0]:

	# 			for rx in [-10.0, -2.1, 0.0, 2.1, 10.0]:

	# 				for ry in [-10.0, -2.1, 0.0, 2.1, 10.0]:

	# 					for rz in [-10.0, -2.1, 0.0, 2.1, 10.0]:

	# 						if rx == 0 and ry == 0 and rz == 0: continue

	# 						pointer = np.array([rx, ry, rz])

	# 						c_ish = R_jeffrey(ai, aj, pointer)

	# 						python_ish = R_jeffrey_python(ai, aj, pointer)

	# 						for i in range(6):

	# 							for j in range(6):

	# 								self.assertAlmostEqual( c_ish[i][j], python_ish[i][j], places = 7 )

	#---------------------------------------------------------------------------

	def test_M_rpy_smith(self):

		box_length = 20.0

		alpha = np.sqrt(np.pi)

		m = 3

		n = 3

		beads = [ Bead(np.random.normal(0.0, 5.0, 3), 1.0) for i in range(100) ]

		pointers = [ [ pointer_pbc(bi, bj, box_length) for bj in beads ] for bi in beads ]

		c_ish = M_rpy_smith(beads, pointers, box_length, alpha, m, n)

		python_ish = M_rpy_smith_python(beads, pointers, box_length, alpha, m, n)

		for i in range(6):
			for j in range(6):
				self.assertAlmostEqual(c_ish[i][j], python_ish[i][j], places = 7)

	#---------------------------------------------------------------------------