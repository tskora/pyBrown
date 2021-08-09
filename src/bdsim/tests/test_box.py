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
import math
import numpy as np
import os
import pickle
import shutil
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from scipy.constants import Boltzmann

from pyBrown.bead import Bead, compute_pointer_pbc_matrix
from pyBrown.box import Box
from pyBrown.diffusion import RPY_M_tt_matrix, RPY_Smith_M_matrix, JO_R_lubrication_correction_F_matrix

#-------------------------------------------------------------------------------

class TestBox(unittest.TestCase):

	def setUp(self):

		self.mock_input = {"hydrodynamics": "nohi", "box_length": 35.0, "T": 298.0,
					  	   "viscosity": 0.01, "external_force": [0, 0, 0],
					  	   "immobile_labels": [], "propagation_scheme": "ermak",
					  	   "check_overlaps": True, "lennard_jones_6": False,
					  	   "lennard_jones_12": False, "energy_unit": "joule",
					  	   "custom_interactions": False, "debug": False,
					  	   "overlap_treshold": 0.0, "max_move_attempts": 1000000,
					  	   "cichocki_correction": True, "divergence_term": False}

		self.test_filename = 'test_box.txt'

	#---------------------------------------------------------------------------

	def test_seed_sync(self):

		self.mock_input["seed"] = 1

		beads = [ Bead(np.array([i, j, k], float), 0.1) for i in range(5) for j in range(5) for k in range(5) ]

		b = Box(beads, self.mock_input)

		for M in range(4, 20):

			for _ in range(M): b.propagate( 0.001 )

			with open(self.test_filename, 'wb') as test_file:

				pickle.dump(b, test_file)

			for _ in range(M//2): b.propagate( 0.001 )

			original = [ bead.r[i] for bead in b.beads for i in range(3) ]

			del b

			with open(self.test_filename, 'rb') as test_file:

				b = pickle.load(test_file)

			for _ in range(M//2): b.propagate( 0.001 )

			restarted = [ bead.r[i] for bead in b.beads for i in range(3) ]

			self.assertSequenceEqual( original, restarted )

	#---------------------------------------------------------------------------

	def test_seed_sync_implicit(self):

		self.mock_input["seed"] = None

		beads = [ Bead(np.array([i, j, k], float), 0.1) for i in range(5) for j in range(5) for k in range(5) ]

		b = Box(beads, self.mock_input)

		for M in range(4, 20):

			for _ in range(M): b.propagate( 0.001 )

			with open(self.test_filename, 'wb') as test_file:

				pickle.dump(b, test_file)

			for _ in range(M//2): b.propagate( 0.001 )

			original = [ bead.r[i] for bead in b.beads for i in range(3) ]

			del b

			with open(self.test_filename, 'rb') as test_file:

				b = pickle.load(test_file)

			for _ in range(M//2): b.propagate( 0.001 )

			restarted = [ bead.r[i] for bead in b.beads for i in range(3) ]

			self.assertSequenceEqual( original, restarted )

	#---------------------------------------------------------------------------

	def test_length_of_force_vector_with_immobile_particles(self):

		self.mock_input["seed"] = None

		beads = [ Bead(np.array([i, j, k], float), 0.1) for i in range(5) for j in range(5) for k in range(5) ]

		for i, bead in enumerate(beads):

			if i%5==0: bead.mobile = False

		b = Box(beads, self.mock_input)

		self.assertEqual( len(b.F), 3*(5*5*5-125/5) )

	#---------------------------------------------------------------------------

	def test_external_force_region(self):

		xFr = {"x": [2.5, 3.5], "y": [-1, 1], "z": [0.5, 2.5]}

		Fex = [7.0, 7.0, 7.0]

		self.mock_input["external_force_region"] = xFr

		self.mock_input["external_force"] = Fex

		self.mock_input["seed"] = 0

		beads = [ Bead(np.array([i, j, k], float), 0.1) for i in range(0,5)
														for j in range(0,5)
														for k in range(0,5) ]

		should_be = []

		for bead in beads:

			if bead.r[0] < xFr["x"][0] or bead.r[0] > xFr["x"][1]:
				should_be += [0.0, 0.0, 0.0]
				continue
			if bead.r[1] < xFr["y"][0] or bead.r[1] > xFr["y"][1]:
				should_be += [0.0, 0.0, 0.0]
				continue
			if bead.r[2] < xFr["z"][0] or bead.r[2] > xFr["z"][1]:
				should_be += [0.0, 0.0, 0.0]
				continue

			should_be += Fex

		b = Box(beads, self.mock_input)

		b.propagate(1e-20)

		self.assertSequenceEqual(should_be, list( b.F ))

	#---------------------------------------------------------------------------

	def test_external_force_region_scarce(self):

		xFr = {"x": [2.5, 3.5], "y": [-1, 1], "z": [0.5, 2.5]}

		Fex = [0.00007, 0.00008, 0.00009]

		dt = 1e-14

		self.mock_input["external_force_region"] = xFr

		self.mock_input["external_force"] = Fex

		self.mock_input["seed"] = 0

		self.mock_input["box_length"] = 75000000.0

		beads = [ Bead(np.array([3, 0, 2], float), 0.1), Bead(np.array([1, 0, 2], float), 0.1) ]

		initial = [ beads[i].r[j] for i in range(2) for j in range(3) ]

		b = Box(beads, self.mock_input)

		b.propagate(dt)

		np.random.seed(0)

		for _ in range(b.draw_count//6): test_displacement = np.random.normal(0.0, 1.0, 6)

		move_canceled_1 = beads[0].r - test_displacement[:3]*b.B[:3]*math.sqrt(2*dt) - dt / b.kBT * b.D[:3] * np.array(Fex)

		move_canceled_2 = beads[1].r - test_displacement[3:]*b.B[3:]*math.sqrt(2*dt)

		move_canceled = [*move_canceled_1, *move_canceled_2]

		for i in range(6): self.assertAlmostEqual(initial[i],move_canceled[i], places=15)

	#---------------------------------------------------------------------------

	def test_random_numbers(self):

		self.mock_input["seed"] = 0

		N = 100

		np.random.seed(0)

		beads = [ Bead(np.array([0.0, 0.0, 0.0], float), 1.0),
				  Bead(np.array([3.0, 0.0, 0.0], float), 1.0),
				  Bead(np.array([0.0, 3.0, 0.0], float), 1.0) ]

		ref = np.random.normal(0.0, 1.0, (N, 3*len(beads)))

		b = Box(beads, self.mock_input)

		for i in range(len(beads)):

			b._generate_random_vector()

			for j in range(3): self.assertEqual(b.N[j], ref[i][j])

			self.assertEqual(b.draw_count, (i+1)*3*len(beads))

	#---------------------------------------------------------------------------

	def test_ermak_nohi(self):

		self.mock_input["seed"] = 0

		self.mock_input["propagation_scheme"] = "ermak"

		self.mock_input["hydrodynamics"] = "nohi"

		Nit = 100

		coords = [-6.0, -3.0, 0.0, 3.0, 6.0]

		dt = 1e-10

		beads = [ Bead(np.array([x, y, z], float), 1.0) for x in coords for y in coords for z in coords ]

		beads_copy = cp.deepcopy( beads )

		np.random.seed( self.mock_input["seed"] )

		N = np.random.normal(0.0, 1.0, Nit*3*len(beads))

		b = Box(beads, self.mock_input)

		for iteration in range(Nit):

			pointers = compute_pointer_pbc_matrix(beads_copy, self.mock_input["box_length"])

			b.propagate(dt)

			D = Boltzmann * self.mock_input["T"] * 10**19 / ( 6.0 * np.pi * self.mock_input["viscosity"] ) * np.ones(3*len(beads))

			B = np.sqrt(D)

			Ni = N[3*iteration*len(beads):3*(iteration+1)*len(beads)]

			BX = B * Ni * math.sqrt(2*dt)

			for i in range(3*len(beads)): self.assertEqual(b.N[i], Ni[i])

			for i, bead in enumerate(beads_copy):

				bead.translate( BX[3*i:3*(i+1)] )

			for i in range(len(beads)):

				for j in range(3):

					self.assertEqual( beads[i].r[j], beads_copy[i].r[j] )

	#---------------------------------------------------------------------------

	def test_ermak_rpy(self):

		self.mock_input["seed"] = 0

		self.mock_input["propagation_scheme"] = "ermak"

		self.mock_input["hydrodynamics"] = "rpy"

		Nit = 100

		coords = [-3.0, 0.0, 3.0]

		dt = 1e-10

		beads = [ Bead(np.array([x, y, z], float), 1.0) for x in coords for y in coords for z in coords ]

		beads_copy = cp.deepcopy( beads )

		np.random.seed( self.mock_input["seed"] )

		N = np.random.normal(0.0, 1.0, Nit*3*len(beads))

		b = Box(beads, self.mock_input)

		for iteration in range(Nit):

			# pointers = [ [ pointer_pbc(beads_copy[i], beads_copy[j], self.mock_input["box_length"]) for i in range(len(beads)) ] for j in range(len(beads)) ]

			pointers = compute_pointer_pbc_matrix(beads_copy, self.mock_input["box_length"])

			b.propagate(dt)

			D = Boltzmann * self.mock_input["T"] * 10**19 / self.mock_input["viscosity"] * RPY_M_tt_matrix(beads_copy, pointers)

			B = np.linalg.cholesky( D )

			Ni = N[3*iteration*len(beads):3*(iteration+1)*len(beads)]

			BX = B @ Ni * math.sqrt(2*dt)

			for i in range(3*len(beads)): self.assertEqual(b.N[i], Ni[i])

			for i, bead in enumerate(beads_copy):

				bead.translate( BX[3*i:3*(i+1)] )

			for i in range(len(beads)):

				for j in range(3):

					self.assertEqual( beads[i].r[j], beads_copy[i].r[j] )

	#---------------------------------------------------------------------------

	def test_ermak_rpy_smith(self):

		self.mock_input["seed"] = 0

		self.mock_input["propagation_scheme"] = "ermak"

		self.mock_input["hydrodynamics"] = "rpy_smith"

		self.mock_input["ewald_alpha"] = np.sqrt( np.pi )

		Nit = 100

		coords = [-3.0, 0.0, 3.0]

		dt = 1e-10

		for n in range(3):

			self.mock_input["ewald_real"] = n

			self.mock_input["ewald_imag"] = n

			beads = [ Bead(np.array([x, y, z], float), 1.0) for x in coords for y in coords for z in coords ]

			beads_copy = cp.deepcopy( beads )

			np.random.seed( self.mock_input["seed"] )

			N = np.random.normal(0.0, 1.0, Nit*3*len(beads))

			b = Box(beads, self.mock_input)

			for iteration in range(Nit):

				# pointers = [ [ pointer_pbc(beads_copy[i], beads_copy[j], self.mock_input["box_length"]) for i in range(len(beads)) ] for j in range(len(beads)) ]

				pointers = compute_pointer_pbc_matrix(beads_copy, self.mock_input["box_length"])

				b.propagate(dt)

				D = Boltzmann * self.mock_input["T"] * 10**19 / self.mock_input["viscosity"] * RPY_Smith_M_matrix(beads_copy, pointers, self.mock_input["box_length"], self.mock_input["ewald_alpha"], n, n)

				B = np.linalg.cholesky( D )

				Ni = N[3*iteration*len(beads):3*(iteration+1)*len(beads)]

				BX = B @ Ni * math.sqrt(2*dt)

				for i in range(3*len(beads)): self.assertEqual(b.N[i], Ni[i])

				for i, bead in enumerate(beads_copy):

					bead.translate( BX[3*i:3*(i+1)] )

				for i in range(len(beads)):

					for j in range(3):

						self.assertAlmostEqual( beads[i].r[j], beads_copy[i].r[j] )

	#---------------------------------------------------------------------------

	def test_midpoint_rpy_smith_lub(self):

		lubrication_cutoff = 10

		self.mock_input["seed"] = 0

		self.mock_input["propagation_scheme"] = "midpoint"

		self.mock_input["hydrodynamics"] = "rpy_smith_lub"

		self.mock_input["lubrication_cutoff"] = lubrication_cutoff

		self.mock_input["ewald_alpha"] = np.sqrt( np.pi )

		Nit = 100

		coords = [-2.2, 0.0, 2.2]

		ms = [ 2.0, 10.0, 100.0 ]

		dt = 1e-10

		for n in range(3):

			for m in ms:

				self.mock_input["ewald_real"] = n

				self.mock_input["ewald_imag"] = n

				self.mock_input["m_midpoint"] = m

				beads = [ Bead(np.array([x, y, z], float), 1.0) for x in coords for y in coords for z in coords ]

				beads_copy = cp.deepcopy( beads )

				np.random.seed( self.mock_input["seed"] )

				N = np.random.normal(0.0, 1.0, Nit*3*len(beads))

				b = Box(beads, self.mock_input)

				for iteration in range(Nit):

					# pointers = [ [ pointer_pbc(beads_copy[i], beads_copy[j], self.mock_input["box_length"]) for i in range(len(beads)) ] for j in range(len(beads)) ]

					pointers = compute_pointer_pbc_matrix(beads_copy, self.mock_input["box_length"])

					b.propagate(dt)

					Mff = RPY_Smith_M_matrix(beads_copy, pointers, self.mock_input["box_length"], self.mock_input["ewald_alpha"], n, n) * 10**19 / self.mock_input["viscosity"]

					Rff = np.linalg.inv(Mff)

					Rlc = JO_R_lubrication_correction_F_matrix(beads, pointers, lubrication_cutoff = lubrication_cutoff, cichocki_correction = self.mock_input["cichocki_correction"]) * self.mock_input["viscosity"] * 10**(-19)

					Rtot = Rlc + Rff

					Dtot = Boltzmann * self.mock_input["T"] * np.linalg.inv(Rtot)

					Btot = np.linalg.cholesky(Dtot)

					Ni = N[3*iteration*len(beads):3*(iteration+1)*len(beads)]

					BX = Btot @ Ni * math.sqrt(2*dt)

					for i in range(3*len(beads)): self.assertEqual(b.N[i], Ni[i])

					for i, bead in enumerate(beads_copy):

						bead.translate( BX[3*i:3*(i+1)] / m )

					pointers = compute_pointer_pbc_matrix(beads_copy, self.mock_input["box_length"])

					Mff_mid = RPY_Smith_M_matrix(beads_copy, pointers, self.mock_input["box_length"], self.mock_input["ewald_alpha"], n, n) * 10**19 / self.mock_input["viscosity"]

					Rff_mid = np.linalg.inv(Mff_mid)

					Rlc_mid = JO_R_lubrication_correction_F_matrix(beads, pointers, lubrication_cutoff = lubrication_cutoff, cichocki_correction = self.mock_input["cichocki_correction"]) * self.mock_input["viscosity"] * 10**(-19)

					Rtot_mid = Rlc_mid + Rff_mid

					Dtot_mid = Boltzmann * self.mock_input["T"] * np.linalg.inv(Rtot_mid)

					for i, bead in enumerate(beads_copy):

						bead.translate( BX[3*i:3*(i+1)] * (1.0 - 1.0/m) )

					Btot_mid = np.linalg.cholesky(Dtot_mid)

					BX_mid = Dtot_mid @ np.linalg.inv( np.transpose(Btot) ) @ Ni * math.sqrt(2*dt)

					BX_drift = ( BX_mid - BX ) * m / 2

					for i, bead in enumerate(beads_copy):

						bead.translate( BX_drift[3*i:3*(i+1)] )

					for i in range(len(beads)):

						for j in range(3):

							self.assertAlmostEqual( beads[i].r[j], beads_copy[i].r[j], places = 9 )

	#---------------------------------------------------------------------------

	def tearDown(self):

		if os.path.exists('test_box.txt'):

			os.remove('test_box.txt')

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------