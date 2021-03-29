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
import math
import numpy as np
import os
import pickle
import shutil
import sys
sys.path.insert(0, '.')
import unittest

from pyBD.box import Box
from pyBD.bead import Bead

#-------------------------------------------------------------------------------

class TestBox(unittest.TestCase):

	def setUp(self):

		self.mock_input = {"hydrodynamics": "nohi", "box_length": 35.0, "T": 298.0,
					  	   "viscosity": 0.01, "external_force": [0, 0, 0],
					  	   "immobile_labels": []}

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

			b.sync_seed()

			for _ in range(M//2): b.propagate( 0.001 )

			restarted = [ bead.r[i] for bead in b.beads for i in range(3) ]

			self.assertSequenceEqual( original, restarted )

	#---------------------------------------------------------------------------

	def test_seed_sync_implicit(self):

		self.mock_input["seed"] = np.random.randint(2**32 - 1)

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

			b.sync_seed()

			for _ in range(M//2): b.propagate( 0.001 )

			restarted = [ bead.r[i] for bead in b.beads for i in range(3) ]

			self.assertSequenceEqual( original, restarted )

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

		self.assertSequenceEqual(should_be, list( b.F0 ))

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

	def tearDown(self):

		if os.path.exists('test_box.txt'):

			os.remove('test_box.txt')

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------