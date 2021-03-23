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

	def test_seed_sync(self):

		mock_input = {"seed": 1, "hydrodynamics": "nohi", "box_length": 35.0, "T": 298.0,
					  "viscosity": 0.01, "external_force": [0, 0, 0]}

		for M in range(4, 20):

			beads = [ Bead(np.array([i, j, k], float), 0.1) for i in range(5) for j in range(5) for k in range(5) ]

			b = Box(beads, mock_input)

			for _ in range(M): b.propagate( 0.001 )

			with open('test_box.txt', 'wb') as test_file:

				pickle.dump(b, test_file)

			for _ in range(M//2): b.propagate( 0.001 )

			original = [ bead.r[i] for bead in b.beads for i in range(3) ]

			del b

			with open('test_box.txt', 'rb') as test_file:

				b = pickle.load(test_file)

			b.sync_seed()

			for _ in range(M//2): b.propagate( 0.001 )

			restarted = [ bead.r[i] for bead in b.beads for i in range(3) ]

			self.assertSequenceEqual( original, restarted )

	#---------------------------------------------------------------------------

	def tearDown(self):

		os.remove('test_box.txt')

#-------------------------------------------------------------------------------

if __name__ == '__main__':

	unittest.main()

#-------------------------------------------------------------------------------