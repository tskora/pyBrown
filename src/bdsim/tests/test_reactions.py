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
import os
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '..') ))
import unittest

from pyBrown.bead import Bead, pointer
from pyBrown.reactions import Reactions

#-------------------------------------------------------------------------------

class TestReactions(unittest.TestCase):

	def setUp(self):

		self.beads = [ Bead([0.0, 0.0, 0.0], 1.0, label = "A"),
					   Bead([0.0, 0.0, 3.0], 1.0, label = "B"),
					   Bead([0.0, 0.0, -3.0], 1.0, label = "A"),
					   Bead([0.0, 0.0, -6.0], 1.0, label = "B"),
					   Bead([0.0, 0.0, -90.0], 1.0, label = "A"),
					   Bead([0.0, 0.0, -6.0], 1.0, label = "C") ]

		self.rij = np.zeros((len(self.beads), len(self.beads), 3))

		for i in range(1, len(self.beads)):
			for j in range(0, i):
				self.rij[i][j] = pointer(self.beads[i], self.beads[j])
				self.rij[j][i] = -self.rij[i][j]

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing(self):

		r = Reactions('example_reaction', 'A+B->C', 'dist A B <= 25.0', 'end_simulation')

		self.assertSequenceEqual(r.substrates, ['A', 'B'])

		self.assertSequenceEqual(r.products, ['C'])

		self.assertEqual(r.condition_types[0], "dist")

		self.assertSequenceEqual(r.conditions[0]['A B'], ('<=', 25.0))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('<=', 25.0))

	#---------------------------------------------------------------------------

	def test_ntuples(self):

		r = Reactions('example_reaction', 'A+B->C', 'dist A B <= 25.0', 'end_simulation')

		r.check_for_reactions(self.beads, self.rij)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------