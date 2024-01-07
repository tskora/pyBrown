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

from pyBrown.bead import Bead, pointer, compute_pointer_pbc_matrix, compute_pointer_immobile_pbc_matrix
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

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.reaction_strings = [
			['example_reaction', 'A+B->C', 'dist A B <= 25.0', 'end_simulation'],
			['example_reaction', 'A->B', 'random 0.4', 'end_simulation'],
			['example_reaction', 'A+B+C->?', 'dist A B > 0.4, dist B C <= 0.3', 'end_simulation'],
			['example_reaction', 'A+B+C->D+E', 'dist A B > 0.4, dist B C <= 0.3, pole_dist B A C < 1.28', 'end_simulation'],
			['example_reaction', 'A+A->?', 'dist A A <= 0.1, random 0.5', 'end_simulation'],
			['example_reaction', 'A+B->C', 'dist A B <= 2.5', 'end_simulation'],
			['example_reaction', 'A+B->C', 'dist A B <= 2.5, time > 20.0', 'end_simulation']
		]

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing(self):

		r = Reactions(*self.reaction_strings[0])

		self.assertSequenceEqual(r.substrates, ['A', 'B'])

		self.assertSequenceEqual(r.products, ['C'])

		self.assertEqual(r.condition_types[0], "dist")

		self.assertSequenceEqual(r.conditions[0]['A B'], ('<=', 25.0))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('<=', 25.0))

		self.assertEqual(r.number_of_substrates, 2)

		self.assertEqual(r.substrates_count['A'], 1)

		self.assertEqual(r.substrates_count['B'], 1)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing2(self):

		r = Reactions(*self.reaction_strings[1])

		self.assertSequenceEqual(r.substrates, ['A'])

		self.assertSequenceEqual(r.products, ['B'])

		self.assertEqual(r.condition_types[0], "random")

		self.assertEqual(r.conditions[0], 0.4)

		self.assertEqual(r.number_of_substrates, 1)

		self.assertEqual(r.substrates_count['A'], 1)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing3(self):

		r = Reactions(*self.reaction_strings[2])

		self.assertSequenceEqual(r.substrates, ['A', 'B', 'C'])

		self.assertSequenceEqual(r.products, ['?'])

		self.assertSequenceEqual(r.condition_types, ["dist", "dist"])

		self.assertSequenceEqual(r.conditions[0]['A B'], ('>', 0.4))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('>', 0.4))

		self.assertSequenceEqual(r.conditions[1]['B C'], ('<=', 0.3))

		self.assertSequenceEqual(r.conditions[1]['C B'], ('<=', 0.3))

		self.assertEqual(r.number_of_substrates, 3)

		self.assertEqual(r.substrates_count['A'], 1)

		self.assertEqual(r.substrates_count['B'], 1)

		self.assertEqual(r.substrates_count['C'], 1)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing4(self):

		r = Reactions(*self.reaction_strings[3])

		self.assertSequenceEqual(r.substrates, ['A', 'B', 'C'])

		self.assertSequenceEqual(r.products, ['D', 'E'])

		self.assertSequenceEqual(r.condition_types, ["dist", "dist", "pole_dist"])

		self.assertSequenceEqual(r.conditions[0]['A B'], ('>', 0.4))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('>', 0.4))

		self.assertSequenceEqual(r.conditions[1]['B C'], ('<=', 0.3))

		self.assertSequenceEqual(r.conditions[1]['C B'], ('<=', 0.3))

		self.assertSequenceEqual(r.conditions[2]['B A C'], ('<', 1.28))

		self.assertEqual(r.number_of_substrates, 3)

		self.assertEqual(r.substrates_count['A'], 1)

		self.assertEqual(r.substrates_count['B'], 1)

		self.assertEqual(r.substrates_count['C'], 1)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing5(self):

		r = Reactions(*self.reaction_strings[4])

		self.assertSequenceEqual(r.substrates, ['A', 'A'])

		self.assertSequenceEqual(r.products, ['?'])

		self.assertSequenceEqual(r.condition_types, ["dist", "random"])

		self.assertSequenceEqual(r.conditions[0]['A A'], ('<=', 0.1))

		self.assertEqual(r.conditions[1], 0.5)

		self.assertEqual(r.number_of_substrates, 2)

		self.assertEqual(r.substrates_count['A'], 2)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing6(self):

		r = Reactions(*self.reaction_strings[5])

		self.assertSequenceEqual(r.substrates, ['A', 'B'])

		self.assertSequenceEqual(r.products, ['C'])

		self.assertSequenceEqual(r.condition_types, ["dist"])

		self.assertSequenceEqual(r.conditions[0]['A B'], ('<=', 2.5))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('<=', 2.5))

		self.assertEqual(r.number_of_substrates, 2)

		self.assertEqual(r.substrates_count['A'], 1)

		self.assertEqual(r.substrates_count['B'], 1)

	#---------------------------------------------------------------------------

	def test_reaction_string_parsing7(self):

		r = Reactions(*self.reaction_strings[6])

		self.assertSequenceEqual(r.substrates, ['A', 'B'])

		self.assertSequenceEqual(r.products, ['C'])

		self.assertSequenceEqual(r.condition_types, ["dist", "time"])

		self.assertSequenceEqual(r.conditions[0]['A B'], ('<=', 2.5))

		self.assertSequenceEqual(r.conditions[0]['B A'], ('<=', 2.5))

		self.assertSequenceEqual(r.conditions[1], ('>', 20.0))

		self.assertEqual(r.number_of_substrates, 2)

		self.assertEqual(r.substrates_count['A'], 1)

		self.assertEqual(r.substrates_count['B'], 1)

	#---------------------------------------------------------------------------

	def test_ntuples(self):

		r = Reactions(*self.reaction_strings[0])

		unt = r._unique_ntuples(list(range(len(self.beads))))

		self.assertEqual(len(unt), 15)

		for i in range(1, 6):

			for j in range(i):

				self.assertTrue( [i, j] in unt or [j, i] in unt )

		self.assertFalse( r._ntuple_is_reactive( [0, 0], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [0, 1], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [0, 2], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [0, 3], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [0, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 5], self.beads ) ) #AC

		self.assertTrue( r._ntuple_is_reactive( [1, 0], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 1], self.beads ) ) #BB
		self.assertTrue( r._ntuple_is_reactive( [1, 2], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 3], self.beads ) ) #BB
		self.assertTrue( r._ntuple_is_reactive( [1, 4], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 5], self.beads ) ) #BC

		self.assertFalse( r._ntuple_is_reactive( [2, 0], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [2, 1], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [2, 2], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [2, 3], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [2, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [2, 5], self.beads ) ) #AC

		self.assertTrue( r._ntuple_is_reactive( [3, 0], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 1], self.beads ) ) #BB
		self.assertTrue( r._ntuple_is_reactive( [3, 2], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 3], self.beads ) ) #BB
		self.assertTrue( r._ntuple_is_reactive( [3, 4], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 5], self.beads ) ) #BC

		self.assertFalse( r._ntuple_is_reactive( [4, 0], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [4, 1], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [4, 2], self.beads ) ) #AA
		self.assertTrue( r._ntuple_is_reactive( [4, 3], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [4, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [4, 5], self.beads ) ) #AC

		self.assertFalse( r._ntuple_is_reactive( [5, 0], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 1], self.beads ) ) #CB
		self.assertFalse( r._ntuple_is_reactive( [5, 2], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 3], self.beads ) ) #CB
		self.assertFalse( r._ntuple_is_reactive( [5, 4], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 5], self.beads ) ) #CC

	#---------------------------------------------------------------------------

	def test_ntuples2(self):

		r = Reactions(*self.reaction_strings[1])

		unt = r._unique_ntuples(list(range(len(self.beads))))

		self.assertEqual(len(unt), 6)

		for i in range(6):

			self.assertTrue( [i] in unt )

		self.assertFalse( r._ntuple_is_reactive( [0, 0], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 1], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [0, 2], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 3], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [0, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 5], self.beads ) ) #AC

		self.assertTrue( r._ntuple_is_reactive( [0], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [1], self.beads ) ) #B
		self.assertTrue( r._ntuple_is_reactive( [2], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [3], self.beads ) ) #B
		self.assertTrue( r._ntuple_is_reactive( [4], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [5], self.beads ) ) #C

		self.assertFalse( r._ntuple_is_reactive( [], self.beads ) )

	#---------------------------------------------------------------------------

	def test_ntuples3(self):

		r = Reactions(*self.reaction_strings[2])

		unt = r._unique_ntuples(list(range(len(self.beads))))

		self.assertEqual(len(unt), 20)

		for i in range(len(self.beads)):

			for j in range(len(self.beads)):

				for k in range(len(self.beads)):

					if i == j or i == k or j == k: continue

					self.assertTrue( [i, j, k] in unt or
									 [j, k, i] in unt or
									 [k, i, j] in unt or
									 [k, j, i] in unt or
									 [j, i, k] in unt or
									 [i, k, j] in unt )

		self.assertFalse( r._ntuple_is_reactive( [0, 1, 5, 2], self.beads ) ) #AAA

		self.assertFalse( r._ntuple_is_reactive( [0, 0, 0], self.beads ) ) #AAA
		self.assertTrue( r._ntuple_is_reactive( [0, 1, 5], self.beads ) ) #ABC
		self.assertTrue( r._ntuple_is_reactive( [0, 5, 1], self.beads ) ) #ACB
		self.assertTrue( r._ntuple_is_reactive( [0, 5, 3], self.beads ) ) #ACB
		self.assertTrue( r._ntuple_is_reactive( [2, 5, 3], self.beads ) ) #ACB
		self.assertFalse( r._ntuple_is_reactive( [2, 4, 3], self.beads ) ) #AAB

		self.assertFalse( r._ntuple_is_reactive( [0], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [1], self.beads ) ) #B
		self.assertFalse( r._ntuple_is_reactive( [2], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [3], self.beads ) ) #B
		self.assertFalse( r._ntuple_is_reactive( [4], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [5], self.beads ) ) #C

		self.assertFalse( r._ntuple_is_reactive( [], self.beads ) )

	#---------------------------------------------------------------------------

	def test_ntuples4(self):

		r = Reactions(*self.reaction_strings[3])

		unt = r._unique_ntuples(list(range(len(self.beads))))

		self.assertEqual(len(unt), 20)

		for i in range(len(self.beads)):

			for j in range(len(self.beads)):

				for k in range(len(self.beads)):

					if i == j or i == k or j == k: continue

					self.assertTrue( [i, j, k] in unt or
									 [j, k, i] in unt or
									 [k, i, j] in unt or
									 [k, j, i] in unt or
									 [j, i, k] in unt or
									 [i, k, j] in unt )

		self.assertFalse( r._ntuple_is_reactive( [0, 1, 5, 2], self.beads ) ) #AAA

		self.assertFalse( r._ntuple_is_reactive( [0, 0, 0], self.beads ) ) #AAA
		self.assertTrue( r._ntuple_is_reactive( [0, 1, 5], self.beads ) ) #ABC
		self.assertTrue( r._ntuple_is_reactive( [0, 5, 1], self.beads ) ) #ACB
		self.assertTrue( r._ntuple_is_reactive( [0, 5, 3], self.beads ) ) #ACB
		self.assertTrue( r._ntuple_is_reactive( [2, 5, 3], self.beads ) ) #ACB
		self.assertFalse( r._ntuple_is_reactive( [2, 4, 3], self.beads ) ) #AAB

		self.assertFalse( r._ntuple_is_reactive( [0], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [1], self.beads ) ) #B
		self.assertFalse( r._ntuple_is_reactive( [2], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [3], self.beads ) ) #B
		self.assertFalse( r._ntuple_is_reactive( [4], self.beads ) ) #A
		self.assertFalse( r._ntuple_is_reactive( [5], self.beads ) ) #C

		self.assertFalse( r._ntuple_is_reactive( [], self.beads ) )

	#---------------------------------------------------------------------------

	def test_ntuples5(self):

		r = Reactions(*self.reaction_strings[4])

		unt = r._unique_ntuples(list(range(len(self.beads))))

		self.assertEqual(len(unt), 15)

		for i in range(1, len(self.beads)):

			for j in range(i):

				self.assertTrue( [i, j] in unt or [j, i] in unt )

		self.assertFalse( r._ntuple_is_reactive( [0, 0], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 1], self.beads ) ) #AB
		self.assertTrue( r._ntuple_is_reactive( [0, 2], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 3], self.beads ) ) #AB
		self.assertTrue( r._ntuple_is_reactive( [0, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [0, 5], self.beads ) ) #AC

		self.assertFalse( r._ntuple_is_reactive( [1, 0], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 1], self.beads ) ) #BB
		self.assertFalse( r._ntuple_is_reactive( [1, 2], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 3], self.beads ) ) #BB
		self.assertFalse( r._ntuple_is_reactive( [1, 4], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [1, 5], self.beads ) ) #BC

		self.assertTrue( r._ntuple_is_reactive( [2, 0], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [2, 1], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [2, 2], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [2, 3], self.beads ) ) #AB
		self.assertTrue( r._ntuple_is_reactive( [2, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [2, 5], self.beads ) ) #AC

		self.assertFalse( r._ntuple_is_reactive( [3, 0], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 1], self.beads ) ) #BB
		self.assertFalse( r._ntuple_is_reactive( [3, 2], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 3], self.beads ) ) #BB
		self.assertFalse( r._ntuple_is_reactive( [3, 4], self.beads ) ) #BA
		self.assertFalse( r._ntuple_is_reactive( [3, 5], self.beads ) ) #BC

		self.assertTrue( r._ntuple_is_reactive( [4, 0], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [4, 1], self.beads ) ) #AB
		self.assertTrue( r._ntuple_is_reactive( [4, 2], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [4, 3], self.beads ) ) #AB
		self.assertFalse( r._ntuple_is_reactive( [4, 4], self.beads ) ) #AA
		self.assertFalse( r._ntuple_is_reactive( [4, 5], self.beads ) ) #AC

		self.assertFalse( r._ntuple_is_reactive( [5, 0], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 1], self.beads ) ) #CB
		self.assertFalse( r._ntuple_is_reactive( [5, 2], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 3], self.beads ) ) #CB
		self.assertFalse( r._ntuple_is_reactive( [5, 4], self.beads ) ) #CA
		self.assertFalse( r._ntuple_is_reactive( [5, 5], self.beads ) ) #CC

	#---------------------------------------------------------------------------

	def test_reaction_criterion(self):

		r = Reactions(*self.reaction_strings[0])

		self.assertTrue( r._reaction_criterion([0, 1], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([1, 0], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([0, 3], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([3, 0], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([2, 1], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([1, 2], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([2, 3], self.beads, [], self.rij, []) )
		self.assertTrue( r._reaction_criterion([3, 2], self.beads, [], self.rij, []) )
		self.assertFalse( r._reaction_criterion([4, 1], self.beads, [], self.rij, []) )
		self.assertFalse( r._reaction_criterion([1, 4], self.beads, [], self.rij, []) )
		self.assertFalse( r._reaction_criterion([4, 3], self.beads, [], self.rij, []) )
		self.assertFalse( r._reaction_criterion([3, 4], self.beads, [], self.rij, []) )

	#---------------------------------------------------------------------------

	def test_reaction_criterion2(self):

		r = Reactions(*self.reaction_strings[1])

		seed = r.seed

		np.random.seed( seed )

		self.assertEqual( r._reaction_criterion([0, 1], self.beads, [], self.rij, []), np.random.uniform(0.0, 1.0) < 0.4 )

	#---------------------------------------------------------------------------

	def test_reaction_criterion3(self):

		r = Reactions(*self.reaction_strings[2])

		self.assertFalse( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

		self.beads[5].translate([0, 0, 8.8])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.assertTrue( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

		self.beads[0].translate([0, 0, 2.7])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.assertFalse( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

	#---------------------------------------------------------------------------

	def test_reaction_criterion4(self):

		r = Reactions(*self.reaction_strings[3])

		self.assertFalse( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

		self.beads[5].translate([0, 0, 8.8])

		self.beads[0].translate([0, 0, 2.5])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.assertFalse( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

		self.beads[5].translate([0, 0, -0.05])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.assertTrue( r._reaction_criterion([0, 1, 5], self.beads, [], self.rij, []) )

	#---------------------------------------------------------------------------

	def test_reaction_criterion5(self):

		r = Reactions(*self.reaction_strings[4])

		seed = r.seed

		np.random.seed( seed )

		self.assertFalse( r._reaction_criterion([0, 2], self.beads, [], self.rij, []) )

		self.beads[2].translate([0.05, -0.05, 2.95])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		for i in range(300):

			self.assertEqual( r._reaction_criterion([0, 2], self.beads, [], self.rij, []), np.random.uniform(0.0, 1.0) < 0.5 )

	#---------------------------------------------------------------------------

	def test_reaction_criterion7(self):

		r = Reactions(*self.reaction_strings[6])

		r.time = 0.0

		self.assertFalse( r._reaction_criterion([0, 1], self.beads, [], self.rij, []) )

		self.beads[1].translate([0.0, 0.0, -1])

		self.rij = compute_pointer_pbc_matrix(self.beads, 10000.0)

		self.assertFalse( r._reaction_criterion([0, 1], self.beads, [], self.rij, []) )

		r.time = 30.0

		self.assertTrue( r._reaction_criterion([0, 1], self.beads, [], self.rij, []) )

	#---------------------------------------------------------------------------

	def test_check_for_reactions(self):

		r = Reactions(*self.reaction_strings[0])

		r.check_for_reactions(self.beads, self.rij)

		self.assertTrue( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions2(self):

		r = Reactions(*self.reaction_strings[5])

		r.check_for_reactions(self.beads, self.rij)

		self.assertFalse( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions3(self):

		beads = [ Bead([0.0, 0.0, 0.0], 1.0, label = 'A'),
				  Bead([0.0, 0.0, 2.0], 1.0, label = 'B'),
				  Bead([0.0, 0.0, 4.0], 1.0, label = 'C') ]

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r = Reactions('example_reaction', 'A+B+C->?', 'pole_dist C B A < 0.9', 'end_simulation')

		r.check_for_reactions(beads, rij)

		self.assertFalse( r.end_simulation )

		beads[1].translate([0.0, 0.0, -0.2])

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r.check_for_reactions(beads, rij)

		self.assertTrue( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions4(self):

		beads = [ Bead([0.0, 0.0, 0.0], 1.0, label = 'A'),
				  Bead([0.0, 0.0, 2.0], 1.0, label = 'B'),
				  Bead([0.0, 0.0, 0.0], 1.0, label = 'C') ]

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r = Reactions('example_reaction', 'A+B+C->?', 'pole_dist C B A < 2.9', 'end_simulation')

		r.check_for_reactions(beads, rij)

		self.assertFalse( r.end_simulation )

		beads[1].translate([0.0, 0.0, -0.2])

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r.check_for_reactions(beads, rij)

		self.assertTrue( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions5(self):

		beads = [ Bead([0.0, 0.0, 0.0], 1.0, label = 'A'),
				  Bead([0.0, 0.0, 2.0], 1.0, label = 'B'),
				  Bead([0.0, 2.0, 2.0], 1.0, label = 'C') ]

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r = Reactions('example_reaction', 'A+B+C->?', 'pole_dist C B A < 2.2', 'end_simulation')

		r.check_for_reactions(beads, rij)

		self.assertFalse( r.end_simulation )

		beads[1].translate([0.0, 0.0, -0.2])

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r.check_for_reactions(beads, rij)

		self.assertTrue( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions7(self):

		beads = [ Bead([0.0, 0.0, 0.0], 1.0, label = 'A'),
				  Bead([0.0, 0.0, 2.0], 1.0, label = 'B') ]

		rij = compute_pointer_pbc_matrix(beads, 10000.0)

		r = Reactions('example_reaction', 'A+B->?', 'dist A B < 100.0, time > 100.0', 'end_simulation')

		for i in range(200):

			r.check_for_reactions(beads, rij, time = i)

			self.assertEqual(i, r.time)

			if i <= 100.0: self.assertFalse( r.end_simulation )
			else: self.assertTrue( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions_igg(self):

		mobile_beads = [ Bead([0.0, 0.0, 160.0], 45.0, label = 'A'),
				  		 Bead([0.0, 0.0, 100.0], 10.0, label = 'B'),
				  		 Bead([0.0, -31.947, 77.631], 24.2, label = 'C'),
				  		 Bead([0.0, -56.521, 60.423], 24.2, label = 'D'),
				  		 Bead([0.0, 31.947, 77.631], 24.2, label = 'E'),
				  		 Bead([0.0, 56.521, 60.423], 24.2, label = 'F') ]
		
		immobile_beads = [ Bead([0.0, 0.0, 0.0], 24.2, label = 'AG', mobile = False) ]

		rij = compute_pointer_pbc_matrix(mobile_beads, 750.0)

		rik = compute_pointer_immobile_pbc_matrix(mobile_beads, immobile_beads, 750.0)

		r = Reactions('no_react', 'B+AG->?', 'dist B AG > 202.0', 'end_simulation')

		r.check_for_reactions(mobile_beads, rij, immobile_beads = immobile_beads, pointers_mobile_immobile = rik)

		self.assertFalse( r.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions_igg2(self):

		mobile_beads = [ Bead([0.0, 0.0, 160.0], 45.0, label = 'A'),
				  		 Bead([0.0, 0.0, 100.0], 10.0, label = 'B'),
				  		 Bead([0.0, -31.947, 77.631], 24.2, label = 'C'),
				  		 Bead([0.0, -56.521, 60.423], 24.2, label = 'D'),
				  		 Bead([0.0, 0.0, 75.0], 24.2, label = 'E'),
				  		 Bead([0.0, 0.0, 49.0], 24.2, label = 'F') ]

		immobile_beads = [ Bead([0.0, 0.0, 0.0], 24.2, label = 'AG', mobile = False) ]

		rij = compute_pointer_pbc_matrix(mobile_beads, 750.0)

		rik = compute_pointer_immobile_pbc_matrix(mobile_beads, immobile_beads, 750.0)

		r = Reactions('react', 'E+F+AG->?', 'pole_dist E F AG < 25.2', 'end_simulation')

		rbis = Reactions('react', 'F+AG->?', 'dist F AG < 49.4', 'end_simulation')

		r.check_for_reactions(mobile_beads, rij, immobile_beads = immobile_beads, pointers_mobile_immobile = rik)

		rbis.check_for_reactions(mobile_beads, rij, immobile_beads = immobile_beads, pointers_mobile_immobile = rik)

		self.assertTrue( r.end_simulation )

		self.assertTrue( rbis.end_simulation )

	#---------------------------------------------------------------------------

	def test_check_for_reactions_igg3(self):

		mobile_beads = [ Bead([0.0, 0.0, 160.0], 45.0, label = 'A'),
				  		 Bead([0.0, 0.0, 100.0], 10.0, label = 'B'),
				  		 Bead([0.0, -31.947, 77.631], 24.2, label = 'C'),
				  		 Bead([0.0, -56.521, 60.423], 24.2, label = 'D'),
				  		 Bead([0.0, 75.0, 75.0], 24.2, label = 'E'),
				  		 Bead([0.0, 0.0, 49.0], 24.2, label = 'F') ]

		immobile_beads = [ Bead([0.0, 0.0, 0.0], 24.2, label = 'AG', mobile = False) ]

		rij = compute_pointer_pbc_matrix(mobile_beads, 750.0)

		rik = compute_pointer_immobile_pbc_matrix(mobile_beads, immobile_beads, 750.0)

		r = Reactions('react', 'E+F+AG->?', 'pole_dist E F AG < 25.2', 'end_simulation')

		rbis = Reactions('react', 'F+AG->?', 'dist F AG < 49.4', 'end_simulation')

		r.check_for_reactions(mobile_beads, rij, immobile_beads = immobile_beads, pointers_mobile_immobile = rik)

		rbis.check_for_reactions(mobile_beads, rij, immobile_beads = immobile_beads, pointers_mobile_immobile = rik)

		self.assertFalse( r.end_simulation )

		self.assertTrue( rbis.end_simulation )

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	unittest.main()

#-------------------------------------------------------------------------------