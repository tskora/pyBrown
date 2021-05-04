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

from itertools import product

#-------------------------------------------------------------------------------

class Reactions():

	# TODO: results of reactions(products), handling of change in bead numbers, possibility
	# of more than one reaction opportunity at single configuration

	def __init__(self, reaction_string, condition_string):

		self._parse_reaction_string(reaction_string)

		self._parse_condition_string(condition_string)

		self.reaction_count = 0

	#-------------------------------------------------------------------------------

	def check_for_reactions(self, beads, pointers):

		bead_indices = [ i for i in range(len(beads)) ]

		unique_ntuples = [ ntuple for ntuple in self._create_unique_ntuples(bead_indices)
						   if self._is_ntuple_reactive(ntuple, beads) ]


		for ntuple in unique_ntuples:

			if self._reaction_criterion(ntuple, beads, pointers): self._reaction_effect(ntuple, beads, pointers)

	#-------------------------------------------------------------------------------

	def _parse_reaction_string(self, reaction_string):

		self.reaction_string = reaction_string

		assert '->' in reaction_string, 'Invalid reaction string: no "->" symbol in reaction string'

		if '<->' in reaction_string:

			lhs, rhs = reaction_string.split('<->')

		elif '->' in reaction_string:

			lhs, rhs = reaction_string.split('->')

		self.substrates = lhs.split('+')

		self.products = rhs.split('+')

		self.number_of_substrates = len(self.substrates)

		self.substrates_count = { name:0 for name in self.substrates }

		for name in self.substrates:

			self.substrates_count[name] += 1

	#-------------------------------------------------------------------------------

	def _parse_condition_string(self, condition_string):

		self.condition_dictionary = {}

		for single_condition in condition_string.split(','):

			label1, label2, string_dist = single_condition.split(' ')

			dist = float(string_dist)

			self.condition_dictionary[label1+" "+label2] = dist

	#-------------------------------------------------------------------------------

	def _reaction_criterion(self, ntuple, beads, pointers):

		for i in range(1, len(ntuple)):

			index_i = ntuple[i]

			bead_i = beads[ntuple[i]]

			for j in range(i):

				index_j = ntuple[j]

				bead_j = beads[ntuple[j]]

				pointer = pointers[index_i][index_j]

				dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

				dist = math.sqrt(dist2)

				if bead_i.label+" "+bead_j.label in self.condition_dictionary.keys():

					if dist > self.condition_dictionary[bead_i.label+" "+bead_j.label]:

						return False

				elif bead_j.label+" "+bead_i.label in self.condition_dictionary.keys():

					if dist > self.condition_dictionary[bead_j.label+" "+bead_i.label]:

						return False

				else:

					print('ERROR IN REACTIONS -- unrecognized condition label')

					1/0

		return True

	#-------------------------------------------------------------------------------

	def _reaction_effect(self, ntuple, beads, pointers):

		print("reaction")

		self.reaction_count += 1

	#-------------------------------------------------------------------------------

	def _create_unique_ntuples(self, indices):

		ntuples = list( product(indices, repeat = self.number_of_substrates) )

		unique_ntuples = set( [ frozenset(ntuple) for ntuple in ntuples if len(frozenset(ntuple)) == self.number_of_substrates ] )

		return list( list(ntuple) for ntuple in unique_ntuples )

	#-------------------------------------------------------------------------------

	def _is_ntuple_reactive(self, ntuple, beads):

		from_ntuple = { name:0 for name in self.substrates }

		for index in ntuple:

			label = beads[index].label

			if label in from_ntuple.keys(): from_ntuple[ label ] += 1

		return self.substrates_count == from_ntuple

	#-------------------------------------------------------------------------------

	def __str__(self):

		string_template = 'Reaction: {}'

		return string_template.format(self.reaction_string)

	#-------------------------------------------------------------------------------

	def __repr__(self):

		return self.__str__()

#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------