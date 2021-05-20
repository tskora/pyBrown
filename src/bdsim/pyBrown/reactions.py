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

	def __init__(self, reaction_name, reaction_string, condition_string, effect_string):

		self.reaction_name = reaction_name.strip()

		self._parse_reaction_string(reaction_string.strip())

		self._parse_condition_string(condition_string.strip())

		self._parse_effect_string(effect_string.strip())

		self.reaction_count = 0

		self.end_simulation = False

	#-------------------------------------------------------------------------------

	def check_for_reactions(self, beads, pointers):

		bead_indices = [ i for i in range(len(beads)) ]

		unique_ntuples = [ ntuple for ntuple in self._unique_ntuples(bead_indices)
						   if self._ntuple_is_reactive(ntuple, beads) ]

		for ntuple in unique_ntuples:

			if self._reaction_criterion(ntuple, beads, pointers): self._reaction_effect(ntuple, beads, pointers)

	#-------------------------------------------------------------------------------

	def _parse_reaction_string(self, reaction_string):

		self.reaction_string = reaction_string

		assert '->' in reaction_string, 'Invalid reaction string: no "->" symbol in reaction string'

		if '<->' in reaction_string:

			# not used yet

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

			label1, label2, sign, string_dist = single_condition.strip().split(' ')

			dist = float(string_dist)

			self.condition_dictionary[label1+" "+label2] = (sign, dist)

			self.condition_dictionary[label2+" "+label1] = (sign, dist)

	#-------------------------------------------------------------------------------

	def _parse_effect_string(self, effect_string):

		self.effect_string = effect_string

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

				# print('dist={}\n'.format(dist))

				assert bead_i.label+" "+bead_j.label in self.condition_dictionary.keys(), 'error in reactions -- unrecognized condition label'

				sign, dist_value = self.condition_dictionary[bead_i.label+" "+bead_j.label]

				if sign == ">":

					if dist <= dist_value:

						return False

				if sign == ">=":

					if dist < dist_value:

						return False

				if sign == "<=":

					if dist > dist_value:

						return False

				if sign == "<":

					if dist >= dist_value:

						return False

		return True

	#-------------------------------------------------------------------------------

	def _reaction_effect(self, ntuple, beads, pointers):

		self.reaction_count += 1

		if self.effect_string == "end_simulation":

			self.end_simulation = True

	#-------------------------------------------------------------------------------

	def _unique_ntuples(self, indices):

		ntuples = list( product(indices, repeat = self.number_of_substrates) )

		unique_ntuples = set( [ frozenset(ntuple) for ntuple in ntuples if len(frozenset(ntuple)) == self.number_of_substrates ] )

		return list( list(ntuple) for ntuple in unique_ntuples )

	#-------------------------------------------------------------------------------

	def _ntuple_is_reactive(self, ntuple, beads):

		from_ntuple = { name:0 for name in self.substrates }

		for index in ntuple:

			label = beads[index].label

			if label in from_ntuple.keys(): from_ntuple[ label ] += 1

		return self.substrates_count == from_ntuple

	#-------------------------------------------------------------------------------

	def __str__(self):

		string_template = 'Reaction: {}'

		return string_template.format(self.reaction_name + self.reaction_string)

	#-------------------------------------------------------------------------------

	def __repr__(self):

		return self.__str__()

#-------------------------------------------------------------------------------

def set_reactions(input_data):

	reactions_for_simulation = []

	if "reaction_configuration_strings" in input_data.keys():

		for reaction_configuration_string in input_data["reaction_configuration_strings"]:

			assert len( reaction_configuration_string.split('|') ) == 4, 'incorrect reaction configuration string'

			reaction_name, reaction_string, condition_string, effect_string = reaction_configuration_string.split('|')

			reactions_for_simulation.append( Reactions(reaction_name, reaction_string, condition_string, effect_string) )

	return reactions_for_simulation

#-------------------------------------------------------------------------------