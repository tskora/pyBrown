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

		self.refresh_box = False

	#-------------------------------------------------------------------------------

	def check_for_reactions(self, mobile_beads, pointers_mobile, immobile_beads = [], pointers_mobile_immobile = []):

		self.refresh_box = False

		beads = mobile_beads + immobile_beads

		bead_indices = [ i for i in range(len(beads)) ]

		unique_ntuples = [ ntuple for ntuple in self._unique_ntuples(bead_indices)
						   if self._ntuple_is_reactive(ntuple, beads) ]

		for ntuple in unique_ntuples:

			if self._reaction_criterion(ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile): self._reaction_effect(ntuple, beads)

	#-------------------------------------------------------------------------------

	def _initialize_pseudorandom_number_generation(self, seed):

		if seed is None:
			self.seed = np.random.randint(2**32 - 1)
		else:
			self.seed = seed
		self.pseudorandom_number_generator = np.random.RandomState(self.seed)
		self.draw_count = 0

	#-------------------------------------------------------------------------------

	def _parse_reaction_string(self, reaction_string):

		self.reaction_string = reaction_string

		assert '->' in reaction_string, 'Invalid reaction string: no "->" symbol in reaction string'

		if '<->' in reaction_string:

			# not used yet

			lhs, rhs = reaction_string.split('<->')

			print('not implemented yet')

			1/0

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

		self.conditions = []

		self.condition_types = []

		for single_condition in condition_string.split(','):

			if single_condition.strip().split(' ')[0] == 'dist':

				self._parse_dist_condition(single_condition)

			elif single_condition.strip().split(' ')[0] == 'pole_dist':

				self._parse_pole_dist_condition(single_condition)

			elif single_condition.strip().split(' ')[0] == 'random':

				self._parse_random_condition(single_condition)

			else:

				print('unknown reaction condition')

				1/0

	#-------------------------------------------------------------------------------

	def _parse_dist_condition(self, single_condition):

		dist_condition_dictionary = {}

		_, label1, label2, sign, string_dist = single_condition.strip().split(' ')

		dist = float(string_dist)

		dist_condition_dictionary[label1+" "+label2] = (sign, dist)

		dist_condition_dictionary[label2+" "+label1] = (sign, dist)

		self.conditions.append(dist_condition_dictionary)

		self.condition_types.append("dist")

	#-------------------------------------------------------------------------------

	def _parse_pole_dist_condition(self, single_condition):

		pole_dist_condition_dictionary = {}

		_, label1, label2, label3, sign, string_dist = single_condition.strip().split(' ')

		dist = float(string_dist)

		pole_dist_condition_dictionary[label1+" "+label2+" "+label3] = (sign, dist)

		self.conditions.append(pole_dist_condition_dictionary)

		self.condition_types.append("pole_dist")

	#-------------------------------------------------------------------------------

	def _parse_angle_condition(self, single_condition):

		angle_condition_dictionary = {}

		_, label1, label2, label3, sign, string_angle = single_condition.strip().split(' ')

		angle = float(string_angle)

		angle_condition_dictionary[label1+" "+label2+" "+label3] = (sign, angle)

		# permutations???

		self.conditions.append(angle_condition_dictionary)

		self.condition_types.append("angle")

	#-------------------------------------------------------------------------------

	def _parse_random_condition(self, single_condition):

		if len( single_condition.strip().split(' ') ) == 3:

			_, probability, seed = single_condition.strip().split(' ')

		elif len( single_condition.strip().split(' ') ) == 2:

			_, probability = single_condition.strip().split(' ')

			probability = float(probability)

			seed = None

		else:

			print('error in parsing condition string')

			1/0

		self._initialize_pseudorandom_number_generation(seed)

		self.conditions.append(probability)

		self.condition_types.append("random")

	#-------------------------------------------------------------------------------

	def _parse_effect_string(self, effect_string):

		self.effect_string = effect_string

	#-------------------------------------------------------------------------------

	def _reaction_criterion(self, ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile):

		answer = True

		for i in range(len(self.conditions)):

			condition = self.conditions[i]

			condition_type = self.condition_types[i]

			if condition_type == "dist": answer = answer and self._reaction_criterion_dist(ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile, condition)

			elif condition_type == "pole_dist": answer = answer and self._reaction_criterion_pole_dist(ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile, condition)

			elif condition_type == "angle": answer = answer and self._reaction_criterion_angle(ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile, condition)

			elif condition_type == "random": answer = answer and self._reaction_criterion_random(condition)

			else: 1/0

		return answer

	#-------------------------------------------------------------------------------

	def _reaction_criterion_dist(self, ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile, condition):

		beads = mobile_beads + immobile_beads

		for i in range(1, len(ntuple)):

			index_i = ntuple[i]

			bead_i = beads[ntuple[i]]

			for j in range(i):

				index_j = ntuple[j]

				bead_j = beads[ntuple[j]]

				if bead_i.label+" "+bead_j.label not in condition.keys():

					continue

				pointer = _return_proper_pointer(index_i, index_j, pointers_mobile, pointers_mobile_immobile)

				if pointer is not None:

					dist2 = pointer[0]*pointer[0] + pointer[1]*pointer[1] + pointer[2]*pointer[2]

					dist = math.sqrt(dist2)

					# print('dist={}\n'.format(dist))

					sign, dist_value = condition[bead_i.label+" "+bead_j.label]

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

				else:

					if bead_j.bead_id in bead_i.bonded_with or bead_i.bead_id in bead_j.bonded_with:

						continue

					else:

						return False

		return True

	#-------------------------------------------------------------------------------

	def _reaction_criterion_pole_dist(self, ntuple, mobile_beads, immobile_beads, pointers_mobile, pointers_mobile_immobile, condition):

		beads = mobile_beads + immobile_beads

		for i in range(len(ntuple)):

			index_i = ntuple[i]

			bead_i = beads[ntuple[i]]

			for j in range(len(ntuple)):

				if j == i: continue

				index_j = ntuple[j]

				bead_j = beads[ntuple[j]]

				for k in range(len(ntuple)):

					if k == i or k == j: continue

					index_k = ntuple[k]

					bead_k = beads[ntuple[k]]

					if bead_i.label+" "+bead_j.label+" "+bead_k.label not in condition.keys():

						continue

					pointer1 = _return_proper_pointer(index_i, index_j, pointers_mobile, pointers_mobile_immobile)

					pointer2 = _return_proper_pointer(index_j, index_k, pointers_mobile, pointers_mobile_immobile)

					if pointer1 is not None and pointer2 is not None:

						shift = pointer1

						shift *= bead_j.a / np.linalg.norm(shift)

						p_pointer = pointer2 - shift

						p_dist = np.sqrt( p_pointer[0]*p_pointer[0] + p_pointer[1]*p_pointer[1] + p_pointer[2]*p_pointer[2] )

						sign, dist_value = condition[bead_i.label+" "+bead_j.label+" "+bead_k.label]

						if sign == ">":

							if p_dist <= dist_value:

								return False

						if sign == ">=":

							if p_dist < dist_value:

								return False

						if sign == "<=":

							if p_dist > dist_value:

								return False

						if sign == "<":

							if p_dist >= dist_value:

								return False

					else:

						1/0

		return True

	#-------------------------------------------------------------------------------

	def _reaction_criterion_random(self, probability):

		a = self.pseudorandom_number_generator.uniform(0.0, 1.0)

		self.draw_count += 1

		return ( a < probability )

	#-------------------------------------------------------------------------------

	def _reaction_effect(self, ntuple, beads):

		self.reaction_count += 1

		if self.effect_string == "end_simulation":

			self.end_simulation = True

		elif self.effect_string == "freeze":

			self._reaction_effect_freeze(ntuple, beads)

		elif self.effect_string == "unfreeze":

			self._reaction_effect_unfreeze(ntuple, beads)

		elif "bond" in self.effect_string:

			self._reaction_effect_bond(ntuple, beads)

		else:

			print('unknown reaction effect')

			1/0

	#-------------------------------------------------------------------------------

	def _reaction_effect_freeze(self, ntuple, beads):

		change_in_mobility = False

		for i in ntuple:

			change_in_mobility = change_in_mobility or beads[i].mobile

			beads[i].mobile = False

		if change_in_mobility:

			for i in ntuple:

				for j in ntuple:

					if i == j: continue

					if beads[j].bead_id in beads[i].bonded_with or beads[i].bead_id in beads[j].bonded_with:

						continue

					else:

						beads[i].bonded_with.append(beads[j].bead_id)

						beads[i].bonded_how[beads[j].bead_id] = (0.0, 0.0)

			self.refresh_box = True

		else:

			self.refresh_box = False

	#-------------------------------------------------------------------------------

	def _reaction_effect_unfreeze(self, ntuple, beads):

		change_in_mobility = False

		for i in ntuple:

			change_in_mobility = change_in_mobility or not beads[i].mobile

			beads[i].mobile = True

		if change_in_mobility:

			for i in ntuple:

				for j in ntuple:

					if i == j: continue

					if beads[j].bead_id in beads[i].bonded_with:

						if beads[i].bonded_how[beads[j].bead_id] == (0.0, 0.0):

							beads[i].bonded_with.remove(beads[j].bead_id)

							del beads[i].bonded_how[beads[j].bead_id]

					elif beads[i].bead_id in beads[j].bonded_with:

						if beads[j].bonded_how[beads[i].bead_id] == (0.0, 0.0):

							beads[j].bonded_with.remove(beads[i].bead_id)

							del beads[j].bonded_how[beads[i].bead_id]

					else:

						print('surprise')

						1/0

			self.refresh_box = True

		else:

			self.refresh_box = False

	#-------------------------------------------------------------------------------

	def _reaction_effect_bond(self, ntuple, beads):

		change_in_bonds = False

		string_segments = self.effect_string.split()
		dist_eq = float( string_segments[1] )
		force_constant = float( string_segments[3] )

		for i in range(1, len(ntuple)):

			for j in range(i):

				b1 = beads[i]

				b2 = beads[j]

				if b2.bead_id in b1.bonded_with or b1.bead_id in b2.bonded_with:

					continue

				else:

					change_in_bonds = change_in_bonds or True

				b1.bonded_with.append(b2.bead_id)

				b1.bonded_how[b2.bead_id] = [dist_eq, force_constant]

		if change_in_bonds:
			self.refresh_box = True
		else:
			self.refresh_box = False

	#-------------------------------------------------------------------------------

	def _unique_ntuples(self, indices):

		ntuples = list( product(indices, repeat = self.number_of_substrates) )

		unique_ntuples = set( [ frozenset(ntuple) for ntuple in ntuples if len(frozenset(ntuple)) == self.number_of_substrates ] )

		return list( list(ntuple) for ntuple in unique_ntuples )

	#-------------------------------------------------------------------------------

	def _ntuple_is_reactive(self, ntuple, beads):

		if len(ntuple) != self.number_of_substrates: return False

		if len(set(ntuple)) < len(ntuple): return False

		from_ntuple = { name:0 for name in self.substrates }

		for index in ntuple:

			label = beads[index].label

			if label in from_ntuple.keys(): from_ntuple[ label ] += 1

		return self.substrates_count == from_ntuple

	#-------------------------------------------------------------------------------

	def __str__(self):

		string_template = 'Reaction: {}'

		return string_template.format(self.reaction_name + ' ' + self.reaction_string)

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

def _return_proper_pointer(i, j, pointers_mobile, pointers_mobile_immobile):

	limitter = len(pointers_mobile)

	if i < limitter and j < limitter:

		return pointers_mobile[i][j]

	elif i < limitter and j >= limitter:

		j -= limitter

		return pointers_mobile_immobile[i][j]

	elif i >= limitter and j < limitter:

		i -= limitter

		return -pointers_mobile_immobile[j][i]

	else:

		return None

#-------------------------------------------------------------------------------