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

#-------------------------------------------------------------------------------

class Interactions():

	def __init__(self, force_function, energy_function, auxiliary_force_parameters):

		self.force = force_function

		self.energy = energy_function

		self.auxiliary_forca_parameters = auxiliary_force_parameters

		# **kwargs as a method of giving arguments to forces

	def compute_forces_and_energy(self, beads, pointers, F, E):

		for i, beadi in enumerate(1, beads):

			beadi_params = ...

			for j in range(i-1):

				beadj = beads[j]

				pointerij = pointers[i][j]

				beadj_params = ...

				F[3*i:3*(i+1)] += self._compute_force(beadi, beadj, pointerij, **auxiliary_force_parameters)

				F[3*j:3*(j+1)] -= self._compute_force(beadi, beadj, pointerij, **auxiliary_force_parameters)

				E += self._compute_pair_energy(beadi, beadj, pointerij, **auxiliary_force_parameters)

	def _compute_force(self, bead1, bead2, pointer, **kwargs):

		return self.force(bead1, bead2, pointer, ...)

	def _compute_pair_energy(self, bead1, bead2, pointer, **kwargs):

		return self.energy(bead1, bead2, pointer, ...)

