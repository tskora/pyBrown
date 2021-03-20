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

import math
import numpy as np

from scipy.constants import Boltzmann

from pyBD.bead import overlap_pbc, distance_pbc, pointer_pbc
from pyBD.diffusion import M_rpy, M_rpy_smith, R_lub_corr
from pyBD.output import timestamp, timing

#-------------------------------------------------------------------------------

class Box():

	def __init__(self, beads, input_data):

		self.beads = beads
		self.inp = input_data

		if "seed" in self.inp: np.random.seed(self.inp["seed"])

		self.mobile_beads = [ b for b in self.beads if b.mobile ]
		self.mobile_bead_indices = [ i for i, b in enumerate(self.beads) if b.mobile ]
		self.immobile_bead_indices = list( set([i for i in range(len(self.beads))]) - set(self.mobile_bead_indices) )
		
		self.box_length = self.inp["box_length"]
		self.T = self.inp["T"]
		self.viscosity = self.inp["viscosity"]
		self.hydrodynamics = self.inp["hydrodynamics"]
		self.Fex = np.array( self.inp["external_force"] )
		self.F0 = no.array( list(self.Fex)**len(self.mobile_beads) )

		if self.hydrodynamics == "nohi":
			self.D = Boltzmann * self.T * 10**19 / 6 / np.pi / np.array( [ self.mobile_beads[i//3].a for i in range(3*len(self.mobile_beads)) ] ) / self.viscosity
			self.B = np.sqrt( self.D )

		if self.hydrodynamics == "rpy_smith" or "rpy_smith_lub":
			self.alpha = self.inp["ewald_alpha"]
			self.m_max = self.inp["ewald_real"]
			self.n_max = self.inp["ewald_imag"]

	#-------------------------------------------------------------------------------

	def __str__(self):

		return 'a'

	#-------------------------------------------------------------------------------

	# @timing
	def propagate(self, dt, build_Dff = True, build_Dnf = True, cholesky = True, overlaps = True):

		if self.hydrodynamics != "nohi": self.compute_rij_matrix()

		if self.hydrodynamics == "rpy" or self.hydrodynamics == "rpy_smith":
			if build_Dff:
				self.compute_Dff_matrix()

		if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":
			if build_Dff:
				self.compute_Dff_matrix()
			if build_Dnf:
				self.compute_Dtot_matrix()

		if self.hydrodynamics != "nohi":
			if cholesky:
				self.decompose_D_matrix()

		# computing displacement due to external force
		if not np.all(self.Fex == 0.0):
			if self.hydrodynamics != "nohi":
				FX = dt / Boltzmann / self.T * self.D @ self.F0
			else:
				FX = dt / Boltzmann / self.T * self.D * self.F0

			# deterministic step
			for i, bead in enumerate( self.mobile_beads ):
				bead.translate( FX[3 * i: 3 * (i + 1)] )
		
		while True:

			# computing stochastic displacement
			if self.hydrodynamics == "nohi":
				BX = self.B * np.random.normal(0.0, 1.0, 3 * len(self.mobile_beads)) * math.sqrt(2 * dt)
			else:
				BX = self.B @ np.random.normal(0.0, 1.0, 3 * len(self.mobile_beads)) * math.sqrt(2 * dt)

			for i, bead in enumerate( self.mobile_beads ):
				# stochastic step
				bead.translate( BX[3 * i: 3 * (i + 1)] )

			if overlaps:

				if self.check_overlaps():
					for i, bead in enumerate( self.mobile_beads ):
						# undo stochastic step
						bead.translate( -BX[3 * i: 3 * (i + 1)] )
				else:
					for i, bead in enumerate( self.mobile_beads ):
						bead.keep_in_box(self.box_length)
					break

			else:

				break

	#-------------------------------------------------------------------------------

	# @timing
	def check_overlaps(self):

		overlaps = False

		for i in range(len(self.beads)-1):
			for j in range(i+1, len(self.beads)):
				pointer = self.beads[i].r - self.beads[j].r
				radii_sum = self.beads[i].a + self.beads[j].a
				radii_sum_pbc = self.box_length - radii_sum
				if ( pointer[0] > radii_sum and pointer[0] < radii_sum_pbc ) or ( pointer[0] < -radii_sum and pointer[0] > -radii_sum_pbc ):
					continue
				elif ( pointer[1] > radii_sum and pointer[1] < radii_sum_pbc ) or ( pointer[1] < -radii_sum and pointer[1] > -radii_sum_pbc ):
					continue
				elif ( pointer[2] > radii_sum and pointer[2] < radii_sum_pbc ) or ( pointer[2] < -radii_sum and pointer[2] > -radii_sum_pbc ):
					continue
				else:
					if overlap_pbc(self.beads[i], self.beads[j], self.box_length):
						return True

		return overlaps

	#-------------------------------------------------------------------------------

	# @timing
	def compute_rij_matrix(self):

		self.rij = np.zeros((len(self.mobile_beads), len(self.mobile_beads), 3))

		for i in range(1, len(self.mobile_beads)):
			for j in range(0, i):
				self.rij[i][j] = pointer_pbc(self.mobile_beads[i], self.mobile_beads[j], self.box_length)
				self.rij[j][i] = -self.rij[i][j]

	#-------------------------------------------------------------------------------

	# @timing
	def compute_Dff_matrix(self):

		if self.hydrodynamics == "rpy":

			self.D = M_rpy(self.mobile_beads, self.rij)

			self.D *= Boltzmann * self.T * 10**19 / self.viscosity

		if self.hydrodynamics == "rpy_smith":

			self.D = M_rpy_smith(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max)

			self.D *= Boltzmann * self.T * 10**19 / self.viscosity

		if self.hydrodynamics == "rpy_lub":

			self.Dff = M_rpy(self.mobile_beads, self.rij)

			self.Rff = np.linalg.inv( self.Dff )

		if self.hydrodynamics == "rpy_smith_lub":

			self.Dff = M_rpy_smith(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max)

			self.Rff = np.linalg.inv( self.Dff )

	#-------------------------------------------------------------------------------

	# @timing
	def compute_Dtot_matrix(self):

		self.R = R_lub_corr(self.mobile_beads, self.rij) + self.Rff

		self.D = Boltzmann * self.T * 10**19 * np.linalg.inv(self.R) / self.viscosity

	#-------------------------------------------------------------------------------

	# @timing
	def decompose_D_matrix(self):

		self.B = np.linalg.cholesky(self.D)

	#-------------------------------------------------------------------------------
