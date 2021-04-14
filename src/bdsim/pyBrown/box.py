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
import sys

from scipy.constants import Boltzmann

from pyBrown.bead import overlap_pbc, distance_pbc, pointer_pbc
from pyBrown.diffusion import RPY_M_matrix, RPY_Smith_M_matrix, JO_R_lubrication_correction_matrix
from pyBrown.output import timestamp, timing

#-------------------------------------------------------------------------------

class Box():

	def __init__(self, beads, input_data):

		self.beads = beads
		self.inp = input_data

		# handling pseudorandom number generation
		self.seed = self.inp["seed"]
		np.random.seed(self.seed)
		self.draw_count = 0

		self.immobile_labels = self.inp["immobile_labels"]
		self.handle_bead_mobility()
		self.labels = self.handle_bead_labels()

		# handling basic physical parameters
		self.box_length = self.inp["box_length"]
		self.T = self.inp["T"]
		self.kBT = Boltzmann*self.T
		self.viscosity = self.inp["viscosity"]

		self.hydrodynamics = self.inp["hydrodynamics"]
		if self.hydrodynamics == "nohi":
			self.D = self.kBT * 10**19 / 6 / np.pi / np.array( [ self.mobile_beads[i//3].a for i in range(3*len(self.mobile_beads)) ] ) / self.viscosity
			self.B = np.sqrt( self.D )
		if self.hydrodynamics == "rpy_smith" or self.hydrodynamics == "rpy_smith_lub":
			self.alpha = self.inp["ewald_alpha"]
			self.m_max = self.inp["ewald_real"]
			self.n_max = self.inp["ewald_imag"]

		self.propagation_scheme = self.inp["propagation_scheme"]
		if self.propagation_scheme == "midpoint":
			self.m_midpoint = self.inp["m_midpoint"]

		self.overlaps = self.inp["check_overlaps"]
		
		# handling external forces
		self.Fex = np.array( self.inp["external_force"] )
		self.F = np.array( list(self.Fex)*len(self.mobile_beads) )
		self.handle_external_force_restricted_to_region()

		# handling flow measurement
		self.is_flux = False
		if "measure_flux" in self.inp.keys():
			self.is_flux = True
			self.flux_normal = np.array(self.inp["measure_flux"]["normal"], float)
			self.flux_plane_point = np.array(self.inp["measure_flux"]["plane_point"], float)
			self.net_flux = {label: 0 for label in self.mobile_labels}

		self.is_concentration = False
		if "measure_concentration" in self.inp.keys():
			self.is_concentration = True
			if "x" in self.inp["measure_concentration"].keys():
				self.is_concentration_region_x = True
				self.concentration_region_x = self.inp["measure_concentration"]["x"]
			else:
				self.is_concentration_region_x = False
			if "y" in self.inp["measure_concentration"].keys():
				self.is_concentration_region_y = True
				self.concentration_region_y = self.inp["measure_concentration"]["y"]
			else:
				self.is_concentration_region_y = False
			if "z" in self.inp["measure_concentration"].keys():
				self.is_concentration_region_z = True
				self.concentration_region_z = self.inp["measure_concentration"]["z"]
			else:
				self.is_concentration_region_z = False
			self.concentration = {label: 0 for label in self.mobile_labels}

	#-------------------------------------------------------------------------------

	def __str__(self):

		return 'simulation box'

	#-------------------------------------------------------------------------------

	# @timing
	def propagate(self, dt, build_Dff = True, build_Dnf = True, cholesky = True):

		if self.is_flux:
			self.net_flux = {label: 0 for label in self.mobile_labels}
		if self.is_concentration:
			self.concentration = {label: 0 for label in self.mobile_labels}

		# for now distances are needed only in hydrodynamics, it will change with adding interbead potential
		if self.hydrodynamics != "nohi":

			self.compute_rij_matrix()

		if self.hydrodynamics != "nohi":

			if build_Dff:
				self.compute_Dff_matrix()

			if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

				if build_Dnf:
					self.compute_Dtot_matrix()

			if cholesky:
				self.decompose_D_matrix()

		self.compute_forces()

		if self.propagation_scheme == "ermak": self.ermak_step(dt)

		if self.propagation_scheme == "midpoint": self.midpoint_step(dt, build_Dff, build_Dnf, cholesky)

		self.keep_beads_in_box()

		if self.is_concentration: self.compute_concentration_in_region()

	#-------------------------------------------------------------------------------

	def ermak_step(self, dt):

		self.deterministic_step(dt)

		while True:

			self.generate_random_vector()

			self.stochastic_step(dt)

			if self.overlaps:

				if self.check_overlaps():
					self.stochastic_step(dt, mult = -1)
				else:
					break

			else:

				break

	#-------------------------------------------------------------------------------

	def midpoint_step(self, dt, build_Dff, build_Dnf, cholesky):

		D0 = np.copy(self.D)

		B0 = np.copy(self.B)

		while True:

			self.deterministic_step(dt, mult = 1.0 / self.m_midpoint)

			self.generate_random_vector()

			self.stochastic_step(dt, mult = 1.0 / self.m_midpoint)

			# for now distances are needed only in hydrodynamics, it will change with adding interbead potential
			if self.hydrodynamics != "nohi":

				self.compute_rij_matrix()

			if self.hydrodynamics != "nohi":

				if build_Dff:
					self.compute_Dff_matrix()

				if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

					if build_Dnf:
						self.compute_Dtot_matrix()

				if cholesky:
					self.decompose_D_matrix()

			deterministic_drift = self.m_midpoint * dt / ( 2 * self.kBT ) * ( self.D - D0 ) @ self.F

			stochastic_drift = self.m_midpoint / 2 * math.sqrt(2 * dt) * ( self.B - B0 ) @ self.N

			drift = stochastic_drift + deterministic_drift

			self.D = D0

			self.B = B0

			self.deterministic_step(dt, mult = 1.0 - 1.0 / self.m_midpoint)

			self.stochastic_step(dt, mult = 1.0 - 1.0 / self.m_midpoint)

			self.translate_beads(drift)

			if self.overlaps:

				if self.check_overlaps():
					self.deterministic_step(dt, mult = -1)
					self.stochastic_step(dt, mult = -1)
					self.translate_beads(-drift)
				else:
					break

			else:

				break

	#-------------------------------------------------------------------------------

	def sync_seed(self):

		np.random.seed(self.seed)

		np.random.normal(0.0, 1.0, self.draw_count)

	#-------------------------------------------------------------------------------

	def handle_bead_mobility(self):

		self.mobile_beads = []
		self.immobile_beads = []
		self.mobile_bead_indices = []
		self.immobile_bead_indices = []

		for i, bead in enumerate( self.beads ):

			if bead.label in self.immobile_labels:
				bead.mobile = False
				self.immobile_beads.append(bead)
				self.immobile_bead_indices.append(i)


			else:
				self.mobile_beads.append(bead)
				self.mobile_bead_indices.append(i)

	#-------------------------------------------------------------------------------

	def handle_bead_labels(self):

		self.labels = []
		self.mobile_labels = []

		for bead in self.beads:
			if bead.label not in self.labels:
				self.labels.append(bead.label)
				if bead.mobile == True:
					self.mobile_labels.append(bead.label)

	#-------------------------------------------------------------------------------

	def handle_external_force_restricted_to_region(self):

		self.is_external_force_region = False
		if "external_force_region" in self.inp.keys():
			self.is_external_force_region = True
			if "x" in self.inp["external_force_region"].keys():
				self.is_external_force_region_x = True
				self.Fex_region_x = self.inp["external_force_region"]["x"]
			else:
				self.is_external_force_region_x = False
			if "y" in self.inp["external_force_region"].keys():
				self.is_external_force_region_y = True
				self.Fex_region_y = self.inp["external_force_region"]["y"]
			else:
				self.is_external_force_region_y = False
			if "z" in self.inp["external_force_region"].keys():
				self.is_external_force_region_z = True
				self.Fex_region_z = self.inp["external_force_region"]["z"]
			else:
				self.is_external_force_region_z = False

	#-------------------------------------------------------------------------------

	def prepare_region_dependent_external_force(self):

		if self.is_external_force_region:
			for i, bead in enumerate(self.mobile_beads):
				if self.is_external_force_region_x:
					if bead.r[0] < self.Fex_region_x[0] or bead.r[0] > self.Fex_region_x[1]:
						self.F[3*i:3*i+3] = np.zeros(3)
						continue
				if self.is_external_force_region_y:
					if bead.r[1] < self.Fex_region_y[0] or bead.r[1] > self.Fex_region_y[1]:
						self.F[3*i:3*i+3] = np.zeros(3)
						continue
				if self.is_external_force_region_z:
					if bead.r[2] < self.Fex_region_z[0] or bead.r[2] > self.Fex_region_z[1]:
						self.F[3*i:3*i+3] = np.zeros(3)
						continue
				self.F[3*i:3*i+3] = self.Fex

	#-------------------------------------------------------------------------------

	# @timing
	def compute_forces(self):

		self.prepare_region_dependent_external_force()

		# more to come -- interactions

	#-------------------------------------------------------------------------------

	# @timing
	def deterministic_step(self, dt, mult = 1.0):

		# computing displacement due to external force
		if self.hydrodynamics != "nohi":
			FX = dt / self.kBT * self.D @ self.F * mult
		else:
			FX = dt / self.kBT * self.D * self.F * mult

		self.translate_beads(FX)

	#-------------------------------------------------------------------------------

	def generate_random_vector(self):

		self.N = np.random.normal(0.0, 1.0, 3 * len(self.mobile_beads))

		self.draw_count += 3 * len(self.mobile_beads)

	#-------------------------------------------------------------------------------

	# @timing
	def stochastic_step(self, dt, mult = 1.0):

		if self.hydrodynamics == "nohi":
			BX = self.B * self.N * math.sqrt(2 * dt) * mult
		else:
			BX = self.B @ self.N * math.sqrt(2 * dt) * mult

		self.translate_beads(BX)

	#-------------------------------------------------------------------------------

	def translate_beads(self, vector):

		for i, bead in enumerate( self.mobile_beads ):
			if self.is_flux: self.net_flux[bead.label] += bead.translate_and_return_flux( vector[3 * i: 3 * (i + 1)], self.flux_normal, self.flux_plane_point )
			else: bead.translate( vector[3 * i: 3 * (i + 1)] )

	#-------------------------------------------------------------------------------

	def keep_beads_in_box(self):

		for i, bead in enumerate( self.mobile_beads ):
			bead.keep_in_box(self.box_length)

	#-------------------------------------------------------------------------------

	def compute_concentration_in_region(self):

		for bead in self.mobile_beads:
			if self.is_concentration_region_x:
				if bead.r[0] < self.concentration_region_x[0] or bead.r[0] > self.concentration_region_x[1]:
					continue
			if self.is_concentration_region_y:
				if bead.r[1] < self.concentration_region_y[0] or bead.r[1] > self.concentration_region_y[1]:
					continue
			if self.is_concentration_region_z:
				if bead.r[2] < self.concentration_region_z[0] or bead.r[2] > self.concentration_region_z[1]:
					continue

			self.concentration[bead.label] += 1

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

			self.D = RPY_M_matrix(self.mobile_beads, self.rij)

			self.D *= self.kBT * 10**19 / self.viscosity

		if self.hydrodynamics == "rpy_smith":

			self.D = RPY_Smith_M_matrix(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max)

			self.D *= self.kBT * 10**19 / self.viscosity

		if self.hydrodynamics == "rpy_lub":

			self.Mff = RPY_M_matrix(self.mobile_beads, self.rij) * 10**19 / self.viscosity

			self.Rff = np.linalg.inv( self.Mff )

		if self.hydrodynamics == "rpy_smith_lub":

			self.Mff = RPY_Smith_M_matrix(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max) * 10**19 / self.viscosity

			self.Rff = np.linalg.inv( self.Mff )

	#-------------------------------------------------------------------------------

	# @timing
	def compute_Dtot_matrix(self):

		self.R = JO_R_lubrication_correction_matrix(self.mobile_beads, self.rij) * self.viscosity / 10**19 + self.Rff

		self.D = self.kBT * np.linalg.inv(self.R)

	#-------------------------------------------------------------------------------

	# @timing
	def decompose_D_matrix(self):

		self.B = np.linalg.cholesky(self.D)

	#-------------------------------------------------------------------------------
