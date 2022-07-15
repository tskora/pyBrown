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

from scipy.constants import Boltzmann

from pyBrown.bead import compute_pointer_pbc_matrix, compute_pointer_immobile_pbc_matrix, check_overlaps, build_connection_matrix
from pyBrown.diffusion import RPY_M_matrix, RPY_Smith_M_matrix, JO_R_lubrication_correction_matrix
from pyBrown.interactions import set_interactions, kcal_per_mole_to_joule, eV_to_joule
from pyBrown.output import timing
from pyBrown.plane import Plane
from pyBrown.reactions import set_reactions

#-------------------------------------------------------------------------------

class Box():
	"""This is a class representing simulation box and allowing to propagate dynamics.

	Constructor method

	:param beads: list of bead objects
	:type beads: class: `list` of objects of class: `Bead`
	:param input_data: class: `dictionary`
	:type input_data: keyword-value pairs
	"""

	def __init__(self, beads, input_data):

		self.beads = beads
		self.inp = input_data
		self.box_length = self.inp["box_length"]

		self._initialize_pseudorandom_number_generation()

		self._handle_bead_mobility()
		self.labels = self._handle_bead_labels()

		self._set_physical_constants()

		self.hydrodynamics = self.inp["hydrodynamics"]
		if self.hydrodynamics == "nohi":
			self._compute_Dff_matrix()
			self._decompose_D_matrix()

		if self.hydrodynamics == "rpy_smith" or self.hydrodynamics == "rpy_smith_lub":
			self._set_ewald_summation_parameters()

		if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":
			self.lubrication_cutoff = self.inp["lubrication_cutoff"]
			self.cichocki_correction = self.inp["cichocki_correction"]

		self.propagation_scheme = self.inp["propagation_scheme"]
		if self.propagation_scheme == "midpoint":
			self.m_midpoint = self.inp["m_midpoint"]

		self.overlaps = self.inp["check_overlaps"]

		if self.overlaps:
			self.connection_matrix = build_connection_matrix(self.beads)
			assert not self._check_overlaps(), 'starting configuration contains overlaps'

		self.is_energy = False
		if "enr_write_freq" in self.inp.keys():
			self.is_energy = True

		self._initialize_force()

		self._initialize_interactions()

		self._initialize_reactions()

		self._set_flux_measurement_parameters()

		self._set_concentration_measurement_parameters()

		if self.inp["hydrodynamics"] != "nohi" or len(self.interactions) > 0 or len(self.reactions) > 0:
			self.rij_is_needed = True
			if len(self.immobile_beads) > 0:
				self.rik_is_needed = True
			else:
				self.rik_is_needed = False
				self.rik = np.zeros((0,0))
		else:
			self.rij_is_needed = False
			self.rik_is_needed = False

		self.is_wall = False
		if "walls" in self.inp.keys():
			self.is_wall = True
			walls = self.inp["walls"]
			self.planes = []
			for index in range(len(walls["normal"])):
				self.planes.append( Plane(plane_point = walls["plane_point"][index], normal_vector = walls["normal"][index]) )
			print(self.planes)

	#-------------------------------------------------------------------------------

	def _refresh_box(self):

		self._handle_bead_mobility()
		self.labels = self._handle_bead_labels()

		self.hydrodynamics = self.inp["hydrodynamics"]
		if self.hydrodynamics == "nohi":
			self._compute_Dff_matrix()
			self._decompose_D_matrix()

		if self.overlaps:
			self.connection_matrix = build_connection_matrix(self.beads)
			assert not self._check_overlaps(), 'starting configuration contains overlaps'

		self._initialize_force()

		self._initialize_interactions()

		# self._initialize_reactions()

		if self.inp["hydrodynamics"] != "nohi" or len(self.interactions) > 0 or len(self.reactions) > 0:
			self.rij_is_needed = True
			if len(self.immobile_beads) > 0:
				self.rik_is_needed = True
			else:
				self.rik_is_needed = False
				self.rik = []
		else:
			self.rij_is_needed = False
			self.rik_is_needed = False

		if self.rij_is_needed: self._compute_rij_matrix()
		if self.rik_is_needed: self._compute_rik_matrix()

	#-------------------------------------------------------------------------------

	# @timing
	def propagate(self, dt, build_Dff = True, build_Dnf = True, cholesky = True):
		"""Single propagation of dynamics

		:param dt: timestep
		:type dt: `float`
		:param build_Dff: switch on/off building Rotne-Prager-Yamakawa tensor in this step, defaults to `True`
		:type build_Dff: `bool`
		:param build_Dnf: switch on/off computing lubrication correction in this step, defaults to `True`
		:type Dnf: `bool`
		:param cholesky: switch on/off performing Choleski decomposition in this step, defaults to `True`
		:type cholesky: `bool`
		"""

		if self.is_wall:
			self.crossed_wall = False
		if self.is_flux:
			self.net_flux = {label: 0 for label in self.mobile_labels}
		if self.is_concentration:
			self.concentration = {label: 0 for label in self.mobile_labels}

		if self.rij_is_needed: self._compute_rij_matrix()
		if self.rik_is_needed: self._compute_rik_matrix()

		self._check_for_reactions()

		if self.rij_is_needed and len(self.rij) == 0: return

		if self.hydrodynamics != "nohi":

			if build_Dff:
				self._compute_Dff_matrix()

			if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

				if build_Dnf:
					self._compute_Dtot_matrix()

			if cholesky:
				self._decompose_D_matrix()

		self._compute_forces()

		if self.propagation_scheme == "ermak": self._ermak_step(dt, build_Dff, build_Dnf)

		if self.propagation_scheme == "midpoint": self._midpoint_step(dt, build_Dff, build_Dnf, cholesky)

		self._keep_beads_in_box()

		if self.is_concentration: self._compute_concentration_in_region()

	#-------------------------------------------------------------------------------

	def _ermak_step(self, dt, build_Dff, build_Dnf):

		self._deterministic_step(dt)

		if self.is_wall:
			deterministic_lead_to_crossing = self.crossed_wall
			self._deterministic_step(dt, mult = -1)
			self.crossed_wall = False

		if self.inp["divergence_term"]:

			if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":
				self._compute_divergence_of_D_matrix(build_Dff, build_Dnf)
				self._translate_beads(self.divergence_term*dt)

		count_move_attempts = 0

		while True:

			count_move_attempts += 1

			self._generate_random_vector()

			self._stochastic_step(dt)

			if self.is_wall:
				if self.crossed_wall:
					self._stochastic_step(dt, mult = -1)
					self.crossed_wall = False
					continue
				if deterministic_lead_to_crossing:
					self._deterministic_step(dt)
					deterministic_still_lead_to_crossing = self.crossed_wall
					if deterministic_still_lead_to_crossing:
						self._deterministic_step(dt, mult = -1)
						self.crossed_wall = False
						continue

			if self.overlaps:

				if self._check_overlaps():
					self._stochastic_step(dt, mult = -1)
					if self.is_wall:
						if deterministic_lead_to_crossing:
							self._deterministic_step(dt, mult = -1)
					self.crossed_wall = False
				else:
					break

			else:

				break

			assert count_move_attempts < self.inp["max_move_attempts"], 'maximal number of move attepts exceeded'

		if self.inp["verbose"]: print('move attempts: {}'.format(count_move_attempts))

	#-------------------------------------------------------------------------------

	def _midpoint_step(self, dt, build_Dff, build_Dnf, cholesky):

		D0 = np.copy(self.D)

		B0 = np.copy(self.B)

		if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

			Mff0 = np.copy(self.Mff)

			Rff0 = np.copy(self.Rff)

			R0 = np.copy(self.R)

		self._deterministic_step(dt, mult = 1.0 / self.m_midpoint)

		count_move_attempts = 0

		while True:

			count_move_attempts += 1

			self._generate_random_vector()

			self._stochastic_step(dt, mult = 1.0 / self.m_midpoint)

			if self.is_wall:
				if self.crossed_wall:
					self._stochastic_step(dt, mult = -1.0 / self.m_midpoint)
					continue#here i am

			if self.overlaps:

				if self._check_overlaps():
					self._stochastic_step(dt, mult = -1.0 / self.m_midpoint)
					continue

			if self.rij_is_needed: self._compute_rij_matrix()

			if self.hydrodynamics != "nohi":

				if build_Dff:
					self._compute_Dff_matrix()

				if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

					if build_Dnf:
						self._compute_Dtot_matrix()

				if cholesky:
					self._decompose_D_matrix()

			deterministic_drift = self.m_midpoint * dt / ( 2 * self.kBT ) * ( self.D - D0 ) @ self.F

			stochastic_drift = self.m_midpoint / 2 * math.sqrt(2 * dt) * ( self.D @ np.linalg.inv( np.transpose(B0) ) - B0 ) @ self.N

			if self.inp["debug"]: print('stochastic drift: {}'.format(stochastic_drift))

			drift = stochastic_drift + deterministic_drift

			self.D = D0

			self.B = B0

			if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

				self.Mff = Mff0

				self.Rff = Rff0

				self.R = R0

			self._deterministic_step(dt, mult = 1.0 - 1.0 / self.m_midpoint)

			self._stochastic_step(dt, mult = 1.0 - 1.0 / self.m_midpoint)

			if self.inp["debug"]: print('DRIFT STEP\n')

			self._translate_beads(drift)

			if self.overlaps:

				if self._check_overlaps():
					self._deterministic_step(dt, mult = -(1.0 - 1.0 / self.m_midpoint))
					self._stochastic_step(dt, mult = -1)
					self._translate_beads(-drift)
				else:
					break

			else:

				break

			assert count_move_attempts < self.inp["max_move_attempts"], 'maximal number of move attepts exceeded'

		if self.inp["verbose"]: print('move attempts: {}'.format(count_move_attempts))

	#-------------------------------------------------------------------------------

	def _deterministic_step(self, dt, mult = 1.0):

		if self.hydrodynamics != "nohi":
			FX = dt / self.kBT * self.D @ self.F * mult
		else:
			FX = dt / self.kBT * self.D * self.F * mult

		if self.inp["debug"]: print('DETERMINISTIC STEP\n')

		self._translate_beads(FX)

	#-------------------------------------------------------------------------------

	def _stochastic_step(self, dt, mult = 1.0):

		if self.hydrodynamics == "nohi":
			BX = self.B * self.N * math.sqrt(2 * dt) * mult
		else:
			BX = self.B @ self.N * math.sqrt(2 * dt) * mult

		if self.inp["debug"]: print('STOCHASTIC STEP\n')

		self._translate_beads(BX)

	#-------------------------------------------------------------------------------

	def _translate_beads(self, vector):

		for i, bead in enumerate( self.mobile_beads ):
			if self.is_flux:
				self.net_flux[bead.label] += bead.translate_and_return_flux( vector[3 * i: 3 * (i + 1)], self.planes )
			elif self.is_wall:
				crossed_wall_now = bead.translate_and_check_for_plane_crossing( vector[3 * i: 3 * (i + 1)], self.planes, self.inp["overlap_radius"] )
				self.crossed_wall = self.crossed_wall or crossed_wall_now
			else: bead.translate( vector[3 * i: 3 * (i + 1)] )

		if self.inp["debug"]: print('translation vector: {}\n'.format(vector.tolist()))

	#-------------------------------------------------------------------------------

	def _generate_random_vector(self):

		self.N = self.pseudorandom_number_generator.normal(0.0, 1.0, 3 * len(self.mobile_beads))

		self.draw_count += 3 * len(self.mobile_beads)

		if self.inp["debug"]: print('random vector: {}\n'.format(self.N.tolist()))

	#-------------------------------------------------------------------------------

	# @timing
	def _compute_forces(self):

		self.E = 0.0

		self._prepare_external_force()

		for interaction in self.interactions:

			self.E += interaction.compute_forces_and_energy(mobile_beads = self.mobile_beads, pointers_mobile = self.rij, F = self.F, immobile_beads = self.immobile_beads, pointers_mobile_immobile = self.rik)

		if self.inp["energy_unit"] == "joule":

			pass

		elif self.inp["energy_unit"] == "kcal/mol":

			# self.E = kcal_per_mole_to_joule(self.E)

			self.F = kcal_per_mole_to_joule(self.F)

		elif self.inp["energy_unit"] == "eV":

			# self.E = eV_to_joule(self.E)

			self.F = eV_to_joule(self.F)

	#-------------------------------------------------------------------------------

	def _check_for_reactions(self):

		for reaction in self.reactions:

			reaction.check_for_reactions(mobile_beads = self.mobile_beads, immobile_beads = self.immobile_beads, pointers_mobile = self.rij, pointers_mobile_immobile = self.rik)

			if reaction.end_simulation:

				self.end_simulation = True

				self.which_reaction_happened = reaction.reaction_name

				break

			if reaction.refresh_box:

				self._refresh_box()

	#-------------------------------------------------------------------------------

	def _prepare_external_force(self):

		if self.is_external_force_region:

			self._prepare_region_dependent_external_force()

		else:

			self.F = np.array(list(self.Fex)*(len(self.mobile_beads)))

	#-------------------------------------------------------------------------------

	# @timing
	def _check_overlaps(self):

		return check_overlaps(self.beads, self.box_length, self.inp["overlap_treshold"], self.connection_matrix, self.inp["overlap_radius"])

	#-------------------------------------------------------------------------------

	# @timing
	def _compute_rij_matrix(self):

		self.rij = compute_pointer_pbc_matrix(self.mobile_beads, self.box_length)

	#-------------------------------------------------------------------------------

	# @timing
	def _compute_rik_matrix(self):

		self.rik = compute_pointer_immobile_pbc_matrix(self.mobile_beads, self.immobile_beads, self.box_length)

	#-------------------------------------------------------------------------------

	# @timing
	def _compute_Dff_matrix(self):

		if self.hydrodynamics == "nohi":

			self.D = self.kBT * 10**19 / 6 / np.pi / np.array( [ self.mobile_beads[i//3].a for i in range(3*len(self.mobile_beads)) ] ) / self.viscosity

			if self.inp["debug"]: print('far field diffusion matrix: {}\n'.format(self.D.tolist()))

		elif self.hydrodynamics == "rpy":

			self.D = RPY_M_matrix(self.mobile_beads, self.rij)

			self.D *= self.kBT * 10**19 / self.viscosity

			if self.inp["debug"]: print('far field diffusion matrix: {}\n'.format(self.D.tolist()))

		elif self.hydrodynamics == "rpy_smith":

			self.D = RPY_Smith_M_matrix(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max)

			self.D *= self.kBT * 10**19 / self.viscosity

			if self.inp["debug"]: print('far field diffusion matrix: {}\n'.format(self.D.tolist()))

		elif self.hydrodynamics == "rpy_lub":

			self.Mff = RPY_M_matrix(self.mobile_beads, self.rij) * 10**19 / self.viscosity

			self.Rff = np.linalg.inv( self.Mff )

			if self.inp["debug"]: print('far field mobility matrix: {}\n'.format(self.Mff.tolist()))

			if self.inp["debug"]: print('far field resistance matrix: {}\n'.format(self.Rff.tolist()))

		elif self.hydrodynamics == "rpy_smith_lub":

			self.Mff = RPY_Smith_M_matrix(self.mobile_beads, self.rij, self.box_length, self.alpha, self.m_max, self.n_max) * 10**19 / self.viscosity

			self.Rff = np.linalg.inv( self.Mff )

			if self.inp["debug"]: print('far field mobility matrix: {}\n'.format(self.Mff.tolist()))

			if self.inp["debug"]: print('far field resistance matrix: {}\n'.format(self.Rff.tolist()))

	#-------------------------------------------------------------------------------

	# @timing
	def _compute_Dtot_matrix(self):

		self.R = JO_R_lubrication_correction_matrix(self.mobile_beads, self.rij, self.lubrication_cutoff, self.cichocki_correction) * self.viscosity / 10**19 + self.Rff

		self.D = self.kBT * np.linalg.inv(self.R)

		if self.inp["debug"]: print('total resistance matrix: {}\n'.format(self.R.tolist()))

		if self.inp["debug"]: print('total diffusion matrix: {}\n'.format(self.D.tolist()))

	#-------------------------------------------------------------------------------

	# @timing
	def _decompose_D_matrix(self):

		if self.hydrodynamics == "nohi":

			self.B = np.sqrt( self.D )

		else:

			self.B = np.linalg.cholesky(self.D)

			if self.inp["debug"]: print('decomposed diffusion matrix: {}\n'.format(self.B.tolist()))

	#-------------------------------------------------------------------------------

	def _compute_divergence_of_D_matrix(self, build_Dff, build_Dnf):

		eps = 0.0000001

		D0 = np.copy(self.D)

		if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

			Mff0 = np.copy(self.Mff)

			Rff0 = np.copy(self.Rff)

			R0 = np.copy(self.R)

		self.divergence_term = -np.sum( self.D, axis = 1 )

		for i, bead in enumerate( self.mobile_beads ):

			for j, vector in enumerate( [ [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps] ] ):

				vector = np.array(vector)

				bead.translate(vector)

				self._compute_rij_matrix()

				if self.hydrodynamics != "nohi":

					if build_Dff:
						self._compute_Dff_matrix()

					if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

						if build_Dnf:
							self._compute_Dtot_matrix()

				self.divergence_term += self.D[:][3*i + j]

				bead.translate(-vector)

		self.D = D0

		if self.hydrodynamics == "rpy_lub" or self.hydrodynamics == "rpy_smith_lub":

			self.Mff = Mff0

			self.Rff = Rff0

			self.R = R0

		self.divergence_term /= eps

	#-------------------------------------------------------------------------------

	def _keep_beads_in_box(self):

		for i, bead in enumerate( self.mobile_beads ):
			bead.keep_in_box(self.box_length)

	#-------------------------------------------------------------------------------

	def _initialize_pseudorandom_number_generation(self):

		if self.inp["seed"] is None:
			self.seed = np.random.randint(2**32 - 1)
		else:
			self.seed = self.inp["seed"]

		self.pseudorandom_number_generator = np.random.RandomState(self.seed)
		self.draw_count = 0

	#-------------------------------------------------------------------------------

	def _set_physical_constants(self):

		self.T = self.inp["T"]
		self.kBT = Boltzmann*self.T
		self.viscosity = self.inp["viscosity"]

	#-------------------------------------------------------------------------------

	def _set_ewald_summation_parameters(self):

		self.alpha = self.inp["ewald_alpha"]
		self.m_max = self.inp["ewald_real"]
		self.n_max = self.inp["ewald_imag"]

	#-------------------------------------------------------------------------------

	def _handle_bead_mobility(self):

		self.immobile_labels = self.inp["immobile_labels"]

		self.mobile_beads = []
		self.immobile_beads = []
		self.mobile_bead_indices = []
		self.immobile_bead_indices = []

		for i, bead in enumerate( self.beads ):
			if bead.label in self.immobile_labels:
				bead.mobile = False

		for i, bead in enumerate( self.beads ):
			if bead.mobile:
				self.mobile_beads.append(bead)
				self.mobile_bead_indices.append(i)
			else:
				self.immobile_beads.append(bead)
				self.immobile_bead_indices.append(i)

	#-------------------------------------------------------------------------------

	def _handle_bead_labels(self):

		self.labels = []
		self.mobile_labels = []

		for bead in self.beads:
			if bead.label not in self.labels:
				self.labels.append(bead.label)
				if bead.mobile == True:
					self.mobile_labels.append(bead.label)

	#-------------------------------------------------------------------------------

	def _handle_external_force_restricted_to_region(self):

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

	def _prepare_region_dependent_external_force(self):

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

	def _initialize_force(self):

		self.Fex = np.array( self.inp["external_force"] )
		self.F = np.zeros( 3*len(self.mobile_beads) )
		self.E = 0.0
		self._handle_external_force_restricted_to_region()

	#-------------------------------------------------------------------------------

	def _initialize_interactions(self):

		self.interactions = set_interactions(self.inp, self.beads)

	#-------------------------------------------------------------------------------

	def _initialize_reactions(self):

		self.reactions = set_reactions(self.inp)

		self.end_simulation = False

	#-------------------------------------------------------------------------------

	def _set_flux_measurement_parameters(self):

		self.is_flux = False
		if "measure_flux" in self.inp.keys():
			self.is_flux = True
			self.flux_normal = np.array(self.inp["measure_flux"]["normal"], float)
			self.flux_plane_point = np.array(self.inp["measure_flux"]["plane_point"], float)
			self.net_flux = {label: 0 for label in self.mobile_labels}

	#-------------------------------------------------------------------------------

	def _set_concentration_measurement_parameters(self):

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

	def _compute_concentration_in_region(self):

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

	def __str__(self):

		return 'simulation box'

	#-------------------------------------------------------------------------------
