#!/usr/bin/env python3

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

import click
import numpy as np
import time

from contextlib import ExitStack
from tqdm import tqdm

from pyBrown.box import Box
from pyBrown.input import read_str_file, InputData
from pyBrown.output import timestamp, write_to_xyz_file, write_to_restart_file,\
						   write_to_con_file, write_to_enr_file, write_to_flux_file

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["output_xyz_filename", "input_str_filename",
						 "box_length", "T", "viscosity", "dt", "number_of_steps"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	defaults = {"debug": False, "verbose": False, "hydrodynamics": "nohi",
				"ewald_alpha": np.sqrt(np.pi), "ewald_real": 0, "ewald_imag": 0,
				"diff_freq": 1, "lub_freq": 1, "chol_freq": 1, "xyz_write_freq": 1,
				"lubrication_cutoff": 1,
				"progress_bar": False,
				"seed": None, "immobile_labels": [],
				"propagation_scheme": "ermak", "m_midpoint": 100,
				"divergence_term": False,
				"check_overlaps": True,
				"external_force": [0.0, 0.0, 0.0],
				"lennard_jones_6": False, "lennard_jones_12": False,
				"lennard_jones_alpha": 4.0,
				"dlvo": False, "dielectric_constant": 78.54, "inverse_debye_length": 0.1,
				"energy_unit": "joule",
				"custom_interactions": False,
				"cichocki_correction": True,
				"overlap_treshold": 0.0, "overlap_radius": "hydrodynamic",
				"max_move_attempts": 1000000}

	all_keywords = required_keywords + list(defaults.keys()) +\
				   [ "output_rst_filename", "rst_write_freq",
				     "output_enr_filename", "enr_write_freq",
				     "filename_range", "custom_interactions_filename",
				     "auxiliary_custom_interactions_keywords",
				     "external_force_region", "measure_flux",
				     "measure_concentration", "walls" ]

	timestamp( 'Reading input from {} file', input_filename )
	i = InputData(input_filename, required_keywords, defaults, all_keywords)
	timestamp( 'Input data:\n{}', i )

	disable_progress_bar = not i.input_data["progress_bar"]

	if "measure_concentration" in i.input_data.keys():
		concentration = True
		con_filename = i.input_data["measure_concentration"]["output_concentration_filename"]
	else:
		concentration = False

	if "measure_flux" in i.input_data.keys():
		flux = True
		flux_filename = i.input_data["measure_flux"]["output_flux_filename"]
	else:
		flux = False

	str_filename = i.input_data["input_str_filename"]
	xyz_filename = i.input_data["output_xyz_filename"]

	if "filename_range" in i.input_data.keys():
		if concentration: con_filenames = [ con_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		if flux: flux_filenames = [ flux_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		str_filenames = [ str_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
	else:
		if concentration: con_filenames = [ con_filename ]
		if flux: flux_filenames = [ flux_filename ]
		str_filenames = [ str_filename ]
		xyz_filenames = [ xyz_filename ]

	dt = i.input_data["dt"]
	n_steps = i.input_data["number_of_steps"]
	n_write = i.input_data["xyz_write_freq"]
	n_diff = i.input_data["diff_freq"]
	n_lub = i.input_data["lub_freq"]
	n_chol = i.input_data["chol_freq"]

	if "rst_write_freq" in i.input_data.keys():
		restart = True
		n_restart = i.input_data["rst_write_freq"]
		rst_filename = i.input_data["output_rst_filename"]
		if "filename_range" in i.input_data.keys():
			rst_filenames = [ rst_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		else:
			rst_filenames = [ rst_filename ]
	else:
		restart = False

	if "enr_write_freq" in i.input_data.keys():
		energy = True
		n_enr = i.input_data["enr_write_freq"]
		enr_filename = i.input_data["output_enr_filename"]
		if "filename_range" in i.input_data.keys():
			enr_filenames = [ enr_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		else:
			enr_filenames = [ enr_filename ]
	else:
		energy = False


	for index in range(len(str_filenames)):

		timestamp( '{} job', str_filenames[index] )

		extra_output_filenames = []

		if concentration:
			con_filename = con_filenames[index]
			extra_output_filenames.append(con_filename)
		if energy:
			enr_filename = enr_filenames[index]
			extra_output_filenames.append(enr_filename)
		if flux:
			flux_filename = flux_filenames[index]
			extra_output_filenames.append(flux_filename)
		str_filename = str_filenames[index]
		xyz_filename = xyz_filenames[index]

		if restart:
			rst_filename = rst_filenames[index]

		start = time.time()

		bs = read_str_file(str_filename)

		box = Box(bs, i.input_data)

		timestamp('Random seed: {}', box.seed)

		with ExitStack() as stack:

			if concentration: con_file = stack.enter_context(open(con_filename, "w", buffering = 1))
			if energy: enr_file = stack.enter_context(open(enr_filename, "w", buffering = 1))
			if flux: flux_file = stack.enter_context(open(flux_filename, "w", buffering = 1))
			xyz_file = stack.enter_context(open(xyz_filename, "w", buffering = 1))

			for j in tqdm( range(n_steps), disable = disable_progress_bar ):

				if i.input_data["debug"]: print('STEP {}\n'.format(j))

				if j % n_write == 0:
					write_to_xyz_file(xyz_file, xyz_filename, j, dt, box.beads)

				if concentration:
					write_to_con_file(con_file, j, dt, box.concentration)

				if flux: 
					write_to_flux_file(flux_file, j, dt, box.net_flux)

				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if energy:
					if j % n_enr == 0:
						write_to_enr_file(enr_file, j, dt, box.E, i.input_data["energy_unit"])

				if restart:
					if j != 0 and j % n_restart == 0:
						write_to_restart_file(rst_filename, index, j, box, xyz_filename, extra_output_filenames)

	
		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()