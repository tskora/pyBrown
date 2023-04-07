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

from pybrown.box import Box
from pybrown.input import read_str_file, InputData
from pybrown.output import timestamp, write_to_xyz_file, write_to_restart_file

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["output_xyz_filename", "input_str_filename",
						 "box_length", "T", "viscosity", "dt",
						 "reaction_configuration_strings"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	defaults = {"debug": False, "verbose": False, "hydrodynamics": "nohi",
				"ewald_alpha": np.sqrt(np.pi), "ewald_real": 0, "ewald_imag": 0,
				"diff_freq": 1, "lub_freq": 1, "chol_freq": 1, "xyz_write_freq": 1,
				"lubrication_cutoff": 1,
				"seed": None, "immobile_labels": [],
				"propagation_scheme": "ermak", "m_midpoint": 100,
				"divergence_term": False,
				"check_overlaps": True,
				"external_force": [0.0, 0.0, 0.0],
				"lennard_jones_6": False, "lennard_jones_12": False,
				"lennard_jones_alpha": 4.0, "energy_unit": "joule",
				"custom_interactions": False,
				"cichocki_correction": True,
				"overlap_treshold": 0.0, "max_move_attempts": 1000000}

	all_keywords = required_keywords + list(defaults.keys()) +\
				   [ "output_rst_filename", "rst_write_freq",
				     "filename_range",
				     "custom_interactions_filename",
				     "auxiliary_custom_interactions_keywords",
				     "external_force_region", "walls" ]

	timestamp( 'Reading input from {} file', input_filename )
	i = InputData(input_filename, required_keywords, defaults, all_keywords)
	timestamp( 'Input data:\n{}', i )

	str_filename = i.input_data["input_str_filename"]
	xyz_filename = i.input_data["output_xyz_filename"]

	if "filename_range" in i.input_data.keys():
		str_filenames = [ str_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
	else:
		str_filenames = [ str_filename ]
		xyz_filenames = [ xyz_filename ]

	dt = i.input_data["dt"]
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

	pathway_count = {}
	for reaction_configuration_string in i.input_data["reaction_configuration_strings"]:
		rname = reaction_configuration_string.split('|')[0].strip()
		pathway_count[rname] = 0

	timestamp('pathway count: {}', pathway_count)

	for index in range(len(str_filenames)):

		timestamp( '{} job', str_filenames[index] )

		str_filename = str_filenames[index]
		xyz_filename = xyz_filenames[index]

		if restart:
			rst_filename = rst_filenames[index]

		start = time.time()

		bs = read_str_file(str_filename, dims = i.input_data["dimensions"])

		box = Box(bs, i.input_data)

		timestamp('Random seed: {}', box.seed)

		with ExitStack() as stack:

			xyz_file = stack.enter_context(open(xyz_filename, "w", buffering = 1))

			j = 0

			while True:

				if j % n_write == 0:
					write_to_xyz_file(xyz_file, xyz_filename, j, dt, box.beads, dims = i.input_data["dimensions"])

				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if restart:
					if j != 0 and j % n_restart == 0:
						write_to_restart_file(rst_filename, index, j, box, xyz_filename, [], pathway_count)

				if box.end_simulation:
					pathway_count[box.which_reaction_happened] += 1
					break

				j += 1

		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

		timestamp('pathway count: {}', pathway_count)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()