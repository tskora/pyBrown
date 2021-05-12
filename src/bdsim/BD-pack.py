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
from scipy.constants import epsilon_0
from tqdm import tqdm

from pyBrown.box import Box
from pyBrown.input import read_str_file, InputData
from pyBrown.output import timestamp, write_to_str_file

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["output_str_filename", "input_str_filename",
						 "box_length", "T", "viscosity", "dt", "number_of_steps"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	defaults = {"debug": False, "verbose": False, "hydrodynamics": "nohi",
				"progress_bar": False, "seed": None,
				"propagation_scheme": "ermak", "check_overlaps": True,
				"lennard_jones_6": False, "lennard_jones_12": False,
				"lennard_jones_alpha": 4.0, "energy_unit": "joule",
				"custom_interactions": False, "coulomb": True,
				"dielectric_constant": epsilon_0, "immobile_labels": [],
				"external_force": [0.0, 0.0, 0.0]}

	all_keywords = required_keywords + list(defaults.keys()) +\
				   [ "filename_range"]

	timestamp( 'Reading input from {} file', input_filename )
	i = InputData(input_filename, required_keywords, defaults, all_keywords)
	timestamp( 'Input data:\n{}', i )

	disable_progress_bar = not i.input_data["progress_bar"]

	str_inp_filename = i.input_data["input_str_filename"]
	str_out_filename = i.input_data["output_str_filename"]

	if "filename_range" in i.input_data.keys():
		str_inp_filenames = [ str_inp_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		str_out_filenames = [ str_out_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
	else:
		str_inp_filenames = [ str_filename ]
		str_out_filenames = [ xyz_filename ]

	dt = i.input_data["dt"]
	n_steps = i.input_data["number_of_steps"]

	for index in range(len(str_inp_filenames)):

		timestamp( '{} job', str_inp_filenames[index] )

		extra_output_filenames = []

		str_inp_filename = str_inp_filenames[index]
		str_out_filename = str_out_filenames[index]

		start = time.time()

		bs = read_str_file(str_inp_filename)

		original_bead_radii = [ b.a for b in bs ]

		original_charges = [ b.charge for b in bs ]

		for k in range(len(bs)):
			bs[k].charge = 3.0

		box = Box(bs, i.input_data)

		timestamp('Random seed: {}', box.seed)

		for j in tqdm( range(n_steps), disable = disable_progress_bar ):
			if j%10 == 0:
				for k in range(len(bs)):
					bs[k].a = 10 * original_bead_radii[k] * ( j + 1.0 ) / n_steps
			box.propagate(dt)

		for i in range(len(bs)): bs[i].charge = original_charges[i]

		with open(str_out_filename, "w") as str_out_file:
			write_to_str_file(str_out_file, box.beads)
	
		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()