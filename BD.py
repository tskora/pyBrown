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

import click
import numpy as np
import time

from scipy.constants import Boltzmann
from tqdm import tqdm

from box import Box
from input import read_str_file, InputData
from output import timestamp

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["box_length", "output_xyz_filename", "input_str_filename",
						 "dt", "T", "viscosity",
						 "number_of_steps", "xyz_write_freq", "diff_freq", "lub_freq",
						 "chol_freq"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	# defaults = {"minimal_distance_between_surfaces":0.0, "max_bond_lengths":2.5e+07,
	# 			"bond_lengths":'hydrodynamic_radii', "number_of_structures":1,
	# 			"float_type": 32}
	defaults = {"debug": False, "hydrodynamics": "nohi", "external_force": [0.0, 0.0, 0.0],
				"ewald_alpha": np.sqrt(np.pi), "ewald_real": 0, "ewald_imag": 0}

	timestamp( 'Reading input from {} file', input_filename )
	i = InputData(input_filename, required_keywords, defaults)
	timestamp( 'Input data:\n{}', i )

	str_filename = i.input_data["input_str_filename"]
	xyz_filename = i.input_data["output_xyz_filename"]
	dt = i.input_data["dt"]
	
	n_steps = i.input_data["number_of_steps"]
	n_write = i.input_data["xyz_write_freq"]
	n_diff = i.input_data["diff_freq"]
	n_lub = i.input_data["lub_freq"]
	n_chol = i.input_data["chol_freq"]

	bs = read_str_file(str_filename)

	box = Box(bs, i.input_data)
	
	with open(xyz_filename, 'w') as output_file:
		start = time.time()
		for i in tqdm( range(n_steps) ):
		# for i in range(n_steps):
			if i % n_write == 0:
				output_file.write('{}\n'.format(len(box.beads)))
				output_file.write('{} time [ps] {}\n'.format(xyz_filename, i*dt))
				for bead in box.beads:
					output_file.write('{} {} {} {}\n'.format(bead.label, *bead.r))
			box.propagate(dt, i%n_diff == 0, i%n_lub == 0, i%n_chol == 0)
	
	end = time.time()
	
	print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()