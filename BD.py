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
import pickle
import shutil
import time

from tqdm import tqdm

from pyBD.box import Box
from pyBD.input import read_str_file, InputData
from pyBD.output import timestamp

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["box_length", "output_xyz_filename", "input_str_filename",
						 "dt", "T", "viscosity", "number_of_steps"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	defaults = {"debug": False, "hydrodynamics": "nohi", "external_force": [0.0, 0.0, 0.0],
				"ewald_alpha": np.sqrt(np.pi), "ewald_real": 0, "ewald_imag": 0, "diff_freq": 1,
				"lub_freq": 1, "chol_freq": 1, "xyz_write_freq": 1, "progress_bar": False,
				"seed": np.random.randint(2**32 - 1), "immobile_labels": []}

	timestamp( 'Reading input from {} file', input_filename )
	i = InputData(input_filename, required_keywords, defaults)
	timestamp( 'Input data:\n{}', i )

	disable_progress_bar = not i.input_data["progress_bar"]

	if "measure_flux" in i.input_data.keys():
		flux = True
		n_flux = i.input_data["flux_freq"]
		flux_filename = i.input_data[""]

	str_filename = i.input_data["input_str_filename"]
	xyz_filename = i.input_data["output_xyz_filename"]

	if "filename_range" in i.input_data.keys():
		str_filenames = [ str_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
	else:
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

	for index in range(len(str_filenames)):

		str_filename = str_filenames[index]
		xyz_filename = xyz_filenames[index]

		if restart:
			rst_filename = rst_filenames[index]

		start = time.time()

		bs = read_str_file(str_filename)

		box = Box(bs, i.input_data)
	
		with open(xyz_filename, 'w', buffering = 1) as output_file:
			for j in tqdm( range(n_steps), disable = disable_progress_bar ):
				if j % n_write == 0:
					output_file.write('{}\n'.format(len(box.beads)))
					output_file.write('{} time [ps] {}\n'.format(xyz_filename, j*dt))
					for bead in box.beads:
						output_file.write('{} {} {} {}\n'.format(bead.label, *bead.r))
				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if restart:
					if j != 0 and j % n_restart == 0:
						with open(rst_filename, 'wb', buffering = 0) as restart_file:
							pickle.dump(index, restart_file)
							pickle.dump(j, restart_file)
							pickle.dump(box, restart_file)
							with open(xyz_filename, "r") as copied_file:
								pickle.dump(copied_file.read(), restart_file)
						shutil.copy(rst_filename, rst_filename+"2")

	
		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()