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
import pickle
import time

from contextlib import ExitStack
from tqdm import tqdm

from pyBrown.box import Box
from pyBrown.input import read_str_file, InputData
from pyBrown.output import timestamp, write_to_xyz_file, write_to_restart_file,\
						   write_to_con_file, write_to_enr_file, write_to_flux_file,\
						   truncate_output_file_during_restart

@click.command()
@click.argument('restart_filename',
				type = click.Path( exists = True ))
@click.option('-j', '--json-filename',
			  required = False,
			  type = click.Path( exists = True ))
def main(restart_filename, json_filename):

	timestamp( 'Reading restart from {} file', restart_filename )

	with open(restart_filename, 'rb') as restart_file:
		index_rst = pickle.load(restart_file)
		j_rst = pickle.load(restart_file)
		box_rst = pickle.load(restart_file)
		xyz_file_length_rst = pickle.load(restart_file)
		if box_rst.is_concentration: con_file_length_rst = pickle.load(restart_file)
		if box_rst.is_energy: enr_file_length_rst = pickle.load(restart_file)
		if box_rst.is_flux: flux_file_length_rst = pickle.load(restart_file)
		pathway_count = pickle.load(restart_file)

	input_data = box_rst.inp

	timestamp( 'Input data loaded from the restart file:\n{}', input_data )

	timestamp( 'Pathway count loaded from the restart file:\n{}', pathway_count )

	# if json_filename != None:
	# 	timestamp( 'Reading input patch from {} file', json_filename )
	# 	input_patch = InputData(json_filename).input_data
	# 	for key in input_patch.keys():
	# 		if key in input_data.keys():
	# 			timestamp('Keyword {} updated: from {} to {}', key, input_data[key], input_patch[key])
	# 		else:
	# 			timestamp('Keyword {} introduced: {}', key, input_patch[key])
	# 		input_data[key] = input_patch[key]

	# timestamp( 'Input data patched with the json file:\n{}', input_data )

	# 1/0

	str_filename = input_data["input_str_filename"]
	xyz_filename = input_data["output_xyz_filename"]
	rst_filename = input_data["output_rst_filename"]

	if "filename_range" in input_data.keys():
		str_filenames = [ str_filename.format(j) for j in range(*input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*input_data["filename_range"]) ]
		rst_filenames = [ rst_filename.format(j) for j in range(*input_data["filename_range"]) ]
	else:
		str_filenames = [ str_filename ]
		xyz_filenames = [ xyz_filename ]
		rst_filenames = [ rst_filename ]

	truncate_output_file_during_restart(xyz_filenames[index_rst], xyz_file_length_rst)

	dt = input_data["dt"]
	n_write = input_data["xyz_write_freq"]
	n_diff = input_data["diff_freq"]
	n_lub = input_data["lub_freq"]
	n_chol = input_data["chol_freq"]
	n_restart = input_data["rst_write_freq"]

	for index in range(index_rst, len(str_filenames)):

		extra_output_filenames = []

		str_filename = str_filenames[index]
		xyz_filename = xyz_filenames[index]
		rst_filename = rst_filenames[index]

		start = time.time()

		if index == index_rst:
			bs = box_rst.beads
			box = box_rst
			j0 = j_rst+1
			filemode = 'a'
		else:
			bs = read_str_file(str_filename)
			box = Box(bs, input_data)
			j0 = 0
			filemode = 'w'

		timestamp('Random seed: {}', box.seed)

		with ExitStack() as stack:

			xyz_file = stack.enter_context(open(xyz_filename, filemode, buffering = 1))

			j = j0

			while True:

				if j % n_write == 0:
					write_to_xyz_file(xyz_file, xyz_filename, j, dt, box.beads)

				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if j != 0 and j % n_restart == 0:
					write_to_restart_file(rst_filename, index, j, box, xyz_filename, extra_output_filenames, pathway_count)

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