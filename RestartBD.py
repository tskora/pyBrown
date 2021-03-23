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
import pickle
import shutil
import time

from tqdm import tqdm

from pyBD.output import timestamp

@click.command()
@click.argument('restart_filename',
				type = click.Path( exists = True ))
def main(restart_filename):

	timestamp( 'Reading restart from {} file', restart_filename )

	with open(restart_filename, 'rb') as restart_file:
		index_rst = pickle.load(restart_file)
		j_rst = pickle.load(restart_file)
		box_rst = pickle.load(restart_file)
		xyz_file_rst = pickle.load(restart_file)

	input_data = box_rst.inp

	str_filename = input_data["input_str_filename"]
	xyz_filename = input_data["output_xyz_filename"]
	rst_filename = input_data["output_rst_filename"]

	if "filename_range" in input_data.keys():
		str_filenames = [ str_filename.format(j) for j in range(*input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*input_data["filename_range"]) ]
		rst_filenames = [ rst_filename.format(j) for j in range(*i.input_data["filename_range"]) ]
	else:
		str_filenames = [ str_filename ]
		xyz_filenames = [ xyz_filename ]
		rst_filenames = [ rst_filename ]

	with open(xyz_filenames[index_rst], 'w') as new_output_file:
		new_output_file.write(xyz_file_rst)

	dt = input_data["dt"]
	n_steps = input_data["number_of_steps"]
	n_write = input_data["xyz_write_freq"]
	n_diff = input_data["diff_freq"]
	n_lub = input_data["lub_freq"]
	n_chol = input_data["chol_freq"]
	n_restart = input_data["rst_write_freq"]

	for index in range(index_rst, len(str_filenames)):

		str_filename = str_filenames[index]
		xyz_filename = xyz_filenames[index]
		rst_filename = rst_filenames[index]

		start = time.time()

		if index == index_rst:
			bs = box_rst.beads
			box = box_rst
			j0 = j_rst+1
			box.sync_seed()
			filemode = 'a'
		else:
			bs = read_str_file(str_filename)
			box = Box(bs, i.input_data)
			j0 = 0
			filemode = 'w'
	
		with open(xyz_filename, filemode, buffering = 1) as output_file:
			for j in tqdm( range(j0, n_steps) ):
			# for j in range(j0, n_steps):
				if j % n_write == 0:
					output_file.write('{}\n'.format(len(box.beads)))
					output_file.write('{} time [ps] {}\n'.format(xyz_filename, j*dt))
					for bead in box.beads:
						output_file.write('{} {} {} {}\n'.format(bead.label, *bead.r))
				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

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