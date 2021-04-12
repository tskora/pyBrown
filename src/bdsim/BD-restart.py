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
import os
import shutil
import time

from contextlib import ExitStack
from tqdm import tqdm

from BD import write_to_xyz_file, write_to_restart_file, write_to_con_file, write_to_flux_file

from pyBrown.box import Box
from pyBrown.input import read_str_file
from pyBrown.output import timestamp

@click.command()
@click.argument('restart_filename',
				type = click.Path( exists = True ))
def main(restart_filename):

	timestamp( 'Reading restart from {} file', restart_filename )

	with open(restart_filename, 'rb') as restart_file:
		index_rst = pickle.load(restart_file)
		j_rst = pickle.load(restart_file)
		box_rst = pickle.load(restart_file)
		xyz_file_length_rst = pickle.load(restart_file)
		if box_rst.is_concentration: con_file_length_rst = pickle.load(restart_file)
		if box_rst.is_flux: flux_file_length_rst = pickle.load(restart_file)

	input_data = box_rst.inp

	timestamp( 'Input data:\n{}', input_data )

	disable_progress_bar = not input_data["progress_bar"]

	if box_rst.is_concentration:
		concentration = True
	else:
		concentration = False

	if box_rst.is_flux:
		flux = True
		n_flux = input_data["measure_flux"]["flux_freq"] # is it needed? it does not work yet anyways
	else:
		flux = False

	if concentration: con_filename = input_data["measure_concentration"]["output_concentration_filename"]
	if flux: flux_filename = input_data["measure_flux"]["output_flux_filename"]
	str_filename = input_data["input_str_filename"]
	xyz_filename = input_data["output_xyz_filename"]
	rst_filename = input_data["output_rst_filename"]

	if "filename_range" in input_data.keys():
		if concentration: con_filenames = [ con_filename.format(j) for j in range(*input_data["filename_range"]) ]
		if flux: flux_filenames = [ flux_filename.format(j) for j in range(*input_data["filename_range"]) ]
		str_filenames = [ str_filename.format(j) for j in range(*input_data["filename_range"]) ]
		xyz_filenames = [ xyz_filename.format(j) for j in range(*input_data["filename_range"]) ]
		rst_filenames = [ rst_filename.format(j) for j in range(*input_data["filename_range"]) ]
	else:
		if concentration: con_filenames = [ con_filename ]
		if flux: flux_filenames = [ flux_filename ]
		str_filenames = [ str_filename ]
		xyz_filenames = [ xyz_filename ]
		rst_filenames = [ rst_filename ]


	if concentration: truncate_file(con_filenames[index_rst], con_file_length_rst)
	if flux: truncate_file(flux_filenames[index_rst], flux_file_length_rst)
	truncate_file(xyz_filenames[index_rst], xyz_file_length_rst)

	dt = input_data["dt"]
	n_steps = input_data["number_of_steps"]
	n_write = input_data["xyz_write_freq"]
	n_diff = input_data["diff_freq"]
	n_lub = input_data["lub_freq"]
	n_chol = input_data["chol_freq"]
	n_restart = input_data["rst_write_freq"]

	for index in range(index_rst, len(str_filenames)):

		extra_output_filenames = []

		if concentration:
			con_filename = con_filenames[index]
			extra_output_filenames.append(con_filename)
		if flux:
			flux_filename = flux_filenames[index]
			extra_output_filenames.append(flux_filename)
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
			box = Box(bs, input_data)
			j0 = 0
			filemode = 'w'

		with ExitStack() as stack:

			if concentration: con_file = stack.enter_context(open(con_filename, filemode, buffering = 1))
			if flux: flux_file = stack.enter_context(open(flux_filename, filemode, buffering = 1))
			xyz_file = stack.enter_context(open(xyz_filename, filemode, buffering = 1))

			for j in tqdm( range(j0, n_steps), disable = disable_progress_bar ):
				if j % n_write == 0:
					write_to_xyz_file(xyz_file, xyz_filename, j, dt, box.beads)

				if concentration:
					write_to_con_file(con_file, j, dt, box.concentration)

				if flux: 
					if j % n_flux == 0:
						write_to_flux_file(flux_file, j, dt, box.net_flux)

				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if j != 0 and j % n_restart == 0:
					write_to_restart_file(rst_filename, index, j, box, xyz_filename, extra_output_filenames)
	
		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

def truncate_file(filename, line):
    
    tmp_filename = filename+'.tmp'
    
    with open(filename, 'r') as source:
        with open(tmp_filename, 'w') as destination:
            for i in range(line):
                destination.write( source.readline() )
                
    with open(filename, 'w') as destination:
        with open(tmp_filename, 'r') as source:
            for line in source:
                destination.write(line)
                
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()