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
import pickle
import shutil
import time

from contextlib import ExitStack
from tqdm import tqdm

from pyBrown.box import Box
from pyBrown.input import read_str_file, InputData
from pyBrown.output import timestamp

@click.command()
@click.argument('input_filename',
				type = click.Path( exists = True ))
def main(input_filename):

	# here the list of keywords that are required for program to work is provided
	required_keywords = ["box_length", "output_xyz_filename", "input_str_filename",
						 "dt", "T", "viscosity", "number_of_steps"]

	# here the dict of keywords:default values is provided
	# if given keyword is absent in JSON, it is added with respective default value
	defaults = {"debug": False, "verbose": False, "hydrodynamics": "nohi",
				"ewald_alpha": np.sqrt(np.pi), "ewald_real": 0, "ewald_imag": 0, "diff_freq": 1,
				"lub_freq": 1, "chol_freq": 1, "xyz_write_freq": 1, "progress_bar": False,
				"seed": np.random.randint(2**32 - 1), "immobile_labels": [],
				"propagation_scheme": "ermak", "check_overlaps": True,
				"external_force": [0.0, 0.0, 0.0]}

	all_keywords = required_keywords + list(defaults.keys()) +\
				   [ "output_rst_filename", "filename_range", "rst_write_freq",
				     "m_midpoint", "external_force_region", "measure_flux" ]

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
		n_flux = i.input_data["measure_flux"]["flux_freq"] # is it needed? it does not work yet anyways
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

	for index in range(len(str_filenames)):

		timestamp( '{} job', str_filenames[index] )

		extra_output_filenames = []

		if concentration:
			con_filename = con_filenames[index]
			extra_output_filenames.append(con_filename)
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

		with ExitStack() as stack:

			if concentration: con_file = stack.enter_context(open(con_filename, "w", buffering = 1))
			if flux: flux_file = stack.enter_context(open(flux_filename, "w", buffering = 1))
			xyz_file = stack.enter_context(open(xyz_filename, "w", buffering = 1))

			for j in tqdm( range(n_steps), disable = disable_progress_bar ):
				if j % n_write == 0:
					write_to_xyz_file(xyz_file, xyz_filename, j, dt, box.beads)

				if concentration:
					write_to_con_file(con_file, j, dt, box.concentration)

				if flux: 
					if j % n_flux == 0:
						write_to_flux_file(flux_file, j, dt, box.net_flux)

				box.propagate(dt, j%n_diff == 0, j%n_lub == 0, j%n_chol == 0)

				if restart:
					if j != 0 and j % n_restart == 0:
						write_to_restart_file(rst_filename, index, j, box, xyz_filename, extra_output_filenames)

	
		end = time.time()
	
		print('{} seconds elapsed'.format(end-start))

#-------------------------------------------------------------------------------

def write_to_con_file(con_file, j, dt, concentration):

	concentration_labels = list(concentration.keys())

	if j == 0:

		first_line_string = 'time/ps' + ' {}' * len(concentration_labels) + '\n'

		con_file.write(first_line_string.format(*concentration_labels))

	else:

		line_string = '{}' + ' {}' * len(concentration) + '\n'

		concentration_for_given_label = [ concentration[key] for key in concentration_labels ]

		con_file.write(line_string.format(j*dt, *concentration_for_given_label))

#-------------------------------------------------------------------------------

def write_to_flux_file(flux_file, j, dt, net_flux):

	net_flux_labels = list(net_flux.keys())

	if j == 0:

		first_line_string = 'time/ps' + ' {}' * len(net_flux_labels) + '\n'

		flux_file.write(first_line_string.format(*net_flux_labels))

	else:

		line_string = '{}' + ' {}'*len(net_flux) + '\n'

		net_flux_for_given_label = [ net_flux[key] for key in net_flux_labels ]

		flux_file.write(line_string.format(j*dt, *net_flux_for_given_label))

#-------------------------------------------------------------------------------

def write_to_xyz_file(xyz_file, xyz_filename, j, dt, beads):

	xyz_file.write('{}\n'.format(len(beads)))
	xyz_file.write('{} time [ps] {}\n'.format(xyz_filename, j*dt))
	for bead in beads:
		xyz_file.write('{} {} {} {}\n'.format(bead.label, *bead.r))

#-------------------------------------------------------------------------------

def write_to_restart_file(restart_filename, index, j, box, xyz_filename, extra_output_filenames = []):

	with open(restart_filename, 'wb', buffering = 0) as restart_file:
		pickle.dump(index, restart_file)
		pickle.dump(j, restart_file)
		pickle.dump(box, restart_file)
		pickle.dump(file_length(xyz_filename), restart_file)
		for extra_output_filename in extra_output_filenames:
			pickle.dump(file_length(extra_output_filename), restart_file)

	shutil.copy(restart_filename, restart_filename+"2")

#-------------------------------------------------------------------------------

def file_length(filename):
	i = -1
	with open(filename, "r") as file:
		for i, l in enumerate(file):
			pass
	return i + 1

#-------------------------------------------------------------------------------

if __name__ == '__main__':
	
	main()