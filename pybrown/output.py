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

import os
import pickle
import shutil
import time

#-------------------------------------------------------------------------------

def timestamp(message, *variables):

	date = time.asctime( time.localtime(time.time()) )

	print( '{}: '.format(date) + message.format(*variables) + '\n' )

#-------------------------------------------------------------------------------

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

#-------------------------------------------------------------------------------

def truncate_output_file_during_restart(filename, line):
    
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

def write_to_xyz_file(xyz_file, xyz_filename, j, dt, beads, dims = 3):

	xyz_file.write('{}\n'.format(len(beads)))
	xyz_file.write('{} time [ps] {}\n'.format(xyz_filename, j*dt))
	for bead in beads:
		if dims == 3: xyz_file.write('{} {} {} {}\n'.format(bead.label, *bead.r))
		elif dims == 2: xyz_file.write('{} {} {} 0.0\n'.format(bead.label, *bead.r))
		else: 1/0

#-------------------------------------------------------------------------------

def write_to_restart_file(restart_filename, index, j, box, xyz_filename, extra_output_filenames = [], extra_data = None):

	with open(restart_filename, 'wb', buffering = 0) as restart_file:
		pickle.dump(index, restart_file)
		pickle.dump(j, restart_file)
		pickle.dump(box, restart_file)
		pickle.dump(_file_length(xyz_filename), restart_file)
		for extra_output_filename in extra_output_filenames:
			pickle.dump(_file_length(extra_output_filename), restart_file)
		if extra_data != None:
			pickle.dump(extra_data, restart_file)

	shutil.copy(restart_filename, restart_filename+"2")

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

def write_to_enr_file(enr_file, j, dt, E, energy_unit_string):

	if j == 0:

		first_line_string = 'time/ps energy/{}\n'.format(energy_unit_string)

		enr_file.write(first_line_string)

	line_string = '{} {}\n'

	enr_file.write(line_string.format(j*dt, E))

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

def _file_length(filename):
	i = -1
	with open(filename, "r") as file:
		for i, l in enumerate(file):
			pass
	return i + 1

#-------------------------------------------------------------------------------