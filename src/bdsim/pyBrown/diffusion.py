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

import ctypes
import numpy as np
import os.path	

from array import array
from ctypes.util import find_library

lib_name = "pyBrown"
lib_path = find_library(lib_name)
lib = ctypes.cdll.LoadLibrary( lib_path )

#-------------------------------------------------------------------------------

def RPY_M_matrix(beads, pointers):

	c_double = ctypes.c_double

	N = len(beads);

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = [pointers[i][j][k] for j in range(N) for i in range(j+1, N) for k in range(3)]
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	N_c = ctypes.c_int(N)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.RPY_M_matrix(a, p, N_c, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------

def RPY_Smith_M_matrix(beads, pointers, box_length, alpha, m, n):

	c_double = ctypes.c_double

	N = len(beads);

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = [pointers[i][j][k] for j in range(N) for i in range(j+1, N) for k in range(3)]
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	box_length_c = ctypes.c_double(box_length)

	alpha_c = ctypes.c_double(alpha)

	m_c = ctypes.c_int(m)

	n_c = ctypes.c_int(n)

	N_c = ctypes.c_int(N)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.RPY_Smith_M_matrix(a, p, box_length_c, alpha_c, m_c, n_c, N_c, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------

def JO_R_lubrication_correction_matrix(beads, pointers, lubrication_cutoff):

	c_double = ctypes.c_double

	N = len(beads);

	a_list = [b.a for b in beads]
	v0 = array('d', a_list)
	a = (c_double * len(v0)).from_buffer(v0)

	p_list = [pointers[i][j][k] for j in range(N) for i in range(j+1, N) for k in range(3)]
	v1 = array('d', p_list)
	p = (c_double * len(v1)).from_buffer(v1)

	N_c = ctypes.c_int(N)
	lubrication_cutoff_c = ctypes.c_double(lubrication_cutoff)

	len_my_list = 9*N*N

	my_list = [0.0]*len_my_list
	v2 = array('d', my_list)
	my_arr = (c_double * len(v2)).from_buffer(v2)

	lib.JO_R_lubrication_correction_matrix(a, p, N_c, lubrication_cutoff_c, my_arr)

	M = np.reshape(my_arr, (3*N, 3*N))

	return M

#-------------------------------------------------------------------------------